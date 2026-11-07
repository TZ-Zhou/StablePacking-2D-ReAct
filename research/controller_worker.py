"""The controller policy in its own process (multiprocessing spawn), spoken to in JSON bytes.

The main process (controllers.ControllerClient) sends a request (observation, base order, tool
table) and waits for the reply; only JSON crosses the pipe, so the worker holds no Wang object,
score matrix, global setting or RNG of the main process, and the main process receives nothing but
parsed JSON. Every stage is bounded: the start (spawn and the ready message) by start_timeout,
the request (sending it and receiving the reply) by one deadline of `timeout` seconds, and the
cleanup by reap_timeout. Any failure kills the worker (kill + join) and the next request starts a
new one:
- worker_timeout: the request was not taken or not answered in time (the worker is killed with
  whatever it was running);
- worker_error: the worker exited, could not start, or reported an exception of its policy;
- worker_protocol_error: a message over MAX_BYTES, not JSON, nested too deep, or not of the protocol.
A policy reply asking for a restart (an abandoned LLM call) also ends the worker after the reply.
`log` (if set) receives the lifecycle of each worker: start (pid) and close (pid, reason, exit
code, whether it was still alive after the reap timeout), as evidence that no worker outlived its
decision; the last worker is closed when the main process exits normally.
"""
import atexit
import json
import multiprocessing
import sys
import threading
import time
import weakref
from pathlib import Path

MAX_BYTES = 1 << 20


def summary(data):
    """A safe description of a bad message: its length and the first 200 characters."""
    return f'{len(data)} bytes: {data[:200].decode("utf-8", "replace")!r}'


def serve(conn, spec):
    """Worker main: build the policy of `spec` (JSON text), say ready, answer requests until EOF.
    Replies are {'ok': reply} or {'error': 'Type: message'}."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import controllers
    try:
        policy = controllers.make_policy(json.loads(spec))
    except Exception as error:
        conn.send_bytes(json.dumps({'error': controllers.error_text(error)}).encode())
        return
    conn.send_bytes(b'{"ready": true}')
    while True:
        try:
            request = json.loads(conn.recv_bytes())
        except EOFError:
            return
        try:
            out = json.dumps({'ok': policy.decide(request)}, allow_nan=False)
        except Exception as error:  # reported, not raised: the main process records the fallback
            out = json.dumps({'error': controllers.error_text(error)})
        conn.send_bytes(out.encode())


def close_at_exit(ref):
    transport = ref()
    if transport is not None:
        transport.close('main_exit')


class ProcessTransport:
    """decide(request) -> ('ok', reply) or (failure, safe description); see the module docstring.
    target: the worker main (tests pass misbehaving ones); it must be importable by name."""

    def __init__(self, spec, timeout=None, start_timeout=120.0, target=serve, reap_timeout=30.0):
        react = spec['kind'] == 'react'
        self.spec, self.target, self.start_timeout, self.reap_timeout = json.dumps(spec), target, start_timeout, reap_timeout
        self.timeout = timeout or (spec['llm_timeout'] * spec['max_tool_calls'] + 30 if react else 30.0)
        self.process = self.conn = self.sender = None
        self.started, self.log, self.exit_hook, self.pid = 0, None, False, None

    def note(self, entry):
        if self.log is not None:
            self.log({'type': 'worker', **entry})

    def receive(self, deadline):
        timeout = max(0.0, deadline - time.monotonic())
        if not self.conn.poll(timeout):
            return 'worker_timeout', f'no reply within {round(timeout, 3)} s'
        try:
            data = self.conn.recv_bytes(MAX_BYTES)
        except EOFError:
            self.process.join(5)
            return 'worker_error', f'the worker exited with code {self.process.exitcode}'
        except OSError:
            return 'worker_protocol_error', f'a message over {MAX_BYTES} bytes'
        try:
            message = json.loads(data)
        except (ValueError, RecursionError):
            return 'worker_protocol_error', f'not JSON: {summary(data)}'
        return 'ok', message

    def send(self, data, deadline):
        """send_bytes blocks while the worker does not read a request larger than the pipe buffer,
        so it runs in a thread the deadline can abandon; killing the worker then breaks the pipe
        and ends the thread."""
        box = {}

        def run():
            try:
                self.conn.send_bytes(data)
            except Exception as error:  # the worker is gone (broken pipe) or the pipe was closed
                box['error'] = error
        self.sender = threading.Thread(target=run, name='controller-send', daemon=True)
        self.sender.start()
        self.sender.join(max(0.0, deadline - time.monotonic()))
        if self.sender.is_alive():
            return 'worker_timeout', 'the worker did not take the request in time'
        if 'error' in box:
            return 'worker_error', f'cannot send the request: {type(box["error"]).__name__}'
        return 'ok', None

    def start(self):
        try:
            context = multiprocessing.get_context('spawn')
            self.conn, child = context.Pipe()
            try:
                self.process = context.Process(target=self.target, args=(child, self.spec), name='controller-worker', daemon=True)
                self.process.start()
            finally:
                child.close()
        except Exception as error:  # cannot spawn: clean both pipe ends and any half-built process
            from controllers import error_text
            self.close('start_failed')
            return 'worker_error', f'cannot start the worker: {error_text(error)}'
        self.started, self.pid = self.started + 1, self.process.pid
        self.note({'action': 'start', 'pid': self.process.pid, 'n': self.started})
        if not self.exit_hook:  # the last worker ends with the main process, and says so (a weak
            atexit.register(close_at_exit, weakref.ref(self))  # reference: a dropped transport's worker gets EOF)
            self.exit_hook = True
        status, message = self.receive(time.monotonic() + self.start_timeout)
        if status == 'ok' and message != {'ready': True}:
            status, message = 'worker_error', (message['error'] if isinstance(message, dict) and isinstance(message.get('error'), str)
                                               else 'the worker did not start')
        return status, message

    def decide(self, request):
        status, message = ('ok', None) if self.process is not None else self.start()
        if status == 'ok':
            deadline = time.monotonic() + self.timeout  # one deadline for sending and receiving
            status, message = self.send(json.dumps(request, allow_nan=False).encode(), deadline)
            if status == 'ok':
                status, message = self.receive(deadline)
        if status == 'ok':
            if isinstance(message, dict) and set(message) == {'error'} and isinstance(message['error'], str):
                status, message = 'worker_error', message['error'][:300]
            elif not (isinstance(message, dict) and set(message) == {'ok'}):
                status, message = 'worker_protocol_error', 'a message that is neither {"ok": reply} nor {"error": text}'
            else:
                message = message['ok']
        if status != 'ok':
            self.close(status)
        elif isinstance(message, dict) and message.get('restart') is True:
            self.close('restart')
        return status, message

    def close(self, reason='close'):
        """Kill the worker (if any), wait at most reap_timeout until it is gone and log whether it is."""
        process, conn, sender = self.process, self.conn, self.sender
        self.process = self.conn = self.sender = None
        if process is not None:
            if process.pid is not None:
                process.kill()
                process.join(self.reap_timeout)
                self.note({'action': 'close', 'pid': process.pid, 'reason': reason, 'exitcode': process.exitcode,
                           'alive': process.is_alive()})
        if sender is not None:
            sender.join(5)
        if conn is not None:
            conn.close()
