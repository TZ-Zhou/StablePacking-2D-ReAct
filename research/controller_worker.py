"""The controller policy in its own process (multiprocessing spawn), spoken to in JSON bytes.

The main process (controllers.ControllerClient) sends a request (observation, base order, tool
table) with send_bytes and waits at most `timeout` seconds for the reply; only JSON crosses the
pipe, so the worker holds no Wang object, score matrix, global setting or RNG of the main process,
and the main process receives nothing but parsed JSON. Any failure kills the worker (kill + join)
and the next request starts a new one:
- worker_timeout: no reply in time (the worker is killed with whatever it was running);
- worker_error: the worker exited, could not start, or reported an exception of its policy;
- worker_protocol_error: a message over MAX_BYTES, not JSON, nested too deep, or not of the protocol.
A policy reply asking for a restart (an abandoned LLM call) also ends the worker after the reply.
"""
import json
import multiprocessing
import sys
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


class ProcessTransport:
    """decide(request) -> ('ok', reply) or (failure, safe description); see the module docstring.
    target: the worker main (tests pass misbehaving ones); it must be importable by name."""

    def __init__(self, spec, timeout=None, start_timeout=120.0, target=serve):
        react = spec['kind'] == 'react'
        self.spec, self.target, self.start_timeout = json.dumps(spec), target, start_timeout
        self.timeout = timeout or (spec['llm_timeout'] * spec['max_tool_calls'] + 30 if react else 30.0)
        self.process = self.conn = None
        self.started = 0

    def receive(self, timeout):
        if not self.conn.poll(timeout):
            return 'worker_timeout', f'no reply within {timeout} s'
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

    def start(self):
        context = multiprocessing.get_context('spawn')
        self.conn, child = context.Pipe()
        self.process = context.Process(target=self.target, args=(child, self.spec), name='controller-worker', daemon=True)
        self.process.start()
        child.close()
        self.started += 1
        status, message = self.receive(self.start_timeout)
        if status == 'ok' and message != {'ready': True}:
            status, message = 'worker_error', (message['error'] if isinstance(message, dict) and isinstance(message.get('error'), str)
                                               else 'the worker did not start')
        return status, message

    def decide(self, request):
        status, message = ('ok', None) if self.process is not None else self.start()
        if status == 'ok':
            try:
                self.conn.send_bytes(json.dumps(request, allow_nan=False).encode())
                status, message = self.receive(self.timeout)
            except OSError as error:  # the worker is gone (broken pipe)
                status, message = 'worker_error', f'cannot send the request: {type(error).__name__}'
        if status == 'ok':
            if isinstance(message, dict) and set(message) == {'error'} and isinstance(message['error'], str):
                status, message = 'worker_error', message['error'][:300]
            elif not (isinstance(message, dict) and set(message) == {'ok'}):
                status, message = 'worker_protocol_error', 'a message that is neither {"ok": reply} nor {"error": text}'
            else:
                message = message['ok']
        if status != 'ok' or (isinstance(message, dict) and message.get('restart') is True):
            self.close()
        return status, message

    def close(self):
        """Kill the worker (if any) and wait until it is gone."""
        if self.process is not None:
            self.process.kill()
            self.process.join(30)
            self.conn.close()
            self.process = self.conn = None
