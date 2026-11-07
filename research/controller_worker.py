"""The controller policy in its own process (multiprocessing spawn), spoken to in JSON bytes.

The main process (controllers.ControllerClient) sends a request (observation, base order, tool
table) and waits for the reply; only JSON crosses the pipe, so the worker holds no Wang object,
score matrix, global setting or RNG of the main process, and the main process receives nothing but
parsed JSON. Every stage runs in a supervised I/O thread the main process waits for with a
deadline, so no blocking call of the pipe or of the spawn can hold Wang's process: the start
(Pipe, spawn and the ready message) is bounded by start_timeout, the request (sending it and
receiving the whole reply) by one deadline of `timeout` seconds, and the whole cleanup by one
deadline of reap_timeout seconds. Each start fills its own slot (pipe end, process), so a start
that returns only after its deadline is still found and killed by the cleanup that holds it.
Any failure kills the worker (kill + join) and the next request starts a new one:
- worker_timeout: the worker was not started, did not take the request or did not answer in time
  (it is killed with whatever it was running);
- worker_error: the worker exited, could not start, or reported an exception of its policy;
- worker_protocol_error: a message over MAX_BYTES, not JSON, nested too deep, or not of the protocol;
- worker_unreaped: an earlier worker could not be confirmed ended (kill failed, still alive after
  reap_timeout, or its start never returned); no request is sent and no worker started until a
  later reap confirms it, so no decision runs beside a worker that may still be running.
A policy reply asking for a restart (an abandoned LLM call) also ends the worker after the reply.
decide(request, settle) calls settle(status, message) before any cleanup (the client validates
the reply and writes its ledger line there; settle may turn an 'ok' into a failure), then closes
the worker of a failure. `log` (if set) receives the lifecycle of each worker: start (pid) and
close (pid, reason, exit code, whether it was still alive after the reap timeout, a kill error),
as evidence that no worker outlived its decision; the last worker is closed when the main process
exits normally.
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
    Replies are {'ok': reply} or {'error': 'Type: message'}; every message is masked
    (controllers.masked) before it leaves the worker, so no API key it read can reach a record."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import controllers
    try:
        policy = controllers.make_policy(json.loads(spec))
    except Exception as error:  # an SDK that fails to build its client included
        conn.send_bytes(controllers.masked(json.dumps({'error': controllers.error_text(error)})).encode())
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
        conn.send_bytes(controllers.masked(out).encode())


def close_at_exit(ref):
    transport = ref()
    if transport is not None:
        transport.close('main_exit')


def error_text(error):
    from controllers import error_text
    return error_text(error)


class ProcessTransport:
    """decide(request, settle=None) -> ('ok', reply) or (failure, safe description); see the module
    docstring. target: the worker main (tests pass misbehaving ones); it must be importable by name."""

    def __init__(self, spec, timeout=None, start_timeout=120.0, target=serve, reap_timeout=30.0):
        react = spec['kind'] == 'react'
        self.spec, self.target, self.start_timeout, self.reap_timeout = json.dumps(spec), target, start_timeout, reap_timeout
        self.timeout = timeout or (spec['llm_timeout'] * spec['max_tool_calls'] + 30 if react else 30.0)
        self.slot = self.io = None  # the current worker's {'process', 'conn'}, filled by its start
        self.unreaped = None  # (slot, I/O thread) of a worker not confirmed ended
        self.started, self.log, self.exit_hook, self.pid = 0, None, False, None

    process = property(lambda self: (self.slot or {}).get('process'))
    conn = property(lambda self: (self.slot or {}).get('conn'))

    def note(self, entry):
        if self.log is not None:
            self.log({'type': 'worker', **entry})

    def supervised(self, run, deadline, late):
        """run() in a daemon I/O thread, waited for until the deadline: its (status, message), or
        ('worker_timeout', late) if it is still blocked (close() then kills the worker, which ends
        the blocked pipe call, and joins the thread)."""
        box = {}

        def body():
            try:
                box['out'] = run()
            except BaseException as error:  # reported to the main thread, never raised in it
                box['error'] = error
        self.io = threading.Thread(target=body, name='controller-io', daemon=True)
        self.io.start()
        self.io.join(max(0.0, deadline - time.monotonic()))
        if self.io.is_alive():
            return 'worker_timeout', late
        if 'error' in box:
            return 'worker_error', error_text(box['error'])
        return box['out']

    def read(self, deadline, slot):
        """The next message of slot's worker, read in the I/O thread (poll, then the whole message)."""
        timeout = max(0.0, deadline - time.monotonic())
        if not slot['conn'].poll(timeout):
            return 'worker_timeout', f'no reply within {round(timeout, 3)} s'
        try:
            data = slot['conn'].recv_bytes(MAX_BYTES)
        except EOFError:
            slot['process'].join(5)
            return 'worker_error', f"the worker exited with code {slot['process'].exitcode}"
        except OSError:
            return 'worker_protocol_error', f'a message over {MAX_BYTES} bytes'
        try:
            message = json.loads(data)
        except (ValueError, RecursionError):
            return 'worker_protocol_error', f'not JSON: {summary(data)}'
        return 'ok', message

    def start(self):
        deadline = time.monotonic() + self.start_timeout
        self.pid, slot = None, {'process': None, 'conn': None}
        self.slot = slot

        def launch():  # fills its own slot only: a late return cannot touch a later worker's
            context = multiprocessing.get_context('spawn')
            slot['conn'], child = context.Pipe()
            try:
                slot['process'] = context.Process(target=self.target, args=(child, self.spec), name='controller-worker', daemon=True)
                try:
                    slot['process'].start()
                except Exception as error:
                    return 'worker_error', f'cannot start the worker: {error_text(error)}'
            finally:
                child.close()
            self.started += 1
            if self.slot is slot:
                self.pid = slot['process'].pid
            self.note({'action': 'start', 'pid': slot['process'].pid, 'n': self.started})
            return self.read(deadline, slot)
        status, message = self.supervised(launch, deadline, f'the worker did not start within {self.start_timeout} s')
        if not self.exit_hook:  # the last worker ends with the main process, and says so (a weak
            atexit.register(close_at_exit, weakref.ref(self))  # reference: a dropped transport's worker gets EOF)
            self.exit_hook = True
        if status == 'ok' and message != {'ready': True}:
            status, message = 'worker_error', (message['error'] if isinstance(message, dict) and isinstance(message.get('error'), str)
                                               else 'the worker did not start')
        return status, message

    def exchange(self, request):
        data, deadline, slot = json.dumps(request, allow_nan=False).encode(), time.monotonic() + self.timeout, self.slot

        def run():
            try:
                slot['conn'].send_bytes(data)
            except Exception as error:  # the worker is gone (broken pipe) or the pipe was closed
                return 'worker_error', f'cannot send the request: {type(error).__name__}'
            return self.read(deadline, slot)
        status, message = self.supervised(run, deadline, f'no reply within {self.timeout} s')
        if status != 'ok':
            return status, message
        if isinstance(message, dict) and set(message) == {'error'} and isinstance(message['error'], str):
            return 'worker_error', message['error'][:300]
        if not (isinstance(message, dict) and set(message) == {'ok'}):
            return 'worker_protocol_error', 'a message that is neither {"ok": reply} nor {"error": text}'
        return 'ok', message['ok']

    def decide(self, request, settle=None):
        if self.unreaped is not None:
            self.close('reap_retry')
        if self.unreaped is not None:
            stuck = self.unreaped[0]['process']
            status, message = 'worker_unreaped', f"worker {getattr(stuck, 'pid', None)} is not confirmed ended; no request is sent"
        else:
            status, message = ('ok', None) if self.process is not None else self.start()
            if status == 'ok':
                status, message = self.exchange(request)
        if settle is not None:
            status, message = settle(status, message)
        if status not in ('ok', 'worker_unreaped'):
            self.close(status)
        elif status == 'ok' and isinstance(message, dict) and message.get('restart') is True:
            self.close('restart')
        return status, message

    def close(self, reason='close'):
        """Kill the worker (if any) and wait, all within one deadline of reap_timeout seconds, until it
        and the I/O thread are gone; log whether they are. The process is read from the slot only
        after the I/O thread is gone (a start returning late has filled it by then). A worker not
        confirmed ended stays held, with its slot, in `unreaped`."""
        slot, io = self.unreaped or (self.slot, self.io)
        slot = slot or {'process': None, 'conn': None}
        self.slot = self.io = self.unreaped = None
        deadline, kill_error = time.monotonic() + self.reap_timeout, None
        left = lambda: max(0.0, deadline - time.monotonic())
        process = lambda: slot['process']
        spawned = lambda: process() is not None and process().pid is not None

        def kill():  # ponytail: kill() itself is not timed (TerminateProcess / SIGKILL return at once)
            nonlocal kill_error
            try:
                process().kill()
            except Exception as error:
                kill_error = error_text(error)
            try:
                process().join(left())
            except Exception as error:
                kill_error = kill_error or error_text(error)
        try:
            if spawned():
                kill()  # ends any pipe call the I/O thread is blocked in
        finally:
            if io is not None:
                io.join(left())
            if spawned() and process().exitcode is None and kill_error is None:
                kill()  # a start that returned only now, or a first kill that did not take
            if slot['conn'] is not None:
                try:
                    slot['conn'].close()
                except Exception:
                    pass
        try:
            alive = (spawned() and process().is_alive()) or (io is not None and io.is_alive())
        except Exception:  # cannot tell: not confirmed
            alive = True
        if spawned() or alive:
            self.note({'action': 'close', 'pid': process().pid if spawned() else None, 'reason': reason,
                       'exitcode': process().exitcode if spawned() else None, 'alive': alive,
                       **({'kill_error': kill_error} if kill_error else {})})
        self.pid = None  # later requests name no closed worker
        if alive:
            self.unreaped = (slot, io)
