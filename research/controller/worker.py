"""The controller policy in its own process (multiprocessing spawn), spoken to in JSON bytes.

The main process (order_controller.ControllerClient) sends a request (observation, base order, tool
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

python (optional): the worker runs as `<python> -I -B worker.py --serve` (subprocess,
4-byte length-prefixed JSON frames over its stdin/stdout) instead of a multiprocessing child of
this interpreter. multiprocessing's spawn executable is process-wide (Wang's Pool reads it too),
so it is never changed. The worker's environment is built, not inherited: WORKER_ENV, the key
variable and paid gate of the spec's provider, and OPENAI_BASE_URL only on a loopback host; its
stderr is discarded. worker_python() gives the registered isolated venv (VENV_MANIFEST: the
Python that has the provider SDKs, its base the frozen stablepacking-baseline environment, so
the worker imports the same numpy as Wang) after checking every file against the manifest.
"""
import atexit
import hashlib
import json
import multiprocessing
import os
import queue
import struct
import subprocess
import sys
import threading
import time
import weakref
from pathlib import Path

MAX_BYTES = 1 << 20
# venv_main runs this file under `python -I`, whose sys.path holds neither the script's directory
# nor the working directory: the research package is found from here, not from an import
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'research' / 'validation'))
import local_setup  # noqa: E402  (this computer's paths; the registered ones without a local-setup file)
REGISTERED_VENV_MANIFEST = REPO / 'research' / 'validation' / 'worker-venv.json'
VENV_MANIFEST = local_setup.worker_venv_manifest(REGISTERED_VENV_MANIFEST)
# Windows needs these to start Python and load its DLLs, matplotlib (imported with Stonepacker2D) a
# home or config directory; nothing else of this process reaches the worker
WORKER_ENV = ('SYSTEMROOT', 'SYSTEMDRIVE', 'WINDIR', 'PATH', 'PATHEXT', 'COMSPEC', 'TEMP', 'TMP',
              'NUMBER_OF_PROCESSORS', 'PROCESSOR_ARCHITECTURE', 'USERPROFILE', 'HOMEDRIVE', 'HOMEPATH',
              'MPLBACKEND', 'MPLCONFIGDIR')


def venv_files(root):
    """{relative path: sha256} of every file of a venv (bytecode caches left out)."""
    root = Path(root)
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}


def venv_digest(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def venv_problems(manifest=VENV_MANIFEST):
    """Why the registered worker venv may not run: absent, a file added, removed or changed, a
    pyvenv.cfg not on the frozen baseline environment, or an interpreter that does not report the
    manifest's prefix, base, version and SDK."""
    m = json.loads(Path(manifest).read_text(encoding='utf-8'))
    root = Path(m['path'])
    if not (root / 'Scripts' / 'python.exe').is_file():
        return [f'worker venv: no interpreter at {root}']
    files = venv_files(root)
    changed = sorted(set(files) ^ set(m['files']) | {f for f in set(files) & set(m['files']) if files[f] != m['files'][f]})
    if changed:
        return [f'worker venv: {len(changed)} files differ from the manifest, e.g. {changed[:3]}']
    probe = ('import json, sys, importlib.metadata as md; print(json.dumps({"prefix": sys.prefix, "base_prefix": sys.base_prefix, '
             '"version": sys.version.split()[0], "sdks": {d: md.version(d) for d in ("openai",)}}))')
    try:
        out = subprocess.run([str(root / 'Scripts' / 'python.exe'), '-I', '-B', '-c', probe], capture_output=True, text=True,
                             timeout=60, env={k: os.environ[k] for k in WORKER_ENV if k in os.environ})
        seen = json.loads(out.stdout)
    except (subprocess.TimeoutExpired, ValueError, OSError) as error:
        return [f'worker venv: the interpreter does not report itself ({type(error).__name__})']
    frozen = str(
        local_setup.env_prefix(
            json.loads((REPO / "research" / "validation" / "environment-manifest.json").read_text(encoding="utf-8"))[
                "prefix"
            ]
        )
    )
    if os.path.normcase(m['base_prefix']) != os.path.normcase(frozen):
        return [f"worker venv: its base {m['base_prefix']} is not the frozen environment {frozen}"]
    want = {k: m[k] for k in ('prefix', 'base_prefix', 'version', 'sdks')}
    return [f'worker venv: {k} is {seen.get(k)!r}, the manifest says {v!r}' for k, v in want.items()
            if (os.path.normcase(seen.get(k) or '') != os.path.normcase(v) if k.endswith('prefix') else seen.get(k) != v)]


def register_venv(root, manifest=VENV_MANIFEST):
    """Write the manifest of the venv at root once (never overwritten): its files, prefix, base,
    version and SDK versions, as its own interpreter reports them."""
    root, manifest = Path(root).resolve(), Path(manifest)
    probe = ('import json, sys, importlib.metadata as md; print(json.dumps({"prefix": sys.prefix, "base_prefix": sys.base_prefix, '
             '"version": sys.version.split()[0], "sdks": {d: md.version(d) for d in ("openai",)}}))')
    seen = json.loads(subprocess.run([str(root / 'Scripts' / 'python.exe'), '-I', '-B', '-c', probe], capture_output=True, text=True,
                                     timeout=60, check=True, env={k: os.environ[k] for k in WORKER_ENV if k in os.environ}).stdout)
    files = venv_files(root)
    data = {'path': str(root), **seen, 'files_sha256': venv_digest(files), 'files': files}
    with os.fdopen(os.open(manifest, os.O_WRONLY | os.O_CREAT | os.O_EXCL), 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1, sort_keys=True)
    return data


def worker_python(manifest=VENV_MANIFEST):
    """The registered worker venv's interpreter, after venv_problems; raises if any."""
    problems = venv_problems(manifest)
    if problems:
        raise RuntimeError('; '.join(problems))
    return str(Path(json.loads(Path(manifest).read_text(encoding='utf-8'))['path']) / 'Scripts' / 'python.exe')


def worker_env(spec, environ=None):
    """The environment of a venv worker for this (full) spec."""
    environ = os.environ if environ is None else environ
    env = {k: environ[k] for k in WORKER_ENV if k in environ}
    llm = spec.get('llm') or {}
    if spec['kind'] == 'react' and llm.get('kind') == 'api' and llm.get('provider') in ('openai', 'anthropic'):
        from urllib.parse import urlparse
        from research.controller import llm_api
        for k in (llm_api.KEY_ENV[llm['provider']], llm_api.PAID_GATE):
            if k in environ:
                env[k] = environ[k]
        if llm['provider'] == 'openai' and urlparse(environ.get('OPENAI_BASE_URL', '')).hostname in llm_api.LOOPBACK:
            env['OPENAI_BASE_URL'] = environ['OPENAI_BASE_URL']
    return env


EOF_, OVERSIZE = object(), object()


class FramedPipe:
    """The parent's end of a venv worker: multiprocessing.Connection's poll / recv_bytes /
    send_bytes / close over the worker's stdin and stdout; a reader thread fills a queue."""

    def __init__(self, write, read):
        self.write, self.read, self.queue, self.pending = write, read, queue.Queue(), None
        threading.Thread(target=self.reader, name='controller-frames', daemon=True).start()

    def reader(self):
        try:
            while True:
                head = self.read.read(4)
                if len(head) < 4:
                    break
                n = struct.unpack('>I', head)[0]
                if n > MAX_BYTES:
                    self.queue.put(OVERSIZE)
                    return
                data = self.read.read(n)
                if len(data) < n:
                    break
                self.queue.put(data)
        except (OSError, ValueError):
            pass
        self.queue.put(EOF_)

    def poll(self, timeout):
        if self.pending is None:
            try:
                self.pending = self.queue.get(timeout=timeout)
            except queue.Empty:
                return False
        return True

    def recv_bytes(self, maxlength=None):
        item, self.pending = (self.pending if self.pending is not None else self.queue.get()), None
        if item is EOF_:
            self.pending = EOF_
            raise EOFError
        if item is OVERSIZE:
            raise OSError('bad message length')
        return item

    def send_bytes(self, data):
        self.write.write(struct.pack('>I', len(data)) + data)
        self.write.flush()

    def close(self):
        for f in (self.write, self.read):
            try:
                f.close()
            except OSError:
                pass


class VenvProcess:
    """A subprocess.Popen with the multiprocessing.Process methods the transport uses."""

    def __init__(self, popen):
        self.popen, self.pid = popen, popen.pid

    exitcode = property(lambda self: self.popen.poll())

    def is_alive(self):
        return self.popen.poll() is None

    def kill(self):
        self.popen.kill()

    def join(self, timeout=None):
        try:
            self.popen.wait(timeout)
        except subprocess.TimeoutExpired:
            pass


class StdioConn:
    """The worker's end: frames on the binary stdin / stdout it was started with."""

    def __init__(self, read, write):
        self.read, self.write = read, write

    def recv_bytes(self, maxlength=None):
        head = self.read.read(4)
        if len(head) < 4:
            raise EOFError
        data = self.read.read(struct.unpack('>I', head)[0])
        return data

    def send_bytes(self, data):
        self.write.write(struct.pack('>I', len(data)) + data)
        self.write.flush()


def venv_main(argv):
    """`worker.py --serve <module> <function> <directory>`: the target from its module
    (found in directory), the spec as the first frame, then target(conn, spec)."""
    _, module, name, directory = argv
    out = os.fdopen(os.dup(sys.stdout.fileno()), 'wb')
    sys.stdout = sys.stderr  # nothing printed can corrupt a frame
    for p in (directory, str(REPO)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import importlib
    target = getattr(importlib.import_module(module), name)
    conn = StdioConn(sys.stdin.buffer, out)
    target(conn, conn.recv_bytes().decode())


def summary(data):
    """A safe description of a bad message: its length and the first 200 characters."""
    return f'{len(data)} bytes: {data[:200].decode("utf-8", "replace")!r}'


def serve(conn, spec):
    """Worker main: build the policy of `spec` (JSON text), say ready, answer requests until EOF.
    Replies are {'ok': reply} or {'error': 'Type: message'}; every message is masked
    (order_controller.masked) before it leaves the worker, so no API key it read can reach a record."""
    from research.controller import order_controller
    try:
        policy = order_controller.make_policy(json.loads(spec))
    except Exception as error:  # an SDK that fails to build its client included
        conn.send_bytes(order_controller.masked(json.dumps({'error': order_controller.error_text(error)})).encode())
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
            out = json.dumps({'error': order_controller.error_text(error)})
        conn.send_bytes(order_controller.masked(out).encode())


def close_at_exit(ref):
    transport = ref()
    if transport is not None:
        transport.close('main_exit')


def error_text(error):
    from research.controller.order_controller import error_text
    return error_text(error)


class ProcessTransport:
    """decide(request, settle=None) -> ('ok', reply) or (failure, safe description); see the module
    docstring. target: the worker main (tests pass misbehaving ones); it must be importable by name."""

    def __init__(self, spec, timeout=None, start_timeout=120.0, target=serve, reap_timeout=30.0, python=None):
        react = spec['kind'] == 'react'
        self.spec, self.target, self.start_timeout, self.reap_timeout = json.dumps(spec), target, start_timeout, reap_timeout
        self.python, self.env = python, None if python is None else worker_env(spec)
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

        def launch_venv():  # the same slot discipline as launch(), a subprocess of self.python
            module = sys.modules[self.target.__module__]
            name = 'research.controller.worker' if self.target.__module__ == '__main__' else self.target.__module__
            cmd = [self.python, '-I', '-B', str(Path(__file__).resolve()), '--serve', name, self.target.__name__,
                   str(Path(module.__file__).resolve().parent)]
            try:
                popen = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                         env=self.env, cwd=str(REPO))
            except Exception as error:
                return 'worker_error', f'cannot start the worker: {error_text(error)}'
            slot['process'], slot['conn'] = VenvProcess(popen), FramedPipe(popen.stdin, popen.stdout)
            self.started += 1
            if self.slot is slot:
                self.pid = popen.pid
            self.note({'action': 'start', 'pid': popen.pid, 'n': self.started})
            try:
                slot['conn'].send_bytes(self.spec.encode())
            except OSError as error:
                return 'worker_error', f'cannot send the spec: {type(error).__name__}'
            return self.read(deadline, slot)

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
        status, message = self.supervised(launch if self.python is None else launch_venv, deadline,
                                          f'the worker did not start within {self.start_timeout} s')
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


if __name__ == "__main__" and sys.argv[1:2] == ["--serve"]:
    venv_main(sys.argv[1:])
