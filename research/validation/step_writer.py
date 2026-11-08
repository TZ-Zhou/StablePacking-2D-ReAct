"""One thread writes every step output of a case in call order, so the building loop never waits for the disk.

Staged next to windows_case_runtime.py in the case folder, which imports it and calls install(build_wall module) in
the parent process unless BASELINE_SYNC_OUTPUT is "1" (run_wall --sync-output, or a wall with --controller). It cannot
import the research package; it needs only the standard library and numpy.

install replaces build_wall's write_trace and gives the module its own global `open`. Everything goes through one queue,
because run_wall.check_commits requires a killed wall to be a prefix of the order build_wall wrote in. Nothing else is
written from here: the outputs are the ones build_wall has always made, in the same modes.
"""

import atexit
import builtins
import io
import json
import os
import queue
import threading

import numpy as np

QUEUE_DEPTH = 32  # a full queue makes the loop wait, which bounds the memory held by pending outputs


class _Pending:
    """An in-memory file that hands its whole content over when closed."""

    def __init__(self, deliver):
        super().__init__()
        self._deliver = deliver

    def close(self):
        if not self.closed:
            content = self.getvalue()
            super().close()
            self._deliver(content)


class _PendingText(_Pending, io.StringIO):
    """Keeps newlines as written: the real text-mode file translates them at write time, as it always did."""


class _PendingBytes(_Pending, io.BytesIO):
    pass


BUFFERED_MODES = {"a": _PendingText, "a+": _PendingText, "wb": _PendingBytes}


def _write_file(path, mode, args, kwargs, content):
    """Reopen with the mode build_wall asked for, so Windows newlines and the encoding do not change."""
    with builtins.open(path, mode, *args, **kwargs) as target:
        target.write(content)


class StepWriter:
    """Runs submitted jobs one at a time in submission order; after a failed job the remaining ones are skipped, so
    the disk holds an exact prefix. The failure is raised by the next submit or flush."""

    def __init__(self, depth=QUEUE_DEPTH):
        self.jobs = queue.Queue(depth)
        self.error = None
        threading.Thread(target=self._run, name="step-writer", daemon=True).start()

    def submit(self, job):
        self._raise_if_failed()
        self.jobs.put(job)

    def flush(self):
        """Wait until every submitted job has run; raise the write error if one failed."""
        self.jobs.join()
        self._raise_if_failed()

    def open(self, file, mode="r", *args, **kwargs):
        """build_wall's `open`: modes a, a+ and wb are written by the thread once closed; any other mode (time.txt's
        w+) waits for the queue first, because it acts on the disk at this point of the call order."""
        if mode not in BUFFERED_MODES or isinstance(file, int):
            self.flush()
            return builtins.open(file, mode, *args, **kwargs)
        path = os.path.abspath(os.fspath(file))
        return BUFFERED_MODES[mode](lambda content: self.submit(lambda: _write_file(path, mode, args, kwargs, content)))

    def trace_writer(self, write_trace):
        """write_trace with its arguments copied now, since the loop goes on updating the wall; the file is written by
        the thread. The JSON round trip keeps the text write_trace stores for meta exactly."""

        def queued(directory, meta, **arrays):
            target = os.path.abspath(os.fspath(directory))
            frozen_meta = json.loads(json.dumps(meta))
            frozen = {name: np.array(value, copy=True, subok=True) for name, value in arrays.items()}
            self.submit(lambda: write_trace(target, frozen_meta, **frozen))

        return queued

    def _raise_if_failed(self):
        if self.error is not None:
            raise self.error

    def _run(self):
        while True:
            job = self.jobs.get()
            try:
                if self.error is None:
                    job()
            except BaseException as error:  # kept for the main thread; this thread must go on draining the queue
                self.error = error
            finally:
                self.jobs.task_done()


def install(build_wall_module):
    """Route build_wall's step outputs through a new StepWriter, flushed again at exit; returns the writer."""
    writer = StepWriter()
    build_wall_module.write_trace = writer.trace_writer(build_wall_module.write_trace)
    build_wall_module.open = writer.open
    atexit.register(writer.flush)
    return writer
