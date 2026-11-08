"""The step writer of a case: call order on disk, snapshots, failures, the bounded queue (no wall is built)."""

import pickle
import tempfile
import threading
import types
import unittest
from pathlib import Path

import numpy as np

from research.validation import step_writer


def trace_stub(directory, meta, **arrays):
    Path(directory).mkdir(exist_ok=True)
    np.savez(Path(directory) / f"e{meta['event']:05d}.npz", **arrays)


class StepWriterTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.tmp = Path(folder.name)
        self.module = types.SimpleNamespace(write_trace=trace_stub)
        self.writer = step_writer.install(self.module)

    def outputs(self):
        return sorted(p.relative_to(self.tmp).as_posix() for p in self.tmp.rglob("*") if p.is_file())

    def write_text(self, name, text):
        with self.module.open(str(self.tmp / name), "a+") as f:
            f.write(text)

    def test_mixed_outputs_land_in_call_order(self):
        seen = []

        def record():  # runs in the writer thread, right after the output before it
            seen.append(self.outputs())

        self.module.write_trace(str(self.tmp / "trace"), {"event": 0}, x=np.arange(3))
        self.writer.submit(record)
        self.write_text("decisions.txt", "0;1\n")
        self.writer.submit(record)
        with self.module.open(str(self.tmp / "base.pkl"), "wb") as f:
            pickle.dump({"a": 1}, f)
        self.writer.submit(record)
        self.writer.flush()
        self.assertEqual(seen[0], ["trace/e00000.npz"])
        self.assertEqual(seen[1], ["decisions.txt", "trace/e00000.npz"])
        self.assertEqual(seen[2], ["base.pkl", "decisions.txt", "trace/e00000.npz"])

    def test_an_array_changed_after_submit_is_written_as_it_was(self):
        wall = np.zeros((4, 4))
        release = threading.Event()
        self.writer.submit(release.wait)
        self.module.write_trace(str(self.tmp / "trace"), {"event": 0}, wall=wall)
        wall[:] = 7
        release.set()
        self.writer.flush()
        with np.load(self.tmp / "trace" / "e00000.npz") as data:
            self.assertEqual(data["wall"].sum(), 0)

    def test_a_write_failure_is_raised_and_nothing_after_it_is_written(self):
        def fail():
            raise OSError("disk full")

        self.write_text("first.txt", "kept\n")
        self.writer.submit(fail)
        self.write_text_ignoring_failure("second.txt")
        with self.assertRaises(OSError):
            self.writer.flush()
        with self.assertRaises(OSError):
            self.writer.submit(lambda: None)
        self.assertEqual(self.outputs(), ["first.txt"])
        self.writer.error = None  # the exit flush of this test's writer

    def write_text_ignoring_failure(self, name):
        try:
            self.write_text(name, "dropped\n")
        except OSError:
            pass

    def test_a_full_queue_makes_the_loop_wait_and_loses_nothing(self):
        release = threading.Event()
        self.writer.submit(release.wait)
        count = step_writer.QUEUE_DEPTH + 5
        done = threading.Event()

        def produce():
            for i in range(count):
                self.write_text("lines.txt", f"{i}\n")
            done.set()

        threading.Thread(target=produce, daemon=True).start()
        self.assertFalse(done.wait(0.3))
        release.set()
        self.assertTrue(done.wait(10))
        self.writer.flush()
        lines = (self.tmp / "lines.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines, [str(i) for i in range(count)])

    def test_text_mode_output_equals_a_direct_write_byte_for_byte(self):
        text = "1;2;3\nsecond line\n"
        with open(self.tmp / "direct.txt", "a") as f:
            f.write(text)
        for mode in ("a", "a+"):
            with self.module.open(str(self.tmp / f"queued{mode[-1]}.txt"), mode) as f:
                f.write(text)
        self.writer.flush()
        direct = (self.tmp / "direct.txt").read_bytes()
        self.assertEqual((self.tmp / "queueda.txt").read_bytes(), direct)
        self.assertEqual((self.tmp / "queued+.txt").read_bytes(), direct)

    def test_another_mode_waits_for_the_queue(self):
        path = str(self.tmp / "time.txt")
        self.write_text("time.txt", "old\n")
        with self.module.open(path, "w+") as f:
            f.write("new\n")
        self.writer.flush()
        self.assertEqual((self.tmp / "time.txt").read_text(encoding="utf-8").splitlines(), ["new"])


if __name__ == "__main__":
    unittest.main()
