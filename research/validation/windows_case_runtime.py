"""Process-local Windows compatibility and observation for original test scripts."""
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import time

import numpy as np


def _restore_worker(constants, numpy_state, random_state):
    module = importlib.import_module("Stonepacker2D.utils.constant")
    module.__dict__.update(constants)
    np.random.set_state(numpy_state)
    random.setstate(random_state)


def _pool(processes):
    module = importlib.import_module("Stonepacker2D.utils.constant")
    constants = {key: value for key, value in vars(module).items() if key.isupper()}
    return multiprocessing.Pool(processes, initializer=_restore_worker,
                                initargs=(constants, np.random.get_state(), random.getstate()))


def configure(namespace):
    """Preserve original decisions; inherit parent settings in spawned workers."""
    importlib.import_module("Stonepacker2D.stack.build_wall").Pool = _pool
    events = Path(os.environ["BASELINE_CASE_EVENTS"])
    start = time.monotonic()

    def emit(kind, **values):
        record = {"event": kind, "elapsed_seconds": round(time.monotonic() - start, 3), **values}
        with events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print("BASELINE_EVENT " + json.dumps(record), flush=True)

    original_build = namespace["build_wall_vendor"]
    original_evaluate = namespace["evaluate_kine"]

    def build(*args, **kwargs):
        constants = importlib.import_module("Stonepacker2D.utils.constant")
        emit("build_enter", input_stones=args[1].get_number_stones(),
             candidates_per_iteration=args[2], build_options=kwargs,
             constants={key: value for key, value in vars(constants).items() if key.isupper()})
        result = original_build(*args, **kwargs)
        emit("build_return", placed_rocks=len(result.placed_rocks), output_dir=constants.get_dir())
        return result

    def evaluate(*args, **kwargs):
        result = original_evaluate(*args, **kwargs)
        if kwargs.get("save_failure"):
            constants = importlib.import_module("Stonepacker2D.utils.constant")
            emit("final_evaluation_complete", placed_rocks=len(args[0].placed_rocks),
                 limit_force=float(result), output_dir=constants.get_dir())
            if os.environ["BASELINE_SELECT_FIRST_WALL"] == "1":
                # Select the first complete original replicate; do not change max_nb_runs.
                raise SystemExit(0)
        return result

    namespace["build_wall_vendor"] = build
    namespace["evaluate_kine"] = evaluate
