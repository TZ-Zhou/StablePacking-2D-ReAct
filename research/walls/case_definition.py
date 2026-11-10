"""What a wall is before it is built: its scene, its staged entry script and the code it is tied to.

research/runner/run_wall.py builds walls from these definitions; nothing here launches a process. Staging follows the
frozen baseline run_original_case.py (copied inputs, Windows main guard, shell cleanup refused, spawned workers inherit
constants/RNG via windows_case_runtime.py).
"""

import hashlib
import json
import os
import re
from pathlib import Path

from research.paths import REPO
from research.storage import case_id as cid

CASES = {  # case -> (entry script, original number of walls in its batch)
    "bricks": ("Tests/Bricks/run.py", 1),
    "almeida-r": ("Tests/Almeida/R/run.py", 1),
    "almeida-pr": ("Tests/Almeida/PR/run.py", 1),
    "almeida-ir": ("Tests/Almeida/IR/run.py", 1),
    "fft-d2-0": ("Tests/FFT_stones/01_run.py", 5),
    "rubble": ("Tests/Rubble_stone_walls/00_run.py", 20),
    "thangavelu-noise20": ("Tests/Thangavelu/00_run.py", 20),
    "thangavelu-noise80": ("Tests/Thangavelu/00_run.py", 20),
}
SOURCE_EDITS = {  # case -> (old, new) replacements that let one entry script serve several scenes
    "thangavelu-noise20": (("/data/stones_noise_80_labels.png", "/data/stones_noise_20_labels.png"),),
}
SUCCESS = ("PROCESS_PASS", "STOPPED_AFTER_PLACEMENTS")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_text(case):
    """The entry script of a case, with its SOURCE_EDITS applied."""
    text = (REPO / CASES[case][0]).read_text(encoding="utf-8")
    for old, new in SOURCE_EDITS.get(case, ()):
        assert text.count(old) == 1, (case, old)
        text = text.replace(old, new)
    return text


def parent_problem(ref):
    """Why `ref` names no wall (None when it does): a case id of research/cases or a research/runs path."""
    folder = cid.wall_dir(ref)
    return None if (folder / "receipt.json").exists() else f"no wall at {folder}"


def stage(source, policy, seed=None):
    """seed=None keeps the source's seed_everything() call (default 20); an int replaces only its
    argument, so every run initialises its own RNG before reading stones and sampling."""
    lines = source.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if "os.system(" in line and "rm -rf" in line and not line.lstrip().startswith("#"):
            # 'rm -rf <dir>/*' becomes: proceed only if <dir>/* matches nothing (Bricks cleans the
            # directory it has just created); never delete anything.
            indent, stripped = line[: len(line) - len(line.lstrip())], line.strip()
            assert stripped.startswith("os.system('rm -rf '+") and stripped.endswith(")"), line
            lines[i] = (
                indent + "if __import__('glob').glob((" + stripped[len("os.system(") : -1] + ")[7:]): "
                "raise FileExistsError('Refusing to clean a non-empty result directory')\n"
            )
    safe = "".join(lines)
    boundary = safe.index("seed_everything()\n")
    header = (
        "\nfrom windows_case_runtime import configure\n"
        "if __name__ == '__main__':\n"
        "    configure(globals())\n"
        "    from Stonepacker2D.utils.constant import set_filter_policy, set_trace\n"
        f"    set_filter_policy(**{policy['filter']!r})\n"
        f"    set_trace({policy['run_id']!r})\n"
    )
    if policy.get("controller") is not None:
        header += (
            f"    __import__('sys').path.insert(0, {str(REPO)!r})\n"
            "    from research.controller.order_controller import make_controller\n"
            "    from Stonepacker2D.utils.constant import set_filter_controller\n"
            f"    set_filter_controller(make_controller({policy['controller']!r}"
            f"{', worker_venv=True' if policy.get('worker_venv') else ''}))\n"
        )
    body = "".join("    " + l if l.strip() else l for l in safe[boundary:].splitlines(keepends=True))
    if seed is not None:
        call = "    seed_everything()\n"
        assert body.startswith(call), body[:40]
        body = f"    seed_everything({int(seed)})\n" + body[len(call) :]
    return safe[:boundary] + header + body


def parse_switch(texts):
    switch = {}
    for text in texts:
        step, order = text.split("=")
        order = tuple(int(c) for c in order.split(","))
        if sorted(order) != [0, 1, 2, 3, 4] or int(step) in switch:
            raise ValueError(f"bad --switch {text}")
        switch[int(step)] = order
    return switch


def worker_venv_identity():
    from research.controller import worker as controller_worker

    m = json.loads(controller_worker.VENV_MANIFEST.read_text(encoding="utf-8"))
    return {
        "manifest": controller_worker.VENV_MANIFEST.relative_to(REPO).as_posix(),
        "manifest_sha256": digest(controller_worker.VENV_MANIFEST),
        **{k: m[k] for k in ("path", "version", "sdks", "files_sha256")},
    }


def with_workers(source, workers):
    """The entry script with `workers` workers instead of its own number; the pool's size does not decide a wall."""
    if workers is None:
        return source
    text, replaced = re.subn(r"set_number_cpu\(\d+\)", f"set_number_cpu({workers})", source)
    assert replaced == 1, replaced
    return text


def with_vendor_type(source, vendor_type):
    """The entry script with build_wall_vendor's vendor_type replaced (e.g. 'full': the whole remaining pile is the
    candidate set at every step, Wang EJOR 2024 Fig. 16c); None keeps the scene's own sampling."""
    if vendor_type is None:
        return source
    pattern = r"vendor_type\s*=\s*(data_config\['vendor_type'\]|'\w+')"
    text, replaced = re.subn(pattern, f"vendor_type = {vendor_type!r}", source)
    assert replaced == 1, replaced
    return text


def code_fingerprint(root=REPO):
    """SHA256 of every executed source file, tracked or not, so a run is tied to its code."""
    files = (
        sorted(Path(root).glob("Stonepacker2D/**/*.py"))
        + [
            Path(root) / "research" / name
            for name in (
                "paths.py",
                "runner/run_wall.py",
                "walls/case_definition.py",
                "controller/order_controller.py",
                "controller/endgame.py",
                "controller/pick.py",
                "controller/worker.py",
                "controller/llm_api.py",
                "walls/check_run.py",
                "analysis/unplaced_stones.py",
                "runner/campaign.py",
                "walls/campaign.py",
                "walls/wall_metrics.py",
                "storage/case_id.py",
                "storage/case_index.py",
            )
        ]
        + [Path(root) / "research" / "validation" / name for name in ("windows_case_runtime.py", "step_writer.py")]
    )
    return {f.relative_to(root).as_posix(): digest(f) for f in files if f.exists()}


def quiet_env():
    """This process's environment without the API key variables: for every child but the case itself
    (the environment guard, git, powershell, taskkill), so the key reaches only the case, which hands
    it to the controller worker and drops it (order_controller.make_controller worker_venv)."""
    from research.controller import llm_api

    return {k: v for k, v in os.environ.items() if k.upper() not in llm_api.KEY_ENV.values()}
