"""Machine-local paths: the registered manifests name the machine that froze them; this module lets
each computer say where its own copy lives, without touching any tracked file (so the tracked bytes,
and every hash recorded from them, are the same on every computer).

research/validation/local-setup.json (not in git; template local-setup.example.json):
  env_prefix              sys.prefix of this computer's frozen baseline environment
  worker_venv_manifest    file name, in this folder, of this computer's worker-venv registration

Without that file the registered paths apply unchanged.
Create it for this computer:  python research/validation/local_setup.py init
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCAL = HERE / "local-setup.json"
LOCAL_VENV_MANIFEST = "worker-venv.local.json"


def load():
    """The local overrides ({} when this computer has none)."""
    return json.loads(LOCAL.read_text(encoding="utf-8")) if LOCAL.is_file() else {}


def env_prefix(registered):
    """This computer's baseline environment prefix, or the registered one."""
    return Path(load().get("env_prefix") or registered)


def worker_venv_manifest(registered):
    """This computer's worker-venv registration file, or the registered one."""
    name = load().get("worker_venv_manifest")
    return HERE / name if name else Path(registered)


def init():
    """Write local-setup.json for the Python running this: its prefix, and the local venv manifest."""
    data = {"env_prefix": str(Path(sys.prefix).resolve()), "worker_venv_manifest": LOCAL_VENV_MANIFEST}
    with open(LOCAL, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    return data


if __name__ == "__main__":
    if sys.argv[1:] == ["init"]:
        print(json.dumps(init(), indent=1))
    else:
        print(__doc__)
