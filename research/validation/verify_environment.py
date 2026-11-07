"""Fail before an experiment if the registered baseline environment has drifted."""
import importlib.metadata as md
import json
from pathlib import Path
import re
import sys

repo = Path(__file__).resolve().parents[2]
registered = json.loads(Path(__file__).with_name("environment-manifest.json").read_text(encoding="utf-8"))
assert Path(sys.prefix).resolve() == Path(registered["prefix"]).resolve(), sys.prefix
assert sys.version == registered["python"], sys.version
assert sys.platform == registered["platform"], sys.platform
actual = {re.sub(r"[-_.]+", "-", d.metadata["Name"]).lower(): d.version for d in md.distributions()}
assert actual == registered["packages"], {
    name: {"expected": registered["packages"].get(name), "actual": actual.get(name)}
    for name in actual.keys() | registered["packages"].keys()
    if actual.get(name) != registered["packages"].get(name)}
native = [json.loads(p.read_text(encoding="utf-8")) for p in (Path(sys.prefix) / "conda-meta").glob("*.json")]
assert {r["name"]: (r["version"], r["build"]) for r in native} == {
    r["name"]: (r["version"], r["build"]) for r in registered["conda_records"]}
import Stonepacker2D
assert Path(Stonepacker2D.__file__).resolve().parent == repo / "Stonepacker2D"
print("PASS: exact registered Python/package versions and Conda builds, named environment, repository module origin")
