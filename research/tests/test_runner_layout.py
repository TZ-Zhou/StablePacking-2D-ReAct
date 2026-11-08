"""Every wall launcher is in research/runner, and the definitions it runs never depend on it."""

import ast
import unittest

from research.paths import REPO

RESEARCH = REPO / "research"
LIBRARIES = ("analysis", "controller", "llm_walls", "perf", "storage", "validation", "walls")
DEFINITIONS = (
    "analysis/counterfactual_branches.py",
    "analysis/lookahead_study.py",
    "llm_walls/dev_walls.py",
    "llm_walls/gpt_counterfactual.py",
    "llm_walls/gpt_wall.py",
    "perf/verify_speedup.py",
    "walls/batch_plan.py",
    "walls/campaign.py",
    "walls/case_definition.py",
)
LAUNCHERS = (
    "batch",
    "campaign",
    "counterfactual_branches",
    "dev_walls",
    "gpt_counterfactual",
    "gpt_wall",
    "lookahead_study",
    "profile_case",
    "run_wall",
    "verify_speedup",
)


def imported_modules(path):
    """Every module named by an import of `path`, for `from a import b` both a and a.b."""
    nodes = list(ast.walk(ast.parse(path.read_text(encoding="utf-8"))))
    plain = {alias.name for node in nodes if isinstance(node, ast.Import) for alias in node.names}
    froms = [node for node in nodes if isinstance(node, ast.ImportFrom) and node.module]
    return plain | {node.module for node in froms} | {f"{n.module}.{alias.name}" for n in froms for alias in n.names}


def has_main_guard(path):
    return any(
        isinstance(node, ast.If) and "__main__" in ast.dump(node.test)
        for node in ast.parse(path.read_text(encoding="utf-8")).body
    )


class RunnerLayout(unittest.TestCase):
    def test_no_library_module_imports_the_runners(self):
        modules = [path for folder in LIBRARIES for path in (RESEARCH / folder).glob("*.py")]
        offenders = [
            path.name for path in modules if any(name.startswith("research.runner") for name in imported_modules(path))
        ]
        self.assertEqual(offenders, [])

    def test_every_launcher_is_a_runner_with_a_command_line(self):
        for name in LAUNCHERS:
            with self.subTest(name):
                self.assertTrue(has_main_guard(RESEARCH / "runner" / f"{name}.py"))

    def test_a_definition_has_no_command_line(self):
        for name in DEFINITIONS:
            with self.subTest(name):
                self.assertFalse(has_main_guard(RESEARCH / name))


if __name__ == "__main__":
    unittest.main()
