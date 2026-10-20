"""Code evolution infrastructure for hermes-agent tool files.

Uses Darwinian Evolver (external CLI, AGPL v3) to evolve Python source files.
Darwinian Evolver is NEVER imported as a Python module — always called as a
subprocess to respect its AGPL v3 license boundary.

Safety invariants:
  - Full test suite must pass (zero tolerance)
  - Function signatures must not change (no breaking API changes)
  - registry.register() calls must not change (no tool discovery changes)
  - All changes go through PRs — no direct commits
"""

import ast
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CodeTarget:
    """Describes a code evolution target."""
    source_file: Path               # Relative to hermes-agent repo root
    target_functions: list[str] = field(default_factory=list)  # Functions to evolve (empty = all)
    test_command: str = "python -m pytest tests/ -q --tb=short --ignore=tests/integration --ignore=tests/e2e --ignore=tests/stress --ignore=tests/manual --ignore=tests/gateway --ignore=tests/ci -n=0"
    bug_description: str = ""       # What bug/edge case we're targeting


@dataclass
class CodeFitnessResult:
    """Result from fitness evaluation of evolved code."""
    score: float                    # 0.0–1.0
    tests_passed: bool
    signatures_unchanged: bool
    details: str


class CodeFitnessEvaluator:
    """Evaluates evolved Python code against hard safety gates.

    Gates (in order — later gates are skipped if earlier ones fail):
    1. Signature preservation: function signatures + registry calls must be identical
    2. Test suite: full pytest must pass (zero tolerance)

    A variant that passes both gates scores 1.0. Any failure scores 0.0.
    """

    def __init__(self, hermes_repo: Path, test_timeout: int = 300):
        self.hermes_repo = hermes_repo
        self.test_timeout = test_timeout

    def score(self, evolved_code: str, target: CodeTarget) -> CodeFitnessResult:
        """Score evolved code. Returns 0.0 if any gate fails.

        Args:
            evolved_code: The evolved Python source code as a string.
            target: The CodeTarget describing what was evolved.

        Returns:
            CodeFitnessResult with score and diagnostic details.
        """
        original_path = self.hermes_repo / target.source_file

        # Read original for signature comparison
        try:
            original_code = original_path.read_text()
        except OSError as e:
            return CodeFitnessResult(
                score=0.0,
                tests_passed=False,
                signatures_unchanged=False,
                details=f"Cannot read original file: {e}",
            )

        # Gate 1: signature preservation
        if not check_signatures_unchanged(original_code, evolved_code):
            return CodeFitnessResult(
                score=0.0,
                tests_passed=False,
                signatures_unchanged=False,
                details="FAILED: function signatures or registry.register() calls changed",
            )

        # Gate 2: write evolved code to temp file and run tests
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            dir=original_path.parent,
            delete=False,
            prefix=f"_evolve_{original_path.stem}_",
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(evolved_code)

        try:
            # Temporarily replace the original file
            backup = original_path.read_bytes()
            original_path.write_text(evolved_code)

            test_result = subprocess.run(
                target.test_command.split(),
                capture_output=True,
                text=True,
                timeout=self.test_timeout,
                cwd=str(self.hermes_repo),
            )
            tests_passed = test_result.returncode == 0

            if tests_passed:
                last_lines = test_result.stdout.strip().split("\n")[-3:] if test_result.stdout else []
                details = "Tests PASSED: " + " | ".join(last_lines)
            else:
                last_lines = test_result.stdout.strip().split("\n")[-5:] if test_result.stdout else []
                details = "Tests FAILED: " + " | ".join(last_lines)

        except subprocess.TimeoutExpired:
            tests_passed = False
            details = f"Tests timed out ({self.test_timeout}s)"
        except Exception as e:
            tests_passed = False
            details = f"Test run error: {e}"
        finally:
            # Always restore the original file
            original_path.write_bytes(backup)
            tmp_path.unlink(missing_ok=True)

        return CodeFitnessResult(
            score=1.0 if tests_passed else 0.0,
            tests_passed=tests_passed,
            signatures_unchanged=True,
            details=details,
        )


def check_signatures_unchanged(original_code: str, evolved_code: str) -> bool:
    """Check that function signatures and registry.register() calls are identical.

    Compares AST-level function definitions to detect any signature changes
    (added/removed/renamed parameters). Also checks that all
    registry.register() call sites are structurally identical.

    Args:
        original_code: The original Python source.
        evolved_code: The evolved Python source.

    Returns:
        True if all signatures and registry calls are unchanged.
    """
    try:
        orig_tree = ast.parse(original_code)
        evolved_tree = ast.parse(evolved_code)
    except SyntaxError:
        return False

    orig_sigs = _extract_function_signatures(orig_tree)
    evolved_sigs = _extract_function_signatures(evolved_tree)

    # All original signatures must be present and identical
    for name, sig in orig_sigs.items():
        if name not in evolved_sigs:
            return False
        if evolved_sigs[name] != sig:
            return False

    # registry.register() call argument names must be unchanged
    orig_registrations = _extract_registry_calls(orig_tree)
    evolved_registrations = _extract_registry_calls(evolved_tree)

    if set(orig_registrations.keys()) != set(evolved_registrations.keys()):
        return False

    for name in orig_registrations:
        if orig_registrations[name] != evolved_registrations[name]:
            return False

    return True


def _extract_function_signatures(tree: ast.AST) -> dict[str, str]:
    """Extract {function_name: signature_str} from an AST."""
    signatures = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # Build canonical signature: name(args, *, kwargs)
        args = node.args
        parts = []

        # Positional args
        n_defaults = len(args.defaults)
        n_no_default = len(args.args) - n_defaults
        for i, arg in enumerate(args.args):
            if i >= n_no_default:
                default_idx = i - n_no_default
                default = ast.unparse(args.defaults[default_idx])
                parts.append(f"{arg.arg}={default}")
            else:
                parts.append(arg.arg)

        if args.vararg:
            parts.append(f"*{args.vararg.arg}")
        elif args.kwonlyargs:
            parts.append("*")

        for i, kwarg in enumerate(args.kwonlyargs):
            if i < len(args.kw_defaults) and args.kw_defaults[i] is not None:
                default = ast.unparse(args.kw_defaults[i])
                parts.append(f"{kwarg.arg}={default}")
            else:
                parts.append(kwarg.arg)

        if args.kwarg:
            parts.append(f"**{args.kwarg.arg}")

        sig = f"{node.name}({', '.join(parts)})"
        signatures[node.name] = sig

    return signatures


def _extract_registry_calls(tree: ast.AST) -> dict[str, frozenset]:
    """Extract {tool_name: frozenset(kwarg_names)} for registry.register() calls."""
    registrations = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Match registry.register(...)
        func = node.func
        is_registry_call = (
            isinstance(func, ast.Attribute)
            and func.attr == "register"
            and isinstance(func.value, ast.Name)
            and func.value.id == "registry"
        )
        if not is_registry_call:
            continue

        # Extract the 'name' keyword argument to identify the tool
        name_val = None
        kwarg_names = set()
        for kw in node.keywords:
            kwarg_names.add(kw.arg)
            if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                name_val = kw.value.value

        if name_val:
            registrations[name_val] = frozenset(kwarg_names)

    return registrations


def _write_fitness_script(
    hermes_repo: Path,
    target: CodeTarget,
    output_path: Path,
) -> None:
    """Write a self-contained fitness evaluation script for Darwinian Evolver.

    Darwinian Evolver calls this script with the evolved file as stdin or argument.
    The script exits 0 if the evolved code passes all gates, non-zero otherwise.
    Outputs a float score (0.0–1.0) to stdout.
    """
    script = f"""#!/usr/bin/env python3
\"\"\"Fitness evaluation script for Darwinian Evolver.\"\"\"
import sys
import subprocess
from pathlib import Path

HERMES_REPO = Path({str(hermes_repo)!r})
TARGET_FILE = HERMES_REPO / {str(target.source_file)!r}
TEST_COMMAND = {target.test_command!r}.split()
TIMEOUT = 300

evolved_file = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if evolved_file is None:
    print("0.0"); sys.exit(1)

evolved_code = evolved_file.read_text()

# Check syntax
try:
    import ast; ast.parse(evolved_code)
except SyntaxError:
    print("0.0"); sys.exit(1)

# Signature check
sys.path.insert(0, str(Path(__file__).parent))
from evolution.code.code_organism import check_signatures_unchanged
original = TARGET_FILE.read_text()
if not check_signatures_unchanged(original, evolved_code):
    print("0.0"); sys.exit(1)

# Test suite
backup = TARGET_FILE.read_bytes()
TARGET_FILE.write_text(evolved_code)
try:
    r = subprocess.run(TEST_COMMAND, capture_output=True, timeout=TIMEOUT, cwd=str(HERMES_REPO))
    score = "1.0" if r.returncode == 0 else "0.0"
finally:
    TARGET_FILE.write_bytes(backup)

print(score)
sys.exit(0 if score == "1.0" else 1)
"""
    output_path.write_text(script)
    output_path.chmod(0o755)
