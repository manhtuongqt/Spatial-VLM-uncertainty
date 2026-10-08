#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path


def main() -> None:
    tests = Path(__file__).resolve().parents[1] / "tests"
    failures = []
    executed = 0
    for path in sorted(tests.glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("test_"):
                continue
            executed += 1
            try:
                function()
                print(f"PASS {path.name}::{name}")
            except Exception as error:
                failures.append(f"{path.name}::{name}: {type(error).__name__}: {error}")
                print(f"FAIL {failures[-1]}")
    if failures:
        raise SystemExit(f"{len(failures)}/{executed} tests failed")
    print(f"PASS all {executed} tests")


if __name__ == "__main__":
    main()
