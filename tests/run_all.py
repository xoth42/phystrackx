"""
run_all.py

Runs every tests/test_*.py module in its own process and prints a summary.
Each GUI module needs its own process: a second app in one process fails on cached button images.

Usage (from the repo root):
    uv run python -m tests.run_all              # everything
    uv run python -m tests.run_all test_tracking   # one or more modules by name

Author: Isam Balghari
"""

import glob
import os
import subprocess
import sys

from tests import harness


def main() -> None:
    harness.setup_env()
    names = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(harness.TESTS_DIR, "test_*.py")))
    wanted = sys.argv[1:]
    results = {}
    for name in names:
        if wanted and name not in wanted:
            continue
        print(f"\n=== {name}", flush=True)
        results[name] = subprocess.run([sys.executable, "-m", f"tests.{name}"],
                                       cwd=harness.REPO, env=os.environ).returncode

    print("\n=== summary")
    for name, rc in results.items():
        print(f"  {'OK  ' if rc == 0 else 'FAIL'} {name}")
    sys.exit(1 if any(rc != 0 for rc in results.values()) else 0)


if __name__ == "__main__":
    main()
