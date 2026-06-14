from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    pytest_tmp = Path(".tmp") / "pytest"
    pytest_tmp.mkdir(parents=True, exist_ok=True)
    commands = [
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "--basetemp", str(pytest_tmp)],
        [sys.executable, "-m", "ruff", "check", "."],
        [sys.executable, "-m", "pyrefly", "check"],
    ]
    for command in commands:
        print(" ".join(command))
        result = subprocess.run(command)
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
