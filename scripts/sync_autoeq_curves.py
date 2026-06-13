from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUTOEQ_REPO_URL = "https://github.com/jaakkopasanen/AutoEq.git"
REPO_DIR = ROOT / ".tmp" / "autoeq"
DEVICE_DESTINATION = ROOT / "curves" / "devices" / "autoeq"
TARGET_DESTINATION = ROOT / "curves" / "targets" / "autoeq"
MANIFEST_PATH = ROOT / "curves" / "autoeq_manifest.json"

sys.path.insert(0, str(ROOT))

from source.curves import SUPPORTED_CURVE_SUFFIXES, load_curve_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync official AutoEQ measurements into local AIEQ curve folders.")
    parser.add_argument("--repo-url", default=AUTOEQ_REPO_URL)
    parser.add_argument("--repo-dir", type=Path, default=REPO_DIR)
    parser.add_argument("--keep-existing", action="store_true", help="Do not clear generated AutoEQ curve folders before copying.")
    args = parser.parse_args()

    repo_dir = args.repo_dir.resolve()
    _ensure_repo(repo_dir, args.repo_url)
    commit = _git(repo_dir, "rev-parse", "--short", "HEAD", capture=True).strip()

    measurements_dir = repo_dir / "measurements"
    targets_dir = repo_dir / "targets"
    device_count = _copy_curve_tree(measurements_dir, DEVICE_DESTINATION, clean=not args.keep_existing)
    target_count = _copy_curve_tree(targets_dir, TARGET_DESTINATION, clean=not args.keep_existing)

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "repo_url": args.repo_url,
                "commit": commit,
                "synced_at": datetime.now(UTC).isoformat(),
                "devices": device_count,
                "targets": target_count,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"AutoEQ synced: {device_count} device curves, {target_count} target curves, commit {commit}.")
    return 0


def _ensure_repo(repo_dir: Path, repo_url: str) -> None:
    if (repo_dir / ".git").exists():
        _git(repo_dir, "sparse-checkout", "set", "measurements", "targets")
        _git(repo_dir, "pull", "--ff-only")
        return

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "git",
            "-c",
            "core.longpaths=true",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            repo_url,
            str(repo_dir),
        ]
    )
    _git(repo_dir, "sparse-checkout", "set", "measurements", "targets")


def _copy_curve_tree(source_root: Path, destination_root: Path, *, clean: bool) -> int:
    if clean and destination_root.exists():
        _safe_rmtree(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    if not source_root.exists():
        return 0

    copied = 0
    for source_path in sorted(source_root.rglob("*")):
        if not source_path.is_file() or source_path.suffix.casefold() not in SUPPORTED_CURVE_SUFFIXES:
            continue
        if not _is_valid_curve(source_path):
            continue
        relative = source_path.relative_to(source_root)
        target_path = destination_root / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        copied += 1
    return copied


def _is_valid_curve(path: Path) -> bool:
    try:
        curve = load_curve_file(path)
    except (OSError, UnicodeDecodeError, ValueError):
        return False
    return curve.freqs.size >= 2


def _git(repo_dir: Path, *args: str, capture: bool = False) -> str:
    return _run(["git", "-C", str(repo_dir), *args], capture=capture)


def _run(command: list[str], *, capture: bool = False) -> str:
    result = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    return result.stdout if capture and result.stdout is not None else ""


def _safe_rmtree(path: Path) -> None:
    resolved_root = ROOT.resolve()
    resolved_path = path.resolve()
    if not str(resolved_path).casefold().startswith(str(resolved_root).casefold()):
        raise RuntimeError(f"Refusing to delete outside workspace: {resolved_path}")
    shutil.rmtree(resolved_path)


if __name__ == "__main__":
    raise SystemExit(main())
