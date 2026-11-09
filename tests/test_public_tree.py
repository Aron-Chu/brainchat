"""Guards for the public tree: no Label Studio export file may sit in the tests, on disk or in git."""

import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PATTERN = "tests/fixtures/ls_export_*"


def test_no_label_studio_export_file_on_disk():
    # Names only: the guard never opens a matching file.
    found = sorted(p.relative_to(REPO_ROOT).as_posix() for p in REPO_ROOT.glob(PATTERN))
    assert found == [], "Label Studio export files must not be in the tree; build exports in code"


def test_no_label_studio_export_file_tracked_by_git():
    git = shutil.which("git")
    if git is None or not (REPO_ROOT / ".git").exists():
        return  # not a git checkout (an unpacked copy): the on-disk guard above still runs
    out = subprocess.run(
        [git, "-C", str(REPO_ROOT), "ls-files", "--", PATTERN],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    assert out == [], "Label Studio export files must not be tracked; build exports in code"


def test_no_tracked_file_is_ignored():
    # A fresh clone or a new root commit drops ignored files, so a tracked file the .gitignore
    # also matches (tests/data/ under a bare ``data/`` rule) would silently go missing.
    git = shutil.which("git")
    if git is None or not (REPO_ROOT / ".git").exists():
        return
    out = subprocess.run(
        [git, "-C", str(REPO_ROOT), "ls-files", "--cached", "--ignored", "--exclude-standard"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    assert out == [], f"tracked files matched by .gitignore: {out}"
