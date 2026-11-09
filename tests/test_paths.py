import os
from pathlib import Path

import pytest

from chatbrain import paths

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBDIRS = ("inbox", "imported", "models", "labelstudio")


def test_data_dir_refuses_onedrive(tmp_path):
    with pytest.raises(paths.UnsafeDataDir): paths.data_dir({"CHATBRAIN_DATA": str(tmp_path / "OneDrive" / "cb")})


def test_data_dir_refuses_work_onedrive_any_case(tmp_path):
    target = tmp_path / "onedrive - Contoso" / "cb"
    with pytest.raises(paths.UnsafeDataDir):
        paths.data_dir({"CHATBRAIN_DATA": str(target)})
    assert not target.exists()


def test_data_dir_refuses_inside_repo():
    target = REPO_ROOT / "data"
    with pytest.raises(paths.UnsafeDataDir):
        paths.data_dir({"CHATBRAIN_DATA": str(target)})
    assert not target.exists()


def test_data_dir_refuses_repo_root_itself():
    with pytest.raises(paths.UnsafeDataDir):
        paths.data_dir({"CHATBRAIN_DATA": str(REPO_ROOT)})


def test_data_dir_refuses_folder_holding_the_repo(tmp_path, monkeypatch):
    fake_repo = tmp_path / "work" / "chat-brain"
    fake_repo.mkdir(parents=True)
    (fake_repo / "pyproject.toml").write_text("", encoding="utf-8")
    monkeypatch.setattr(paths, "REPO_ROOT", fake_repo)
    with pytest.raises(paths.UnsafeDataDir):
        paths.data_dir({"CHATBRAIN_DATA": str(tmp_path / "work")})
    assert not (tmp_path / "work" / "inbox").exists()


def test_data_dir_refuses_git_worktree(tmp_path):
    (tmp_path / ".git").mkdir()
    target = tmp_path / "nested" / "cb"
    with pytest.raises(paths.UnsafeDataDir):
        paths.data_dir({"CHATBRAIN_DATA": str(target)})
    assert not target.exists()


def test_data_dir_refuses_git_file_marker(tmp_path):
    # Linked worktrees and submodules mark their root with a `.git` file.
    (tmp_path / ".git").write_text("gitdir: elsewhere", encoding="utf-8")
    with pytest.raises(paths.UnsafeDataDir):
        paths.data_dir({"CHATBRAIN_DATA": str(tmp_path / "cb")})


def test_data_dir_refuses_unset_or_empty():
    with pytest.raises(paths.UnsafeDataDir):
        paths.data_dir({})
    with pytest.raises(paths.UnsafeDataDir):
        paths.data_dir({"CHATBRAIN_DATA": "  "})


def test_data_dir_refuses_relative_path():
    with pytest.raises(paths.UnsafeDataDir):
        paths.data_dir({"CHATBRAIN_DATA": "brainchat-data"})


def test_data_dir_creates_subdirs(tmp_path):
    target = tmp_path / "cb"
    got = paths.data_dir({"CHATBRAIN_DATA": str(target)})
    assert got == target.resolve()
    for name in SUBDIRS:
        assert (got / name).is_dir()
    # A second call is a no-op on an existing folder.
    assert paths.data_dir({"CHATBRAIN_DATA": str(target)}) == got


def test_data_dir_fails_closed_without_repo_root(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path / "no-such-repo")
    with pytest.raises(paths.UnsafeDataDir):
        paths.data_dir({"CHATBRAIN_DATA": str(tmp_path / "cb")})


# --- path normalisation (Task 1 review minors) --------------------------------------------

EXT = "\\\\?\\"  # the Windows extended-length prefix
ONEDRIVE_VARS = ("OneDrive", "OneDriveCommercial", "OneDriveConsumer")


def _fake_repo(tmp_path, monkeypatch) -> Path:
    """A repo root with pyproject.toml and no .git, so the .git rule can't refuse for it."""
    fake_repo = tmp_path / "work" / "chat-brain"
    fake_repo.mkdir(parents=True)
    (fake_repo / "pyproject.toml").write_text("", encoding="utf-8")
    monkeypatch.setattr(paths, "REPO_ROOT", fake_repo)
    assert not any((p / ".git").exists() for p in (fake_repo, *fake_repo.parents))
    return fake_repo


def test_data_dir_refuses_inside_repo_without_git(tmp_path, monkeypatch):
    fake_repo = _fake_repo(tmp_path, monkeypatch)
    inside = fake_repo / "data"
    targets = (
        str(inside),
        str(fake_repo),
        str(inside).upper(),
        EXT + str(inside),
        "//?/" + str(inside).replace(os.sep, "/"),
    )
    for target in targets:
        with pytest.raises(paths.UnsafeDataDir, match="inside the chat-brain repo"):
            paths.data_dir({"CHATBRAIN_DATA": target})
    assert not inside.exists()


def test_data_dir_refuses_folder_holding_the_repo_extended_path(tmp_path, monkeypatch):
    _fake_repo(tmp_path, monkeypatch)
    for target in (EXT + str(tmp_path / "work"), EXT + str(tmp_path / "work").upper(), EXT + str(tmp_path)):
        with pytest.raises(paths.UnsafeDataDir, match="holds the chat-brain repo"):
            paths.data_dir({"CHATBRAIN_DATA": target})
    assert not (tmp_path / "work" / "inbox").exists()
    assert not (tmp_path / "inbox").exists()


def test_data_dir_accepts_an_extended_length_path(tmp_path):
    target = tmp_path / "cb"
    got = paths.data_dir({"CHATBRAIN_DATA": EXT + str(target)})
    assert got == target.resolve()
    assert not str(got).startswith(EXT)
    assert (target / "inbox").is_dir()


@pytest.mark.parametrize("segment", ["cb.", "cb ", "cb..", "cb. ", ".", ".."])
@pytest.mark.parametrize("prefix", ["", EXT])
def test_data_dir_refuses_a_segment_ending_in_a_dot_or_space(tmp_path, segment, prefix):
    target = prefix + str(tmp_path) + os.sep + segment + os.sep + "inner"
    with pytest.raises(paths.UnsafeDataDir, match="ends in a space or a dot"):
        paths.data_dir({"CHATBRAIN_DATA": target})
    assert sorted(p.name for p in tmp_path.iterdir()) == []


@pytest.mark.parametrize("prefix", ["", EXT])
def test_data_dir_trailing_dot_cannot_reach_the_repo(tmp_path, monkeypatch, prefix):
    # Windows drops a trailing dot, so "chat-brain." can name the repo itself; with the
    # extended-length prefix it would instead create a look-alike folder beside it.
    fake_repo = _fake_repo(tmp_path, monkeypatch)
    with pytest.raises(paths.UnsafeDataDir):
        paths.data_dir({"CHATBRAIN_DATA": prefix + str(fake_repo) + "." + os.sep + "data"})
    assert not (fake_repo / "data").exists()
    assert sorted(p.name for p in fake_repo.parent.iterdir()) == ["chat-brain"]


@pytest.mark.parametrize("var", ONEDRIVE_VARS)
def test_data_dir_refuses_folders_under_onedrive_env_vars(tmp_path, monkeypatch, var):
    for v in ONEDRIVE_VARS:
        monkeypatch.delenv(v, raising=False)
    synced = tmp_path / "Synced Files"  # no "OneDrive" segment: only the env rule can refuse
    synced.mkdir()
    monkeypatch.setenv(var, str(synced))
    for target in (synced, synced / "cb", str(synced / "cb").upper()):
        with pytest.raises(paths.UnsafeDataDir, match=var):
            paths.data_dir({"CHATBRAIN_DATA": str(target)})
    monkeypatch.setenv(var, EXT + str(synced))
    with pytest.raises(paths.UnsafeDataDir, match=var):
        paths.data_dir({"CHATBRAIN_DATA": str(synced / "cb")})
    assert not (synced / "cb").exists()
    # A sibling whose name only starts the same way is not under it.
    sibling = tmp_path / "Synced Files2" / "cb"
    assert paths.data_dir({"CHATBRAIN_DATA": str(sibling)}) == sibling.resolve()


def test_data_dir_reads_onedrive_vars_from_the_given_env_too(tmp_path, monkeypatch):
    for v in ONEDRIVE_VARS:
        monkeypatch.delenv(v, raising=False)
    synced = tmp_path / "Synced Files"
    with pytest.raises(paths.UnsafeDataDir, match="OneDriveCommercial"):
        paths.data_dir({"CHATBRAIN_DATA": str(synced / "cb"), "OneDriveCommercial": str(synced)})
    assert not (synced / "cb").exists()


def test_data_dir_ignores_unset_empty_or_relative_onedrive_vars(tmp_path, monkeypatch):
    for v in ONEDRIVE_VARS:
        monkeypatch.delenv(v, raising=False)
    target = tmp_path / "cb"
    env = {"CHATBRAIN_DATA": str(target), "OneDrive": "", "OneDriveCommercial": "   ", "OneDriveConsumer": "cb"}
    assert paths.data_dir(env) == target.resolve()


def test_resolve_checks_like_data_dir_and_creates_nothing(tmp_path, monkeypatch):
    target = tmp_path / "cb"
    assert paths.resolve({"CHATBRAIN_DATA": str(target)}) == target.resolve()
    assert not target.exists()  # checked, nothing made
    fake_repo = _fake_repo(tmp_path, monkeypatch)
    for unsafe in ({}, {"CHATBRAIN_DATA": "cb"}, {"CHATBRAIN_DATA": str(fake_repo / "data")},
                   {"CHATBRAIN_DATA": str(tmp_path / "OneDrive" / "cb")}):
        with pytest.raises(paths.UnsafeDataDir):
            paths.resolve(unsafe)
    assert paths.data_dir({"CHATBRAIN_DATA": str(target)}) == paths.resolve({"CHATBRAIN_DATA": str(target)})
