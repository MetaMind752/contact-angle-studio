"""Launcher process/ownership tests; no application server or installer is started."""

import argparse
import errno
import importlib.util
import io
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def launcher():
    source = Path(__file__).resolve().parents[1] / "scripts" / "launch_studio.py"
    spec = importlib.util.spec_from_file_location("studio_launcher_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_app_identity_includes_data_library(launcher, tmp_path):
    data = tmp_path / "data with spaces"
    assert launcher.same_studio({"name": launcher.APP_NAME, "data_dir": str(data)}, data)
    assert not launcher.same_studio({"name": "Other app", "data_dir": str(data)}, data)
    assert not launcher.same_studio({"name": launcher.APP_NAME, "data_dir": str(tmp_path)}, data)
    assert not launcher.same_studio({"name": launcher.APP_NAME}, data)
    assert not launcher.same_studio(None, data)


@pytest.mark.parametrize("url", ["https://127.0.0.1:8765", "http://example.org", "http://127.0.0.1:8765@evil.example"])
def test_service_probe_never_contacts_nonloopback(launcher, monkeypatch, url):
    monkeypatch.setattr(launcher.urllib.request, "build_opener", lambda *_: pytest.fail("External request"))
    assert launcher.probe_info(url) is None


@pytest.mark.parametrize("payload,expected", [
    ({"name": "Contact Angle Studio", "version": "0.3.0"}, True),
    ({"name": "Different program"}, False),
    (["Contact Angle Studio"], False),
    ("malformed", False),
])
def test_probe_validates_returned_app(launcher, monkeypatch, payload, expected):
    body = b"malformed" if payload == "malformed" else json.dumps(payload).encode()

    class Opener:
        def open(self, request, timeout):
            assert request.full_url == "http://127.0.0.1:8765/api/info"
            assert timeout <= 1
            return io.BytesIO(body)

    monkeypatch.setattr(launcher.urllib.request, "build_opener", lambda *_: Opener())
    assert bool(launcher.probe_info("http://127.0.0.1:8765")) is expected


@pytest.mark.parametrize("url", ["http://example.org:8765", "http://127.0.0.1:8765/other", "http://127.0.0.1:8765?x=1", "http://user:password@127.0.0.1:8765", "http://127.0.0.1:99999"])
def test_state_cannot_open_remote_or_arbitrary_urls(launcher, tmp_path, url):
    path = tmp_path / "launcher-state.json"
    launcher.write_state(path, {"url": url})
    assert launcher.state_url(path) is None


def test_state_atomic_write_and_valid_url(launcher, tmp_path):
    path = tmp_path / "launcher-state.json"
    launcher.write_state(path, {"launch_id": "first", "url": "http://127.0.0.1:8765/"})
    launcher.write_state(path, {"launch_id": "second", "url": "http://127.0.0.1:8766"})
    assert json.loads(path.read_text())["launch_id"] == "second"
    assert launcher.state_url(path) == "http://127.0.0.1:8766"
    assert list(tmp_path.iterdir()) == [path]


def test_repeat_launcher_reuses_owned_studio(launcher, tmp_path, monkeypatch):
    lock, existing = launcher.acquire_lock(tmp_path, timeout=1, log_path=tmp_path / "log")
    assert lock and existing is None
    try:
        launcher.write_state(tmp_path / "launcher-state.json", {"url": "http://127.0.0.1:8766"})
        monkeypatch.setattr(launcher, "probe_info", lambda *_: {"name": launcher.APP_NAME, "data_dir": str(tmp_path)})
        second_lock, url = launcher.acquire_lock(tmp_path, timeout=1, log_path=tmp_path / "log")
        assert second_lock is None
        assert url == "http://127.0.0.1:8766"
    finally:
        lock.close()


def test_occupied_foreign_port_is_left_alone(launcher, tmp_path, monkeypatch):
    sockets = []

    class Socket:
        def __init__(self, *_):
            self.closed = False
            sockets.append(self)

        def setsockopt(self, *_):
            pass

        def bind(self, address):
            assert address[0] == "127.0.0.1"
            if address[1] == 8765:
                raise OSError(errno.EADDRINUSE, "foreign service")
            self.address = address

        def listen(self, _backlog):
            pass

        def close(self):
            self.closed = True

    monkeypatch.setattr(launcher.socket, "socket", Socket)
    monkeypatch.setattr(launcher, "probe_info", lambda *_: {"name": "Different app"})
    url, reserved = launcher.reserve_port(8765, tmp_path)
    assert url == "http://127.0.0.1:8766"
    assert sockets[0].closed and not reserved.closed


def test_existing_matching_app_needs_no_socket(launcher, tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "probe_info", lambda *_: {"name": launcher.APP_NAME, "data_dir": str(tmp_path)})
    monkeypatch.setattr(launcher.socket, "socket", lambda *_: pytest.fail("Extra server/socket"))
    assert launcher.reserve_port(8765, tmp_path) == ("http://127.0.0.1:8765", None)


def test_missing_dependencies_install_only_in_project_venv(launcher, tmp_path, monkeypatch):
    project = tmp_path / "project with spaces"
    project.mkdir()
    base = project / "base python"
    base.touch()
    local = project / ".venv/bin/python"
    commands = []
    monkeypatch.setattr(launcher, "runtime_probe", lambda p: {"version": [3, 13], "missing": [] if Path(p) == local else ["numpy"], "is_venv": Path(p) == local})

    def setup(command, _cwd, _log):
        commands.append(command)
        if "venv" in command:
            local.parent.mkdir(parents=True)
            local.touch()

    monkeypatch.setattr(launcher, "run_setup", setup)
    runtime = launcher.ensure_runtime(project, str(base), True, False, None)
    assert runtime == local
    assert commands[0] == [str(base), "-m", "venv", str(project / ".venv")]
    assert commands[1][0] == str(local)
    assert commands[1][-1] == str(project)
    assert "--no-input" in commands[1]


def test_no_install_fails_without_setup_mutation(launcher, tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "runtime_probe", lambda _: {"version": [3, 13], "missing": ["cv2"]})
    monkeypatch.setattr(launcher, "run_setup", lambda *_: pytest.fail("Unrequested installation"))
    with pytest.raises(launcher.LauncherError, match="依赖缺失"):
        launcher.ensure_runtime(tmp_path, None, False, False, None)
    assert not (tmp_path / ".venv").exists()


def test_invalid_venv_never_installs_into_global_python(launcher, tmp_path, monkeypatch):
    local = tmp_path / ".venv/bin/python"
    local.parent.mkdir(parents=True)
    local.touch()
    monkeypatch.setattr(launcher, "runtime_probe", lambda _: {"version": [3, 13], "missing": ["cv2"], "is_venv": False})
    monkeypatch.setattr(launcher, "run_setup", lambda *_: pytest.fail("Global installation"))
    with pytest.raises(launcher.LauncherError, match="独立虚拟环境"):
        launcher.ensure_runtime(tmp_path, None, True, False, None)


def test_server_command_keeps_paths_and_fd_separate(launcher):
    command = launcher.server_command("/folder with spaces/python", 17)
    assert command[:3] == ["/folder with spaces/python", "-m", "uvicorn"]
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--fd") + 1] == "17"


def test_failed_child_does_not_wait_or_probe(launcher, tmp_path, monkeypatch):
    class Child:
        def poll(self):
            return 1

    monkeypatch.setattr(launcher, "probe_info", lambda *_: pytest.fail("Probe after child failure"))
    with pytest.raises(launcher.LauncherError, match="启动失败"):
        launcher.wait_ready(Child(), "http://127.0.0.1:8765", tmp_path, 1)


def test_stop_targets_only_owned_child(launcher):
    events = []

    class Child:
        def poll(self):
            return None

        def send_signal(self, sig):
            events.append(sig)

        def wait(self, timeout):
            events.append(timeout)
            if timeout == 8:
                raise subprocess.TimeoutExpired("owned", timeout)
            return 0

        def terminate(self):
            events.append("terminate")

    launcher.stop_child(Child())
    assert events == [signal.SIGINT, 8, "terminate", 3]
    launcher.stop_child(None)


def test_check_mode_has_no_data_directory_side_effect(launcher, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(launcher, "project_root", lambda: tmp_path)
    monkeypatch.setattr(launcher, "ensure_runtime", lambda *_: Path("/test/python"))
    args = argparse.Namespace(data_dir=None, check=True, python=None)
    assert launcher.run(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ok"
    assert not (tmp_path / "data").exists()


@pytest.mark.skipif(not shutil.which("zsh"), reason="macOS shortcut requires zsh")
def test_command_resolves_shortcut_and_folder_with_spaces(tmp_path):
    source = Path(__file__).resolve().parents[1]
    project = tmp_path / "接触角 工作台 with spaces"
    (project / "scripts").mkdir(parents=True)
    shutil.copy2(source / "scripts/launch_studio.py", project / "scripts/launch_studio.py")
    shutil.copy2(source / "启动接触角.command", project / "启动接触角.command")
    shortcut = tmp_path / "快捷方式 有空格.command"
    shortcut.symlink_to(project / "启动接触角.command")
    env = os.environ.copy()
    env["CONTACT_ANGLE_PYTHON"] = sys.executable
    result = subprocess.run([str(shortcut), "--check"], env=env, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["project"] == str(project)
    assert not (project / "data").exists()
