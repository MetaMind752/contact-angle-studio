"""Real detached lifetime checks plus process-identity guard regressions."""
from __future__ import annotations

import argparse
import http.server
import importlib.util
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "launch_studio.py"


@pytest.fixture
def launcher():
    spec = importlib.util.spec_from_file_location("launcher_background_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def args(tmp_path):
    return argparse.Namespace(
        data_dir=str(tmp_path), port=23456, python=sys.executable,
        headless=True, no_install=True, install=False, startup_timeout=20,
        lock_timeout=20, background_timeout=30,
    )


def test_background_command_keeps_options_and_paths_separate(launcher, tmp_path):
    config = args(tmp_path / "图片 库")
    command = launcher.background_command(config, Path(config.data_dir))
    assert command[:2] == [sys.executable, str(SCRIPT)]
    assert "--background-supervisor" in command and "--headless" in command
    assert command[command.index("--data-dir") + 1] == str(tmp_path / "图片 库")
    assert command[command.index("--python") + 1] == sys.executable
    assert "--no-install" in command


@pytest.mark.parametrize("change", ["token", "pid", "data", "legacy"])
def test_instance_identity_requires_token_pid_and_library(launcher, tmp_path, change):
    record = {"launch_id": "a" * 32, "pid": 442}
    info = {"name": launcher.APP_NAME, "data_dir": str(tmp_path),
            "launch_id": record["launch_id"], "server_pid": 442}
    assert launcher.matching_instance(info, record, tmp_path)
    if change == "token":
        info["launch_id"] = "b" * 32
    elif change == "pid":
        info["server_pid"] = 443
    elif change == "data":
        info["data_dir"] = str(tmp_path / "other")
    else:
        info.pop("launch_id")
    assert not launcher.matching_instance(info, record, tmp_path)


@pytest.mark.parametrize("kind", ["token", "signature"])
def test_stop_refuses_mismatched_identity_without_signaling(launcher, tmp_path, monkeypatch, kind):
    config = args(tmp_path)
    record = {"launch_id": "a" * 32, "pid": 442, "server_signature": "original"}
    info = {"name": launcher.APP_NAME, "data_dir": str(tmp_path),
            "launch_id": "a" * 32, "server_pid": 442}
    if kind == "token":
        info["launch_id"] = "b" * 32
    monkeypatch.setattr(launcher, "managed_status", lambda _: (record, "http://127.0.0.1:23456", info))
    monkeypatch.setattr(launcher, "process_signature", lambda _: "changed")
    monkeypatch.setattr(launcher.os, "kill", lambda *_: pytest.fail("Signaled unverified process"))
    with pytest.raises(launcher.LauncherError, match="不匹配"):
        launcher.stop_studio(config)


@pytest.mark.parametrize("pid", [None, True, 0, 1, -12])
def test_process_signature_rejects_unusable_pids(launcher, monkeypatch, pid):
    monkeypatch.setattr(launcher.subprocess, "run", lambda *_a, **_k: pytest.fail("Unexpected ps"))
    assert launcher.process_signature(pid) is None


def test_status_does_not_create_data_directory(launcher, tmp_path, capsys):
    directory = tmp_path / "not created"
    assert launcher.show_status(args(directory)) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["running"] is False and status["managed"] is False
    assert not directory.exists()


def invoke(data, port, *options):
    command = [sys.executable, str(SCRIPT), "--data-dir", str(data), "--port", str(port),
               "--python", sys.executable, "--no-install", "--headless", *options]
    return subprocess.run(command, capture_output=True, text=True, timeout=40, check=False)


def cleanup_owned(launcher, data, port):
    current = launcher.read_state(data)
    if current:
        info = launcher.probe_info(current["url"])
        if launcher.matching_instance(info, current, data):
            result = invoke(data, port, "--stop")
            assert result.returncode == 0, result.stderr


@pytest.mark.skipif(os.name != "posix", reason="Detached CLI supports POSIX hosts")
def test_background_parent_returns_and_service_survives_reopen_and_stop(launcher, tmp_path):
    data = tmp_path / "data with spaces"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    record = None
    try:
        env = os.environ.copy()
        env["CONTACT_ANGLE_PYTHON"] = sys.executable
        result = subprocess.run(
            [str(PROJECT / "启动接触角.command"), "--data-dir", str(data), "--port", str(port),
             "--no-install", "--headless", "--background-timeout", "30"],
            capture_output=True, text=True, timeout=40, check=False, env=env,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        record = launcher.read_state(data)
        assert record["mode"] == "background"
        assert os.getsid(record["supervisor_pid"]) == record["supervisor_pid"]
        assert os.getsid(record["pid"]) == record["pid"]
        time.sleep(0.3)
        assert launcher.matching_instance(launcher.probe_info(record["url"]), record, data)
        again = invoke(data, port, "--background")
        assert again.returncode == 0, again.stderr
        assert launcher.read_state(data)["pid"] == record["pid"]
        assert launcher.read_state(data)["supervisor_pid"] == record["supervisor_pid"]
        status = invoke(data, port, "--status")
        assert status.returncode == 0
        assert json.loads(status.stdout)["managed"] is True
        launcher.write_state(data / "launcher-state.json", {**record, "server_signature": "wrong"})
        refused = invoke(data, port, "--stop")
        assert refused.returncode == 1
        assert launcher.matching_instance(launcher.probe_info(record["url"]), record, data)
        launcher.write_state(data / "launcher-state.json", record)
        stopped = invoke(data, port, "--stop")
        assert stopped.returncode == 0, stopped.stdout + stopped.stderr
        assert launcher.probe_info(record["url"]) is None
        assert not (data / "launcher-state.json").exists()
    finally:
        if record and launcher.read_state(data):
            launcher.write_state(data / "launcher-state.json", record)
        cleanup_owned(launcher, data, port)


@pytest.mark.skipif(os.name != "posix", reason="Detached CLI supports POSIX hosts")
def test_foreign_port_stays_alive_while_owned_service_is_stopped(launcher, tmp_path):
    class Foreign(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"name":"Unrelated service"}')

        def log_message(self, *_args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Foreign)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    data = tmp_path / "private data"
    try:
        result = invoke(data, port, "--background", "--background-timeout", "30")
        assert result.returncode == 0, result.stdout + result.stderr
        record = launcher.read_state(data)
        assert record["url"] != f"http://127.0.0.1:{port}"
        assert invoke(data, port, "--stop").returncode == 0
        assert thread.is_alive()
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            pass
    finally:
        cleanup_owned(launcher, data, port)
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
