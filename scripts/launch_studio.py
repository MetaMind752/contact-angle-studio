#!/usr/bin/env python3
"""Launch the local Contact Angle Studio, including first-run environment setup.

This file uses only the Python standard library so it also runs before the
application dependencies are installed. The child server is bound to loopback;
only this launcher's own child process is stopped during cleanup.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser
from pathlib import Path

APP_NAME = "Contact Angle Studio"
HOST = "127.0.0.1"
MIN_PYTHON = (3, 11)
DEPENDENCIES = ("fastapi", "uvicorn", "numpy", "scipy", "cv2", "pydantic", "PIL", "multipart")


class LauncherError(Exception):
    """A launch/setup failure with an actionable user-facing message."""


def project_root():
    return Path(__file__).resolve().parents[1]


def announce(message, log_path=None):
    print(message, flush=True)
    if log_path is not None and not os.environ.get("CONTACT_ANGLE_LOG_REDIRECTED"):
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(time.strftime("%Y-%m-%d %H:%M:%S ") + message + "\n")


def probe_info(url, timeout=0.4):
    """Read only a loopback URL and never reuse an unrelated HTTP service."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "http" or parsed.hostname != HOST or parsed.username or parsed.password:
        return None
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, _request, _fp, _code, _message, _headers, _newurl):
            return None

    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        request = urllib.request.Request(url.rstrip("/") + "/api/info")
        with opener.open(request, timeout=timeout) as response:
            payload = response.read(65_537)
        if len(payload) > 65_536:
            return None
        result = json.loads(payload)
        return result if isinstance(result, dict) and result.get("name") == APP_NAME else None
    except (OSError, ValueError, urllib.error.URLError):
        return None


def same_studio(info, data_dir):
    if not info or info.get("name") != APP_NAME or not isinstance(info.get("data_dir"), str):
        return False
    return Path(info["data_dir"]).expanduser().resolve() == data_dir.resolve()


def state_url(path):
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        url = record.get("url", "")
        parsed = urllib.parse.urlparse(url)
        if (
            parsed.scheme == "http"
            and parsed.hostname == HOST
            and parsed.port
            and not parsed.username
            and not parsed.password
            and parsed.path in ("", "/")
            and not parsed.query
            and not parsed.fragment
        ):
            return url.rstrip("/")
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return None


def write_state(path, record):
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as temp:
        temp.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        temporary = Path(temp.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def acquire_lock(data_dir, timeout, log_path):
    """Return an owned lock, or an already-ready matching Studio URL."""
    try:
        import fcntl
    except ImportError as exc:
        raise LauncherError("双击启动器目前适用于 macOS/Linux；其他系统请按说明运行 API。") from exc
    lock = (data_dir / ".launcher.lock").open("a+")
    deadline = time.monotonic() + timeout
    waiting = False
    while True:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lock, None
        except BlockingIOError:
            url = state_url(data_dir / "launcher-state.json")
            if url and same_studio(probe_info(url), data_dir):
                lock.close()
                return None, url
            if not waiting:
                announce("另一个启动窗口正在准备同一资料库，等待它就绪……", log_path)
                waiting = True
            if time.monotonic() >= deadline:
                lock.close()
                raise LauncherError(
                    "另一个启动窗口仍在安装或启动。请查看原启动窗口和 launcher.log，稍后再次打开。"
                )
            time.sleep(0.25)


def reserve_port(preferred, data_dir, previous_url=None):
    """Find a matching running app, otherwise reserve a private loopback socket."""
    candidates = []
    if previous_url and same_studio(probe_info(previous_url), data_dir):
        return previous_url, None
    for port in range(preferred, min(65_536, preferred + 21)):
        url = f"http://{HOST}:{port}"
        info = probe_info(url)
        if same_studio(info, data_dir):
            return url, None
        candidates.append((port, url))
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((HOST, port))
            sock.listen(128)
            return url, sock
        except OSError as exc:
            sock.close()
            if exc.errno not in (errno.EADDRINUSE, errno.EACCES):
                raise LauncherError(f"无法监听本机端口：{exc}") from exc
    ports = ", ".join(str(port) for port, _ in candidates)
    raise LauncherError(f"这些本机端口均被占用：{ports}。可使用 --port 选择其他起始端口。")


def runtime_probe(python):
    code = (
        "import importlib,json,sys; "
        "missing=[]; "
        "\nfor name in " + repr(DEPENDENCIES) + ":"
        "\n try: importlib.import_module(name)"
        "\n except (ImportError,OSError): missing.append(name)"
        "\nprint(json.dumps({'version':list(sys.version_info[:2]),'missing':missing,'is_venv':sys.prefix!=sys.base_prefix}))"
    )
    try:
        completed = subprocess.run(
            [str(python), "-c", code], capture_output=True, text=True, timeout=60, check=False
        )
        if completed.returncode:
            return None
        value = json.loads(completed.stdout.strip().splitlines()[-1])
        return value if isinstance(value, dict) else None
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None


def run_setup(command, cwd, log_path):
    announce("正在准备运行环境，详细安装输出写入 launcher.log……", log_path)
    with log_path.open("a", encoding="utf-8") as stream:
        result = subprocess.run(
            command, cwd=cwd, stdin=subprocess.DEVNULL, stdout=stream,
            stderr=subprocess.STDOUT, check=False,
        )
    if result.returncode:
        raise LauncherError(
            "运行环境准备未完成。请检查网络和 launcher.log；程序没有关闭 SSL 证书校验。\n"
            "联网后可重新双击启动，或在项目目录执行：\n"
            '  python3 -m venv .venv\n  .venv/bin/python -m pip install -e "."'
        )


def ensure_runtime(project, explicit, allow_install, force_install, log_path):
    local_python = project / ".venv" / "bin" / "python"
    if explicit:
        candidate = Path(shutil.which(explicit) or explicit).expanduser().absolute()
    elif local_python.is_file():
        candidate = local_python
    else:
        candidate = Path(sys.executable)
    state = runtime_probe(candidate)
    if state is not None and tuple(state.get("version", [])) >= MIN_PYTHON:
        if not state.get("missing") and not force_install:
            return candidate
    elif not allow_install:
        raise LauncherError("没有可用的 Python 3.11+ 运行环境；可用 --python 指定解释器。")
    if not allow_install:
        missing = ", ".join(state.get("missing", [])) if state else "Python 运行环境"
        raise LauncherError(f"运行依赖缺失：{missing}。去掉 --no-install 后可自动准备项目环境。")

    # Never install application dependencies into a system/global Python.
    if not local_python.is_file():
        base = candidate
        if state is None or tuple(state.get("version", [])) < MIN_PYTHON:
            raise LauncherError(
                "首次运行需要 Python 3.11 或更高版本。安装后重新双击，"
                "或设置 CONTACT_ANGLE_PYTHON 指向该解释器。"
            )
        announce("首次运行：创建项目内 .venv 环境。首次安装需要联网。", log_path)
        run_setup([str(base), "-m", "venv", str(project / ".venv")], project, log_path)
    local_state = runtime_probe(local_python)
    if not local_state or not local_state.get("is_venv"):
        raise LauncherError(
            "项目 .venv 不是有效的独立虚拟环境。请将该目录改名保留，"
            "再用 Python 3.11+ 执行 python3 -m venv .venv。"
        )
    run_setup(
        [str(local_python), "-m", "pip", "--disable-pip-version-check", "install", "--no-input", "-e", str(project)],
        project, log_path,
    )
    state = runtime_probe(local_python)
    if not state or tuple(state.get("version", [])) < MIN_PYTHON or state.get("missing"):
        raise LauncherError("安装后依赖检查仍未通过。请查看 launcher.log 并重新安装项目环境。")
    return local_python


def server_command(python, file_descriptor):
    return [
        str(python), "-m", "uvicorn", "contact_angle.api:app", "--host", HOST,
        "--fd", str(file_descriptor), "--no-access-log",
    ]


def open_browser(url, headless, log_path):
    announce(f"接触角工作台：{url}", log_path)
    if not headless:
        try:
            opened = webbrowser.open(url)
        except webbrowser.Error:
            opened = False
        if not opened:
            announce("浏览器未能自动打开，请将上面的本机网址复制到浏览器。", log_path)


def wait_ready(child, url, data_dir, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise LauncherError("后端启动失败或提前退出。请查看 launcher.log 中的错误。")
        if same_studio(probe_info(url), data_dir):
            return
        time.sleep(0.2)
    raise LauncherError("后端未在规定时间内就绪。请查看 launcher.log；可增大 --startup-timeout。")


def stop_child(child):
    if child is None or child.poll() is not None:
        return
    child.send_signal(signal.SIGINT)
    try:
        child.wait(timeout=8)
    except subprocess.TimeoutExpired:
        child.terminate()
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=3)


def data_directory(args):
    return Path(args.data_dir or os.environ.get("CONTACT_ANGLE_DATA_DIR", project_root() / "data")).expanduser().resolve()


def read_state(data_dir):
    try:
        record = json.loads((data_dir / "launcher-state.json").read_text(encoding="utf-8"))
        return record if isinstance(record, dict) else None
    except (OSError, ValueError):
        return None


def process_signature(pid):
    """The creation time and command protect against reusing a stored PID."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
        return None
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "lstart=,command="],
            capture_output=True, text=True, timeout=3, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def matching_instance(info, record, data_dir):
    return bool(
        same_studio(info, data_dir) and record
        and isinstance(record.get("launch_id"), str)
        and len(record["launch_id"]) == 32
        and info.get("launch_id") == record["launch_id"]
        and info.get("server_pid") == record.get("pid")
    )


def managed_status(data_dir):
    record = read_state(data_dir)
    url = state_url(data_dir / "launcher-state.json")
    info = probe_info(url) if url else None
    return record, url, info


def stop_studio(args):
    data_dir = data_directory(args)
    record, url, info = managed_status(data_dir)
    if not same_studio(info, data_dir):
        if info is not None:
            raise LauncherError("状态文件指向其他服务，未停止任何程序。")
        announce("该资料库的工作台未在运行。")
        return 0
    if not matching_instance(info, record, data_dir):
        raise LauncherError("运行实例与启动记录不匹配，未停止任何程序。旧版服务请关闭原启动窗口。")
    pid = record.get("pid")
    expected = record.get("server_signature")
    if not expected or process_signature(pid) != expected or pid == os.getpid():
        raise LauncherError("进程身份与启动记录不匹配，未停止任何程序。")
    # Signal only the live, authenticated server. Its supervisor observes the
    # clean exit and releases its own lock/state; no process-group guessing.
    again = probe_info(url)
    if not matching_instance(again, record, data_dir):
        raise LauncherError("工作台实例已改变，请重试。")
    os.kill(pid, signal.SIGINT)
    deadline = time.monotonic() + max(15, args.startup_timeout)
    while time.monotonic() < deadline:
        active = probe_info(url)
        if not matching_instance(active, record, data_dir):
            current = read_state(data_dir)
            if not current or current.get("launch_id") != record["launch_id"]:
                announce("工作台已停止。")
                return 0
        time.sleep(0.1)
    raise LauncherError("服务仍在完成关闭，请查看 launcher.log，稍后重试。")


def show_status(args):
    data_dir = data_directory(args)
    record, url, info = managed_status(data_dir)
    running = same_studio(info, data_dir)
    print(json.dumps({
        "running": running, "url": url if running else None,
        "data_dir": str(data_dir), "mode": record.get("mode", "legacy") if running and record else None,
        "managed": matching_instance(info, record, data_dir),
        "pid": info.get("server_pid") if running else None,
        "supervisor_pid": record.get("supervisor_pid") if running and record else None,
    }, ensure_ascii=False))
    return 0


def background_command(args, data_dir):
    command = [
        sys.executable, str(Path(__file__).resolve()), "--background-supervisor", "--headless",
        "--port", str(args.port), "--data-dir", str(data_dir),
        "--startup-timeout", str(args.startup_timeout), "--lock-timeout", str(args.lock_timeout),
    ]
    if args.python:
        command += ["--python", args.python]
    if args.no_install:
        command.append("--no-install")
    if args.install:
        command.append("--install")
    return command


def start_background(args):
    project, data_dir = project_root(), data_directory(args)
    data_dir.mkdir(parents=True, exist_ok=True)
    log_path = data_dir / "launcher.log"
    _record, url, info = managed_status(data_dir)
    if same_studio(info, data_dir):
        announce("已复用正在运行的工作台。", log_path)
        open_browser(url, args.headless, log_path)
        return 0
    env = os.environ.copy()
    env["CONTACT_ANGLE_LOG_REDIRECTED"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    announce("正在后台启动工作台……", log_path)
    with log_path.open("a", encoding="utf-8") as log:
        supervisor = subprocess.Popen(
            background_command(args, data_dir), cwd=project, env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    deadline = time.monotonic() + args.background_timeout
    try:
        while time.monotonic() < deadline:
            _record, url, info = managed_status(data_dir)
            if same_studio(info, data_dir):
                open_browser(url, args.headless, log_path)
                announce("工作台已在后台运行，可以关闭启动窗口并继续使用终端。", log_path)
                return 0
            if supervisor.poll() is not None:
                raise LauncherError("后台启动未完成。请查看资料库目录中的 launcher.log。")
            time.sleep(0.15)
        raise LauncherError("后台准备超时。请查看 launcher.log；可增大 --background-timeout 后重试。")
    except BaseException:
        if supervisor.poll() is None:
            os.killpg(supervisor.pid, signal.SIGTERM)
            try:
                supervisor.wait(timeout=15)
            except subprocess.TimeoutExpired:
                # During first-run setup the installer shares this owned session.
                # Cancel that session, rather than leaving an unattended setup.
                os.killpg(supervisor.pid, signal.SIGTERM)
                supervisor.wait(timeout=5)
        raise


def run(args):
    project = project_root()
    data_dir = data_directory(args)
    if args.check:
        runtime = ensure_runtime(project, args.python, False, False, None)
        print(json.dumps({"project": str(project), "python": str(runtime), "data_dir": str(data_dir), "status": "ok"}, ensure_ascii=False))
        return 0
    if getattr(args, "stop", False):
        return stop_studio(args)
    if getattr(args, "status", False):
        return show_status(args)
    if getattr(args, "background", False):
        return start_background(args)
    data_dir.mkdir(parents=True, exist_ok=True)
    log_path = data_dir / "launcher.log"
    announce(f"图片、草稿、结果与日志目录：{data_dir}", log_path)
    lock, existing = acquire_lock(data_dir, args.lock_timeout, log_path)
    if existing:
        announce("已复用正在运行的工作台。", log_path)
        open_browser(existing, args.headless, log_path)
        return 0
    child = None
    sock = None
    launch_id = uuid.uuid4().hex
    state_path = data_dir / "launcher-state.json"
    try:
        previous = state_url(state_path)
        url, sock = reserve_port(args.port, data_dir, previous)
        if sock is None:
            announce("已复用同一资料库的现有工作台。", log_path)
            open_browser(url, args.headless, log_path)
            return 0
        runtime = ensure_runtime(project, args.python, not args.no_install, args.install, log_path)
        env = os.environ.copy()
        env["CONTACT_ANGLE_DATA_DIR"] = str(data_dir)
        env["PYTHONUNBUFFERED"] = "1"
        env["CONTACT_ANGLE_LAUNCH_ID"] = launch_id
        env["PYTHONPATH"] = str(project / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        announce("启动本机服务……", log_path)
        with log_path.open("a", encoding="utf-8") as log:
            child = subprocess.Popen(
                server_command(runtime, sock.fileno()), cwd=project, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=(sock.fileno(),),
            )
        wait_ready(child, url, data_dir, args.startup_timeout)
        write_state(state_path, {
            "launch_id": launch_id, "pid": child.pid, "url": url, "data_dir": str(data_dir),
            "supervisor_pid": os.getpid(),
            "mode": "background" if getattr(args, "background_supervisor", False) else "foreground",
            "server_signature": process_signature(child.pid),
        })
        # Uvicorn now holds a duplicate descriptor; only the owned child listens.
        sock.close()
        sock = None
        open_browser(url, args.headless, log_path)
        if getattr(args, "background_supervisor", False):
            announce("后台服务已就绪；使用停止接触角.command 或 --stop 可停止。", log_path)
        else:
            announce("前台运行：按 Ctrl+C 或关闭此窗口可停止服务。", log_path)
        while child.poll() is None:
            time.sleep(0.25)
        if child.returncode:
            raise LauncherError("后端已退出，请查看 launcher.log。重新双击可再次启动。")
        return 0
    except KeyboardInterrupt:
        announce("正在停止本次启动的工作台……", log_path)
        return 0
    finally:
        stop_child(child)
        if sock is not None:
            sock.close()
        try:
            record = json.loads(state_path.read_text(encoding="utf-8"))
            if record.get("launch_id") == launch_id:
                state_path.unlink(missing_ok=True)
        except (OSError, ValueError, AttributeError):
            pass
        if lock is not None:
            lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="接触角本机工作台启动器")
    parser.add_argument("--port", type=int, default=8765, help="首选端口，冲突时向后寻找可用端口")
    parser.add_argument("--data-dir", help="个人图片、草稿、测量结果与日志的保存目录")
    parser.add_argument("--python", default=os.environ.get("CONTACT_ANGLE_PYTHON"), help="已有依赖的 Python 解释器，或用于创建 .venv 的 Python 3.11+")
    parser.add_argument("--headless", action="store_true", help="启动后不自动打开浏览器")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--background", action="store_true", help="后台启动，服务就绪后立即返回终端")
    mode.add_argument("--foreground", action="store_true", help="前台运行，Ctrl+C 停止服务")
    mode.add_argument("--stop", action="store_true", help="安全停止同一资料库的受管工作台")
    mode.add_argument("--status", action="store_true", help="查看同一资料库的运行状态")
    parser.add_argument("--background-supervisor", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--background-timeout", type=float, default=600, help="等待后台准备环境与服务就绪的秒数")
    installation = parser.add_mutually_exclusive_group()
    installation.add_argument("--no-install", action="store_true", help="依赖缺失时报错，不自动安装")
    installation.add_argument("--install", action="store_true", help="在项目 .venv 内重新安装当前项目及依赖")
    parser.add_argument("--check", action="store_true", help="只检查运行依赖，不启动服务或安装")
    parser.add_argument("--startup-timeout", type=float, default=45, help="等待后端就绪的秒数")
    parser.add_argument("--lock-timeout", type=float, default=120, help="等待另一个启动窗口就绪的秒数")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65_535:
        parser.error("端口必须在1—65535之间。")
    if args.startup_timeout <= 0 or args.lock_timeout <= 0 or args.background_timeout <= 0:
        parser.error("等待时间必须大于0秒。")

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    for signum in (signal.SIGTERM, getattr(signal, "SIGHUP", signal.SIGTERM)):
        signal.signal(signum, interrupted)
    try:
        return run(args)
    except (LauncherError, OSError) as exc:
        print(f"\n未能启动：{exc}", file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
