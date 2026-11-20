"""Atomic file-backed personal workspace, with one bounded measurement worker."""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import cv2

from .core import decode_image
from .models import MeasurementError

IDENTIFIER = re.compile(r"^[a-f0-9]{32}$")


def timestamp():
    return datetime.now(UTC).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class StudioStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        (self.root / "images").mkdir(parents=True, exist_ok=True)
        (self.root / "results").mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="contact-angle")
        atexit.register(self.worker.shutdown, wait=False, cancel_futures=True)
        for path in (self.root / "results").glob("*/job.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
                if job["state"] in ("queued", "running"):
                    job.update(
                        state="error",
                        error={
                            "code": "interrupted",
                            "message": "上次测量随服务关闭中断，请重新运行。",
                        },
                    )
                    write_json(path, job)
            except (OSError, ValueError, KeyError):
                continue

    def directory(self, kind, identifier):
        if kind not in ("images", "results") or not IDENTIFIER.fullmatch(identifier):
            raise KeyError(identifier)
        path = self.root / kind / identifier
        if not path.is_dir():
            raise KeyError(identifier)
        return path

    def image(self, identifier):
        try:
            return json.loads(
                (self.directory("images", identifier) / "info.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise KeyError(identifier) from exc

    def images(self):
        rows = []
        for path in (self.root / "images").glob("*/info.json"):
            try:
                rows.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return sorted(rows, key=lambda r: r["created_at"], reverse=True)

    def add_image(self, data, filename):
        # Validate before creating any file. Keep original bytes for reproducibility.
        image = decode_image(data)
        digest = hashlib.sha256(data).hexdigest()
        with self.lock:
            for row in self.images():
                if row["sha256"] == digest:
                    return row
            identifier = uuid.uuid4().hex
            directory = self.root / "images" / identifier
            directory.mkdir()
            success, png = cv2.imencode(".png", image)
            if not success:
                raise MeasurementError("encode_failed", "无法创建图片预览。")
            (directory / "original").write_bytes(data)
            (directory / "preview.png").write_bytes(png.tobytes())
            row = {
                "id": identifier,
                "filename": Path(filename or "液滴图片").name[:240],
                "width": image.shape[1],
                "height": image.shape[0],
                "sha256": digest,
                "created_at": timestamp(),
                "preview_url": f"/api/images/{identifier}/preview.png",
            }
            write_json(directory / "info.json", row)
            return row

    def draft(self, identifier):
        path = self.directory("images", identifier) / "draft.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def save_draft(self, identifier, options):
        with self.lock:
            write_json(self.directory("images", identifier) / "draft.json", options)

    def job(self, identifier):
        try:
            return json.loads(
                (self.directory("results", identifier) / "job.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise KeyError(identifier) from exc

    def jobs(self, image_id):
        self.image(image_id)
        rows = []
        for path in (self.root / "results").glob("*/job.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
                if job["image_id"] == image_id:
                    rows.append(job)
            except (OSError, ValueError, KeyError):
                continue
        return sorted(rows, key=lambda r: r["created_at"], reverse=True)

    def queue(self, image_id, options):
        self.image(image_id)
        with self.lock:
            pending = sum(
                1 for path in (self.root / "results").glob("*/job.json") if self._pending(path)
            )
            if pending >= 16:
                raise MeasurementError("queue_full", "测量队列已满，请等待当前任务完成。")
            identifier = uuid.uuid4().hex
            directory = self.root / "results" / identifier
            directory.mkdir()
            job = {
                "id": identifier,
                "image_id": image_id,
                "options": options,
                "state": "queued",
                "created_at": timestamp(),
                "result": None,
                "error": None,
            }
            write_json(directory / "job.json", job)
            self.worker.submit(self._run, identifier)
            return job

    @staticmethod
    def _pending(path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))["state"] in ("queued", "running")
        except (OSError, ValueError, KeyError):
            return False

    def _run(self, identifier):
        import logging

        from .studio_service import execute_job

        job = self.job(identifier)
        path = self.directory("results", identifier) / "job.json"
        job["state"] = "running"
        write_json(path, job)
        try:
            result = execute_job(self, job)
            job.update(state="done", result=result)
        except MeasurementError as exc:
            messages = {
                "unresolved_profile": "未找到能可靠拟合的完整轮廓。请核对基线和分析区域，或改为手动打点。",
                "surface_unresolved": "无法从两侧表面确定基线，请手动标记基线两个端点。",
                "baseline_unresolved": "表面边缘不足，请手动标记基线两个端点。",
                "no_drop": "分析区域中未找到可分离的液滴，请核对基线与液滴范围。",
                "no_admissible_models": "没有足够可靠的候选模型，请改用手动边缘点核查。",
                "clipped_drop": "液滴轮廓触及分析区域边界，请扩大区域保留完整轮廓。",
            }
            job.update(
                state="error",
                error={
                    "code": exc.code,
                    "message": messages.get(exc.code, exc.message),
                    "diagnostic": exc.message,
                },
            )
        except Exception:  # Preserve failures and keep the personal worker alive.
            logging.getLogger(__name__).exception("Measurement job %s failed", identifier)
            job.update(
                state="error",
                error={
                    "code": "measurement_failed",
                    "message": "本次拟合未完成。请检查基线及接触区，或改用手动打点。详细原因已记录在本机日志。",
                },
            )
        job["finished_at"] = timestamp()
        write_json(path, job)


_stores = {}
_stores_lock = threading.Lock()


def get_store():
    root = Path(os.environ.get("CONTACT_ANGLE_DATA_DIR", str(Path.cwd() / "data"))).resolve()
    with _stores_lock:
        if root not in _stores:
            _stores[root] = StudioStore(root)
        return _stores[root]
