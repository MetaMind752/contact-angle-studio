"""Personal web studio endpoints. Uploads stay in the local workspace."""

import os
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from . import __version__
from .core import MAX_IMAGE_PIXELS, MAX_UPLOAD_BYTES, decode_image
from .manual_arc import contour_observations, point, preview_contour
from .models import MeasurementError
from .studio_models import ContourPreviewRequest, JobRequest, Orientation, StudioOptions
from .studio_service import estimate_surface_points, manual_orientation
from .studio_store import get_store

router = APIRouter(prefix="/api", tags=["Personal studio"])


def failure(exc):
    if isinstance(exc, KeyError):
        return HTTPException(
            404, detail={"code": "not_found", "message": "找不到这张图片或测量记录。"}
        )
    return HTTPException(
        413 if exc.code == "image_too_large" else 422,
        detail={"code": exc.code, "message": exc.message},
    )


@router.get("/info")
def info():
    return {
        "name": "Contact Angle Studio",
        "version": __version__,
        "launch_id": os.environ.get("CONTACT_ANGLE_LAUNCH_ID"),
        "server_pid": os.getpid(),
        "supervisor_pid": os.getppid(),
        "data_dir": str(get_store().root),
        "limits": {"max_upload_bytes": MAX_UPLOAD_BYTES, "max_pixels": MAX_IMAGE_PIXELS},
    }


@router.get("/images")
def images():
    return {"images": get_store().images()}


@router.post("/images", status_code=201)
async def upload(image: Annotated[UploadFile, File()]):
    try:
        data = await image.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise MeasurementError("image_too_large", "图片超过20MiB，请先裁剪或导出单帧。")
        return await run_in_threadpool(get_store().add_image, data, image.filename)
    except MeasurementError as exc:
        raise failure(exc) from exc
    finally:
        await image.close()


@router.get("/images/{image_id}/preview.png")
def preview(image_id: str):
    try:
        path = get_store().directory("images", image_id) / "preview.png"
        return FileResponse(path, media_type="image/png")
    except KeyError as exc:
        raise failure(exc) from exc


@router.get("/images/{image_id}/baseline")
def baseline(
    image_id: str,
    polarity: str = "dark",
    algorithm: str = "general",
    orientation: Orientation = "auto",
):
    try:
        data = (get_store().directory("images", image_id) / "original").read_bytes()
        detected = estimate_surface_points(decode_image(data), polarity, algorithm, orientation)
        return {
            **detected,
            "source": "automatic",
            "warnings": [],
        }
    except (KeyError, MeasurementError) as exc:
        raise failure(exc) from exc


@router.post("/images/{image_id}/baseline")
def preview_surface(image_id: str, options: StudioOptions):
    try:
        data = (get_store().directory("images", image_id) / "original").read_bytes()
        marks = [
            point.model_dump() for point in (options.baseline_points or options.baseline or [])
        ]
        return estimate_surface_points(
            decode_image(data),
            options.polarity,
            options.algorithm,
            manual_orientation(options.model_dump(mode="json")),
            marks if options.baseline_source == "manual" else None,
        )
    except (KeyError, MeasurementError) as exc:
        raise failure(exc) from exc


@router.post("/images/{image_id}/contour-preview")
def preview_manual_contour(image_id: str, request: ContourPreviewRequest):
    try:
        image = get_store().image(image_id)
        width, height = image["width"], image["height"]
        options = request.model_dump(mode="json")
        xy = contour_observations(options["contour_points"], width, height)
        fitted = preview_contour(xy, width, height, options["baseline"], request.orientation)
        return {"points": [point(p) for p in fitted]}
    except (KeyError, MeasurementError) as exc:
        raise failure(exc) from exc


@router.get("/images/{image_id}/draft")
def draft(image_id: str):
    try:
        return {"options": get_store().draft(image_id)}
    except KeyError as exc:
        raise failure(exc) from exc


@router.put("/images/{image_id}/draft")
def save_draft(image_id: str, options: StudioOptions):
    try:
        get_store().save_draft(image_id, options.model_dump(mode="json"))
        return {"saved": True}
    except KeyError as exc:
        raise failure(exc) from exc


@router.post("/jobs", status_code=202)
def start_job(request: JobRequest):
    try:
        return get_store().queue(request.image_id, request.options.model_dump(mode="json"))
    except (KeyError, MeasurementError) as exc:
        raise failure(exc) from exc


@router.get("/jobs/{job_id}")
def job(job_id: str):
    try:
        return get_store().job(job_id)
    except KeyError as exc:
        raise failure(exc) from exc


@router.get("/images/{image_id}/results")
def history(image_id: str):
    try:
        return {"results": get_store().jobs(image_id)}
    except KeyError as exc:
        raise failure(exc) from exc


@router.get("/results/{job_id}/{filename}")
def download(job_id: str, filename: str):
    types = {
        "result.json": "application/json",
        "results.csv": "text/csv; charset=utf-8",
        "annotation.png": "image/png",
        "bundle.zip": "application/zip",
    }
    if filename not in types:
        raise HTTPException(404, detail={"code": "not_found", "message": "找不到导出文件。"})
    try:
        store = get_store()
        item = store.job(job_id)
        if item["state"] != "done":
            raise HTTPException(
                409, detail={"code": "not_ready", "message": "测量尚未完成，暂时没有导出文件。"}
            )
        stem = Path(store.image(item["image_id"])["filename"]).stem
        suffix = Path(filename).suffix
        return FileResponse(
            store.directory("results", job_id) / filename,
            media_type=types[filename],
            filename=f"{stem}-{job_id[:6]}{suffix}",
        )
    except KeyError as exc:
        raise failure(exc) from exc


@router.get("/demo")
def demo():
    rows = []
    for angle in (90, 120):
        image = np.full((480, 700, 3), 246, dtype=np.uint8)
        radius = 145
        baseline_y = 360
        cy = baseline_y + radius * np.cos(np.deg2rad(angle))
        cv2.circle(image, (350, round(cy)), radius, (22, 22, 22), -1, cv2.LINE_AA)
        image[baseline_y:, :] = 225
        encoded = cv2.imencode(".png", image)[1].tobytes()
        row = get_store().add_image(encoded, f"合成演示-{angle}度.png")
        if get_store().draft(row["id"]) is None:
            observed = estimate_surface_points(image)
            opt = StudioOptions(
                baseline=observed["baseline"],
                baseline_points=observed["baseline_points"],
                baseline_source="automatic",
            )
            get_store().save_draft(row["id"], opt.model_dump(mode="json"))
        rows.append(row)
    return {"images": rows}
