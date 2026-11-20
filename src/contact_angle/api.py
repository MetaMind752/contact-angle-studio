"""Local HTTP interface. Start with ``contact-angle serve`` and open /docs."""

import base64
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from . import __version__
from .core import MAX_UPLOAD_BYTES, measure_bytes
from .models import MeasurementError, MeasurementOptions, MeasurementResult
from .studio_api import router as studio_router


class MeasureResponse(MeasurementResult):
    annotation_png_base64: str | None = None


app = FastAPI(
    title="Contact Angle Measurement",
    version=__version__,
    description=(
        "The studio /api/jobs endpoint measures sessile or inverted drops in original "
        "image coordinates (x right, y down). Results describe the angle inside the liquid. "
        "The legacy /measure endpoint takes a JSON options field with an explicit straight "
        "baseline and a droplet above it."
    ),
)


STATIC = Path(__file__).with_name("static")
app.include_router(studio_router)
app.mount("/static", StaticFiles(directory=STATIC, check_dir=False), name="studio-static")
app.mount("/assets", StaticFiles(directory=STATIC, check_dir=False), name="studio-assets")


@app.middleware("http")
async def local_studio_guard(request, call_next):
    if request.url.path.startswith("/api/"):
        host = request.headers.get("host", "").split(":")[0].lower()
        if host not in ("127.0.0.1", "localhost", "testserver"):
            return JSONResponse(
                {"detail": {"code": "local_only", "message": "请通过本机地址打开测量工作台。"}},
                status_code=403,
            )
        origin = request.headers.get("origin")
        if origin and origin != str(request.base_url).rstrip("/"):
            return JSONResponse(
                {
                    "detail": {
                        "code": "origin_denied",
                        "message": "请在测量工作台页面中执行此操作。",
                    }
                },
                status_code=403,
            )
        if request.method in ("POST", "PUT"):
            size = request.headers.get("content-length", "0")
            if not size.isdigit() or int(size) > MAX_UPLOAD_BYTES + 1024 * 1024:
                return JSONResponse(
                    {
                        "detail": {
                            "code": "request_too_large",
                            "message": "上传内容过大，请分张上传。",
                        }
                    },
                    status_code=413,
                )
    return await call_next(request)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(
        STATIC / "index.html", media_type="text/html", headers={"Cache-Control": "no-cache"}
    )


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.post("/measure", response_model=MeasureResponse, response_model_exclude_unset=True)
async def measure(
    image: Annotated[UploadFile, File(description="PNG, JPEG or other supported image file")],
    options: Annotated[
        str,
        Form(
            description=(
                'JSON, e.g. {"baseline":[{"x":20,"y":400},{"x":620,"y":400}],'
                '"method":"local_polynomial"}'
            )
        ),
    ],
    include_annotation: Annotated[bool, Form()] = False,
):
    try:
        if len(options) > 65_536:
            raise HTTPException(
                422, detail={"code": "invalid_options", "message": "Options too long."}
            )
        try:
            parameters = MeasurementOptions.model_validate_json(options)
        except ValidationError as exc:
            raise HTTPException(
                422,
                detail={
                    "code": "invalid_options",
                    "message": "Measurement options are invalid.",
                    "errors": exc.errors(
                        include_input=False, include_context=False, include_url=False
                    ),
                },
            ) from exc
        # Read at most one byte beyond the cap; do not load arbitrarily large uploads in RAM.
        data = await image.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                413,
                detail={"code": "image_too_large", "message": "Image exceeds the 20 MiB limit."},
            )
        try:
            outcome = await run_in_threadpool(measure_bytes, data, parameters)
        except MeasurementError as exc:
            raise HTTPException(
                413 if exc.code == "image_too_large" else 422,
                detail={"code": exc.code, "message": exc.message},
            ) from exc
        result = outcome.result.model_dump(mode="json")
        if include_annotation:
            result["annotation_png_base64"] = base64.b64encode(outcome.annotation_png).decode(
                "ascii"
            )
        return result
    finally:
        await image.close()
