"""Command-line measurement, reproducible file exports, and local server launcher."""

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

from pydantic import ValidationError

from . import __version__
from .core import MAX_UPLOAD_BYTES, measure_bytes
from .models import MeasurementError, MeasurementOptions

OUTPUT_NAMES = ("result.json", "annotated.png", "results.csv", "sensitivity.csv")


def _fail(code: str, message: str) -> int:
    print(
        json.dumps({"error": {"code": code, "message": message}}, ensure_ascii=False),
        file=sys.stderr,
    )
    return 1


def _measure(args: argparse.Namespace) -> int:
    output = args.output
    if not args.overwrite and any((output / name).exists() for name in OUTPUT_NAMES):
        return _fail(
            "output_exists",
            "Output files already exist. Choose a new directory or use --overwrite.",
        )
    try:
        options = MeasurementOptions.model_validate_json(args.options.read_text(encoding="utf-8"))
        with args.image.open("rb") as source:
            data = source.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise MeasurementError("image_too_large", "Image exceeds the 20 MiB limit.")
        outcome = measure_bytes(data, options)
        output.mkdir(parents=True, exist_ok=True)
        result = outcome.result
        sensitivity = result.sensitivity
        # Stage all exports before publishing them to avoid truncated JSON/PNG files.
        with tempfile.TemporaryDirectory(prefix=".contact-angle-", dir=output) as staging_dir:
            staging = Path(staging_dir)
            (staging / "result.json").write_text(
                result.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            (staging / "annotated.png").write_bytes(outcome.annotation_png)
            with (staging / "results.csv").open("w", newline="", encoding="utf-8-sig") as target:
                fields = [
                    "image_sha256",
                    "algorithm_version",
                    "method",
                    "left_angle_deg",
                    "right_angle_deg",
                    "mean_angle_deg",
                    "asymmetry_deg",
                    "left_fit_rms_px",
                    "right_fit_rms_px",
                    "status",
                    "warnings",
                    "sensitivity_enabled",
                    "sensitivity_sensitive",
                    "sensitivity_accepted_runs",
                    "sensitivity_excluded_runs",
                    "sensitivity_failed_runs",
                    "left_sensitivity_min_deg",
                    "left_sensitivity_max_deg",
                    "left_sensitivity_span_deg",
                    "left_sensitivity_max_abs_delta_deg",
                    "right_sensitivity_min_deg",
                    "right_sensitivity_max_deg",
                    "right_sensitivity_span_deg",
                    "right_sensitivity_max_abs_delta_deg",
                ]
                writer = csv.DictWriter(target, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "image_sha256": result.image_sha256,
                        "algorithm_version": result.algorithm_version,
                        "method": result.method,
                        "left_angle_deg": result.left.angle_deg,
                        "right_angle_deg": result.right.angle_deg,
                        "mean_angle_deg": result.mean_angle_deg,
                        "asymmetry_deg": result.asymmetry_deg,
                        "left_fit_rms_px": result.left.fit_rms_px,
                        "right_fit_rms_px": result.right.fit_rms_px,
                        "status": result.status,
                        "warnings": "; ".join(result.warnings),
                        "sensitivity_enabled": sensitivity is not None,
                        "sensitivity_sensitive": sensitivity.sensitive if sensitivity else "",
                        "sensitivity_accepted_runs": sensitivity.accepted_runs
                        if sensitivity
                        else "",
                        "sensitivity_excluded_runs": sensitivity.excluded_runs
                        if sensitivity
                        else "",
                        "sensitivity_failed_runs": sensitivity.failed_runs if sensitivity else "",
                        **(
                            {
                                f"{side}_sensitivity_{field}": getattr(
                                    getattr(sensitivity, side), field
                                )
                                for side in ("left", "right")
                                if getattr(sensitivity, side) is not None
                                for field in ("min_deg", "max_deg", "span_deg", "max_abs_delta_deg")
                            }
                            if sensitivity
                            else {}
                        ),
                    }
                )
            with (staging / "sensitivity.csv").open(
                "w", newline="", encoding="utf-8-sig"
            ) as target:
                fields = [
                    "name",
                    "method",
                    "status",
                    "left_angle_deg",
                    "right_angle_deg",
                    "left_delta_deg",
                    "right_delta_deg",
                    "error",
                    "parameters_json",
                ]
                writer = csv.DictWriter(target, fieldnames=fields)
                writer.writeheader()
                for run in sensitivity.runs if sensitivity else []:
                    row = run.model_dump(exclude={"parameters"})
                    row["parameters_json"] = json.dumps(
                        run.parameters, ensure_ascii=False, sort_keys=True
                    )
                    writer.writerow(row)
            if args.overwrite:
                for name in OUTPUT_NAMES:
                    os.replace(staging / name, output / name)
            else:
                created = []
                try:
                    for name in OUTPUT_NAMES:
                        os.link(staging / name, output / name)
                        created.append(output / name)
                except OSError:
                    for path in created:
                        path.unlink(missing_ok=True)
                    raise
        print(
            json.dumps(
                {
                    "output": str(output.resolve()),
                    "left_angle_deg": result.left.angle_deg,
                    "right_angle_deg": result.right.angle_deg,
                    "status": result.status,
                    "sensitivity": sensitivity.model_dump(exclude={"runs"})
                    if sensitivity
                    else None,
                },
                ensure_ascii=False,
            )
        )
        return 0
    except MeasurementError as exc:
        return _fail(exc.code, exc.message)
    except ValidationError as exc:
        errors = exc.errors(include_input=False, include_context=False, include_url=False)
        return _fail("invalid_options", json.dumps(errors, ensure_ascii=False))
    except (OSError, UnicodeError) as exc:
        return _fail("file_error", str(exc))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local sessile-drop contact angle measurement")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    measurement = commands.add_parser(
        "measure", help="Measure an image; export JSON, PNG and sensitivity CSV"
    )
    measurement.add_argument("image", type=Path)
    measurement.add_argument(
        "--options", required=True, type=Path, help="JSON file with baseline and fitting options"
    )
    measurement.add_argument("--output", required=True, type=Path, help="Destination directory")
    measurement.add_argument(
        "--overwrite", action="store_true", help="Replace existing export files"
    )

    server = commands.add_parser("serve", help="Start the local API; documentation is at /docs")
    server.add_argument("--host", default="127.0.0.1", help="Bind address (default: loopback only)")
    server.add_argument("--port", default=8765, type=int)

    args = parser.parse_args(argv)
    if args.command == "measure":
        return _measure(args)
    import uvicorn

    if not 1 <= args.port <= 65535:
        return _fail("invalid_port", "Port must be between 1 and 65535.")
    try:
        uvicorn.run("contact_angle.api:app", host=args.host, port=args.port)
    except OSError as exc:
        return _fail("server_error", str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
