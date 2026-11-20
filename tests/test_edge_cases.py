"""Regression checks for decoded image limits and fallback provenance."""

import cv2
import numpy as np
import pytest

from contact_angle.core import _subpixel_contour, decode_image, measure_bytes
from contact_angle.models import MeasurementError, MeasurementOptions


def test_long_thin_image_rejected_before_opencv_remap():
    image = np.full((120, 40000), 240, dtype=np.uint8)
    _, encoded = cv2.imencode(".png", image)
    with pytest.raises(MeasurementError, match="32767"):
        decode_image(encoded.tobytes())


def test_long_contour_probes_are_chunked():
    gray = np.full((50, 100), 240, dtype=np.uint8)
    gray[:, :50] = 20
    xy = np.tile([49.0, 25.0], (40000, 1))
    refined = _subpixel_contour(xy, gray, np.array([0, 0]), 130, "dark")
    assert refined.shape == xy.shape
    assert np.all(np.isfinite(refined))
    assert np.max(np.abs(refined[:, 0] - 49.5)) < 0.1


def test_requested_small_window_failure_records_fallback():
    image = np.full((160, 240), 240, dtype=np.uint8)
    cv2.circle(image, (120, 100), 35, 20, -1)
    image[100:, :] = 240
    _, encoded = cv2.imencode(".png", image)
    options = MeasurementOptions(
        baseline=[{"x": 20, "y": 100}, {"x": 220, "y": 100}],
        fit_fraction=0.04,
        excluded_regions=[{"x": 80, "y": 91, "width": 80, "height": 3}],
    )
    result = measure_bytes(encoded.tobytes(), options).result
    assert result.requested_parameters is not None
    assert result.requested_parameters.fit_fraction == 0.04
    assert result.parameters.fit_fraction != 0.04
    assert result.parameters.baseline == options.baseline
    assert result.status == "review"
    assert any(
        run.name == "requested_fit_failed" and run.status == "failed"
        for run in result.sensitivity.runs
    )
