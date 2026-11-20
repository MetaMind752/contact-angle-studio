"""Default adapters return measurements only; reference labels are never passed."""

from __future__ import annotations


def current(data: bytes, baseline=None):
    if baseline is not None:
        from contact_angle.core import measure_bytes
        from contact_angle.models import MeasurementOptions

        result = measure_bytes(data, MeasurementOptions(baseline=baseline)).result
        if result.sensitivity and result.sensitivity.accepted_runs == 0:
            raise ValueError(
                "No admissible measurement method; diagnostic angles cannot be scored."
            )
        return result.model_dump(mode="json")
    from contact_angle.fibrous import measure_fibrous_bytes

    result, _ = measure_fibrous_bytes(data)
    return result


def fibrous(data: bytes, baseline=None):
    from contact_angle.fibrous import measure_fibrous_bytes

    result, _ = measure_fibrous_bytes(data, baseline)
    return result


def original(data: bytes, baseline=None):
    if baseline is None:
        raise ValueError("The original core needs a supplied baseline.")
    from contact_angle.core import measure_bytes
    from contact_angle.models import MeasurementOptions

    result = measure_bytes(data, MeasurementOptions(baseline=baseline)).result
    if result.sensitivity and result.sensitivity.accepted_runs == 0:
        raise ValueError("Original core had no admissible methods.")
    return result.model_dump(mode="json")


def general(data: bytes, baseline=None):
    from contact_angle.general import measure_general_bytes

    return measure_general_bytes(data, baseline)
