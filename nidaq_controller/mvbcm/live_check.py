"""Score typed inputs against GET /api/live. Does not open an NI task.

Blank analog inputs are 0, then the same conversion as ``sweep.py`` /
``emc.py``. Digital blank is OFF. Rows use the MV BCM CSV columns.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

try:
    from nidaq_controller.mvbcm.live import (
        LiveError,
        digital_active,
        fetch_live,
        in_limits,
        measure,
    )
except ModuleNotFoundError:
    from mvbcm.live import (
        LiveError,
        digital_active,
        fetch_live,
        in_limits,
        measure,
    )

CSV_COLUMNS = (
    "group",
    "test_id",
    "step",
    "signal",
    "phase",
    "stimulus_value",
    "unit",
    "measured",
    "field",
    "min",
    "max",
    "result",
    "timestamp",
    "note",
)
DEFAULT_RESULTS = Path(__file__).resolve().parents[1] / "logs" / "mvbcm_results.csv"

# sweep.py: 4 mA = 0 A, 20 mA = full scale. emc.py: vctrl is already volts.
_COIL_FS_A = 50.0
_COIL_TOL_A = 0.25
_IMTR_FS_A = 100.0
_IMTR_TOL_A = 1.0
_IPH_FS_MV = 333.0
_IPH_FS_A = 20.0
_IPH_TOL_A = 0.2
_VCTRL_TOL_V = 5.0
_MA_ZERO = 4.0
_MA_SPAN = 16.0

# (signal, input unit written to the CSV, conversion kind)
ANALOG = (
    ("tc1", "mA", "coil"),
    ("tc2", "mA", "coil"),
    ("cc", "mA", "coil"),
    ("imtr", "mA", "imtr"),
    ("iph", "mV RMS", "iph"),
    ("vctrl", "V", "vctrl"),
)
DIGITAL = ("trip", "backtrip", "close", "52a", "52b")
_DIGITAL_ALIASES = {
    "trip": ("trip",),
    "backtrip": ("backtrip", "backuptrip", "backup", "backup_trip"),
    "close": ("close",),
    "52a": ("52a", "switch52a", "a52"),
    "52b": ("52b", "switch52b", "b52"),
}
_ON = {"1", "on", "true", "yes"}
_OFF = {"", "0", "off", "false", "no"}


def parse_inputs(analog, digital) -> tuple[dict[str, float], dict[str, bool]]:
    """Blank analog fields are 0. Blank digital fields are OFF."""
    if analog is None:
        analog = {}
    if digital is None:
        digital = {}
    if not isinstance(analog, dict):
        raise ValueError("analog must be an object")
    if not isinstance(digital, dict):
        raise ValueError("digital must be an object")
    parsed_analog: dict[str, float] = {}
    for signal, _unit, _kind in ANALOG:
        parsed_analog[signal] = _blank_float(analog.get(signal), signal)
    parsed_digital: dict[str, bool] = {}
    for signal in DIGITAL:
        parsed_digital[signal] = _on_off(digital.get(signal), signal)
    return parsed_analog, parsed_digital


def convert_analog(kind: str, raw: float) -> tuple[float, float, float, str]:
    """Return expected, min, max, engineering unit."""
    if kind == "coil":
        expected = (raw - _MA_ZERO) / _MA_SPAN * _COIL_FS_A
        tol, unit = _COIL_TOL_A, "A"
    elif kind == "imtr":
        expected = (raw - _MA_ZERO) / _MA_SPAN * _IMTR_FS_A
        tol, unit = _IMTR_TOL_A, "A"
    elif kind == "iph":
        expected = raw / _IPH_FS_MV * _IPH_FS_A
        tol, unit = _IPH_TOL_A, "A"
    elif kind == "vctrl":
        expected = raw
        tol, unit = _VCTRL_TOL_V, "V"
    else:
        raise ValueError(f"unknown analog kind {kind!r}")
    return expected, expected - tol, expected + tol, unit


def score_once(payload: dict, analog: dict[str, float], digital: dict[str, bool], step: int) -> list[dict]:
    """One live sample, one CSV row per channel."""
    rows = []
    for signal, input_unit, kind in ANALOG:
        raw = float(analog.get(signal, 0.0))
        expected, low, high, eng = convert_analog(kind, raw)
        try:
            measured = measure(payload, signal, "rms_eng")
        except LiveError as exc:
            rows.append(_row(
                step, signal, raw, input_unit, None, "rms_eng",
                low, high, "FAIL", str(exc),
            ))
            continue
        ok = in_limits(measured, low, high)
        note = "" if ok else f"expected [{_fmt(low)}, {_fmt(high)}] {eng}, got {_fmt(measured)}"
        rows.append(_row(
            step, signal, raw, input_unit, measured, "rms_eng",
            low, high, "PASS" if ok else "FAIL", note,
        ))
    for signal in DIGITAL:
        want = bool(digital.get(signal, False))
        bit = "1" if want else "0"
        stimulus = "on" if want else "off"
        try:
            got = _digital_got(payload, signal)
        except LiveError as exc:
            rows.append(_row(
                step, signal, stimulus, "active", None, "digital.active",
                bit, bit, "FAIL", str(exc),
            ))
            continue
        ok = got == want
        measured = "1" if got else "0"
        note = "" if ok else f"expected {'ON' if want else 'OFF'}, got {'ON' if got else 'OFF'}"
        rows.append(_row(
            step, signal, stimulus, "active", measured, "digital.active",
            bit, bit, "PASS" if ok else "FAIL", note,
        ))
    return rows


def fetch_and_score(
    board: str,
    analog: dict[str, float],
    digital: dict[str, bool],
    step: int,
    timeout: float = 5.0,
) -> list[dict]:
    return score_once(fetch_live(board, timeout=timeout), analog, digital, step)


def live_error_row(step: int, note: str) -> dict:
    """One FAIL row when GET /api/live itself fails."""
    return _row(step, "", "", "", None, "", "", "", "FAIL", note)


def append_results(path: Path, rows: list[dict]) -> None:
    """Append scored rows to the same CSV the test script uses."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def _blank_float(value, signal: str) -> float:
    if value is None or (isinstance(value, str) and not value.strip()):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{signal} must be a number") from exc


def _on_off(value, signal: str) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in _OFF:
        return False
    if text in _ON:
        return True
    raise ValueError(f"{signal} must be on or off")


def _digital_got(payload: dict, signal: str) -> bool:
    last: LiveError | None = None
    for name in _DIGITAL_ALIASES[signal]:
        try:
            return digital_active(payload, name)
        except LiveError as exc:
            last = exc
    if last is not None:
        raise last
    raise LiveError(f"signal {signal!r} not in live digital")


def _row(step, signal, stimulus, unit, measured, field, low, high, result, note) -> dict:
    return {
        "group": "live_check",
        "test_id": "LIVE",
        "step": str(step),
        "signal": signal,
        "phase": "check",
        "stimulus_value": _fmt(stimulus),
        "unit": unit,
        "measured": "" if measured is None else _fmt(measured),
        "field": field,
        "min": _fmt(low),
        "max": _fmt(high),
        "result": result,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "note": note,
    }


def _fmt(value) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        return value
    return f"{float(value):.6g}"
