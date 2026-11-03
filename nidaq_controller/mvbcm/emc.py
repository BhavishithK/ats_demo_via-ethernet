"""EMC immunity hold: steady analogs, 120 V on P1.6, trip/close every minute.

TC1, TC2, CC, and Imtr stay at 12 mA. Iph is a 50 Hz sine at 160 mA RMS.
P1.6 stays high so the fixture puts 120 V DC on Vctrl. Neither line is
part of the trip map.

Every 7 s the live ``rms_eng`` values are scored. Trip is at 0 s and close
is at 10 s. After each pulse the script polls until the new
``operations.recent`` row and the matching counter arrive, or until 10 s
after that pulse. The close pulse does not move earlier if the trip row
is already in. A ``fault`` or ``motor_run`` row, or a move in
``fault_total`` / ``motor_run_count``, fails. This module is not called
from the main MV BCM run.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Any

from nidaq_controller.mvbcm.breaker_ops import (
    MvbcmStopped,
    _ensure_daq_path,
    _match_line,
    _raise_if_stopped,
    default_trip_channels,
    device_name_from_channels,
    find_digital_module,
    open_breaker_controller,
    resolve_channels,
)
from nidaq_controller.mvbcm.channel_map import GROUPS, MappedChannel
from nidaq_controller.mvbcm.live import (
    LiveError,
    b7_value,
    fetch_live,
    in_limits,
    max_recent_ts,
    measure,
    operations_recent,
)
from nidaq_controller.mvbcm.stimuli.current import CurrentStimulus, find_9266
from nidaq_controller.mvbcm.stimuli.voltage import VoltageStimulus, find_9264

HOLD_MA = 12.0
IPH_MA = 160.0
IPH_FS_A = 20.0
IPH_FS_MV = 333.0
FREQUENCY_HZ = 50.0
VCTRL_LINE = "port1/line6"

COIL_A = 25.0
COIL_TOL_A = 0.25
IMTR_A = 50.0
IMTR_TOL_A = 1.0
IPH_A = IPH_MA / 1000.0
IPH_TOL_A = 0.05
VCTRL_V = 120.0
VCTRL_TOL_V = 5.0

CYCLE_S = 60.0
RMS_STEP_S = 7.0
OP_GAP_S = 10.0
POLL_S = 0.5
# (pulse offset, score deadline, play kind, expected rec_slot, counter +1)
OPS = (
    (0.0, OP_GAP_S, "trip", "trip", "trip_total"),
    (OP_GAP_S, OP_GAP_S * 2, "close", "close", "close_total"),
)
RMS_SPECS = (
    ("tc1", COIL_A, COIL_TOL_A, "A"),
    ("tc2", COIL_A, COIL_TOL_A, "A"),
    ("cc", COIL_A, COIL_TOL_A, "A"),
    ("imtr", IMTR_A, IMTR_TOL_A, "A"),
    ("iph", IPH_A, IPH_TOL_A, "A"),
    ("vctrl", VCTRL_V, VCTRL_TOL_V, "V"),
)
COUNTERS = ("trip_total", "close_total")
QUIET = ("fault_total", "motor_run_count")
UNWANTED_SLOTS = ("fault", "motor_run")

_CLOSED = {
    "trip_coil": False,
    "backup_trip_coil": False,
    "switch_52a": True,
    "switch_52b": False,
    "close_coil": False,
}


def iph_millivolts_rms() -> float:
    """Board millivolts RMS for 160 mA on the 20 A / 333 mV iph scale."""
    return (IPH_MA / 1000.0) / IPH_FS_A * IPH_FS_MV


def _current_channels() -> list[MappedChannel]:
    for group in GROUPS:
        if group.id == "tc_current":
            return list(group.channels)
    raise RuntimeError("tc_current group is not configured")


def _iph_channel() -> MappedChannel:
    for group in GROUPS:
        if group.id == "iph_voltage" and group.channels:
            return group.channels[0]
    raise RuntimeError("iph_voltage group is not configured")


def _hold_levels(channels: list[MappedChannel]) -> dict[str, float]:
    return {channel.signal: HOLD_MA for channel in channels}


def run_emc(
    board: str,
    *,
    voltage_device: str | None = None,
    current_device: str | None = None,
    channels: dict[str, str] | None = None,
    module=None,
    controller_factory=None,
    fetch=None,
    on_row=None,
    should_stop=None,
    timeout: float = 5.0,
) -> list[dict]:
    """Hold the EMC stimuli and repeat the minute sequence until stop."""
    if fetch is None:
        def fetch(url, _timeout=timeout):
            return fetch_live(url, timeout=_timeout)

    vdev = voltage_device or find_9264()
    cdev = current_device or find_9266()
    resolved, controller, module = _open_digital(channels, module, controller_factory)
    vctrl = _resolve_vctrl(module, resolved)
    iph = _iph_channel()
    millivolts = iph_millivolts_rms()
    stop_event = threading.Event()

    voltage: VoltageStimulus | None = None
    current: CurrentStimulus | None = None
    current_channels = _current_channels()
    rows: list[dict] = []
    try:
        voltage = VoltageStimulus(vdev, [iph])
        current = CurrentStimulus(cdev, current_channels)
        current.write(_hold_levels(current_channels))
        voltage.write({iph.signal: millivolts}, frequency=FREQUENCY_HZ)
        _prime_closed(controller, resolved)
        _set_line(controller, vctrl, True)
        _raise_if_stopped(should_stop)
        cycle_no = 1
        while True:
            _raise_if_stopped(should_stop)
            cycle_rows = _run_cycle(
                fetch, board, controller, resolved, stop_event, should_stop,
                cycle_no, on_row,
            )
            rows.extend(cycle_rows)
            cycle_no += 1
    except MvbcmStopped:
        raise
    except Exception as exc:
        row = _result(0, "emc", "run", "", "", "", "", "", "FAIL", str(exc))
        rows.append(row)
        if on_row is not None:
            on_row(row)
        return rows
    finally:
        stop_event.set()
        try:
            _set_line(controller, vctrl, False)
        except Exception:
            pass
        if current is not None:
            try:
                current.write({channel.signal: 0.0 for channel in current_channels})
            except Exception:
                pass
            try:
                current.close()
            except Exception:
                pass
        if voltage is not None:
            try:
                voltage.write({iph.signal: 0.0}, frequency=FREQUENCY_HZ)
            except Exception:
                pass
            try:
                voltage.close()
            except Exception:
                pass


def _run_cycle(
    fetch, board, controller, resolved, stop_event, should_stop, cycle_no, on_row,
) -> list[dict]:
    """One minute. Trip at 0 s, close at 10 s. Poll each event for 10 s."""
    rows: list[dict] = []
    cycle_t0 = time.perf_counter()
    rms_at = 0.0

    def _emit(batch: list[dict]) -> None:
        rows.extend(batch)
        if on_row is not None:
            for row in batch:
                on_row(row)

    def _score_due_rms(payload, error) -> None:
        nonlocal rms_at
        now = time.perf_counter()
        while rms_at < CYCLE_S and now + 0.05 >= cycle_t0 + rms_at:
            _emit(_score_rms(payload, error, cycle_no, rms_at))
            rms_at += RMS_STEP_S
            if rms_at >= CYCLE_S or now + 0.05 < cycle_t0 + rms_at:
                break
            payload, error = _read_live(fetch, board)

    for pulse_at, deadline, kind, slot, counter in OPS:
        _wait_until(cycle_t0 + pulse_at, should_stop, stop_event)
        pending = _start_operation(
            fetch, board, controller, resolved, stop_event, kind, slot, counter,
        )
        passed = _poll_operation(
            fetch, board, pending, cycle_t0 + deadline, stop_event, should_stop,
            _score_due_rms,
        )
        _emit(_score_operation(passed[0], passed[1], pending, cycle_no))
        _wait_until(cycle_t0 + deadline, should_stop, stop_event)
    while rms_at < CYCLE_S:
        _wait_until(cycle_t0 + rms_at, should_stop, stop_event)
        payload, error = _read_live(fetch, board)
        _score_due_rms(payload, error)
    _wait_until(cycle_t0 + CYCLE_S, should_stop, stop_event)
    return rows


def _poll_operation(fetch, board, pending, deadline, stop_event, should_stop, on_sample):
    """Poll until ``deadline``. The first sample that matches is the score.

    An early match does not end the wait. The next pulse stays on its mark,
    and RMS samples that fall inside the window are still taken.
    """
    payload, error = None, None
    passed = None
    while time.perf_counter() < deadline:
        _raise_if_stopped(should_stop)
        if stop_event.is_set():
            raise MvbcmStopped("stopped by user")
        payload, error = _read_live(fetch, board)
        on_sample(payload, error)
        if passed is None and _operation_ready(payload, pending):
            passed = (payload, error)
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        _sleep(min(POLL_S, remaining), should_stop, stop_event)
    if passed is not None:
        return passed
    return payload, error


def _operation_ready(payload, pending: dict) -> bool:
    """True when the one new row and the counters match this command."""
    if payload is None or pending.get("before_ts") is None or pending.get("drive_error") is not None:
        return False
    if not isinstance(pending.get("before"), dict):
        return False
    fresh = [
        row for row in _rows_after(payload, pending["before_ts"])
        if str(row.get("rec_slot") or "").lower() != "backup_trip"
    ]
    slots = [str(row.get("rec_slot") or "").lower() for row in fresh]
    if slots != [pending["slot"]] or any(name in UNWANTED_SLOTS for name in slots):
        return False
    before = pending["before"]
    try:
        for name in COUNTERS:
            got = int(b7_value(payload, name))
            expect = before[name] + (1 if name == pending["counter"] else 0)
            if got != expect:
                return False
        for name in QUIET:
            if int(b7_value(payload, name)) != before[name]:
                return False
    except (LiveError, KeyError, TypeError, ValueError):
        return False
    return True


def _sleep(seconds: float, should_stop, stop_event: threading.Event) -> None:
    _wait_until(time.perf_counter() + seconds, should_stop, stop_event)


def _start_operation(fetch, board, controller, resolved, stop_event, kind, slot, counter) -> dict:
    """Snapshot, play the command, and return the state the next tick scores."""
    before_ts, before, snap_error = _snapshot(fetch, board)
    drive_error = None
    try:
        _play(controller, _channels_for(resolved, kind), kind, stop_event)
    except MvbcmStopped:
        raise
    except Exception as exc:
        drive_error = exc
    return {
        "kind": kind,
        "slot": slot,
        "counter": counter,
        "before_ts": before_ts,
        "before": before,
        "snap_error": snap_error,
        "drive_error": drive_error,
    }


def _channels_for(resolved: dict[str, str], kind: str) -> dict[str, str]:
    """Trip omits the backup coil so only the main trip line pulses."""
    if kind == "trip":
        return {key: value for key, value in resolved.items() if key != "backup_trip_coil"}
    return dict(resolved)


def _play(controller, channels: dict[str, str], kind: str, stop_event: threading.Event) -> None:
    _ensure_daq_path()
    from daq_io.di_waveform import OperationStopped

    operation = "close" if kind == "close" else "trip"
    loop = getattr(controller, "_op_loop_thread", None)
    if loop is not None and loop.is_alive():
        raise RuntimeError("Stop the loop before applying a single trip or close")
    lock = controller._waveform_lock
    if not lock.acquire(blocking=False):
        raise RuntimeError("Trip/close already running")
    try:
        controller.play_breaker_operation(channels, operation, stop_event)
    except OperationStopped as exc:
        raise MvbcmStopped("stopped by user") from exc
    finally:
        lock.release()


def _score_rms(payload, error, cycle_no: int, mark: float) -> list[dict]:
    note = _digital_note(payload)
    when = f"t={mark:.0f}s"
    rows = []
    for signal, expect, tol, unit in RMS_SPECS:
        low, high = expect - tol, expect + tol
        if payload is None:
            rows.append(_result(
                cycle_no, signal, "rms", f"{expect:.3f}", unit, "",
                f"{low:.3f}", f"{high:.3f}", "FAIL",
                f"{when}; {error or 'no live payload'}",
            ))
            continue
        try:
            measured = measure(payload, signal, "rms_eng")
        except LiveError as exc:
            rows.append(_result(
                cycle_no, signal, "rms", f"{expect:.3f}", unit, "",
                f"{low:.3f}", f"{high:.3f}", "FAIL", f"{when}; {exc}; {note}",
            ))
            continue
        ok = in_limits(measured, low, high)
        detail = f"{when}; rms_eng; {note}"
        if not ok:
            detail = f"expected [{low:.3f}, {high:.3f}] {unit}, got {measured:.3f}; {detail}"
        rows.append(_result(
            cycle_no, signal, "rms", f"{expect:.3f}", unit, f"{measured:.3f}",
            f"{low:.3f}", f"{high:.3f}", "PASS" if ok else "FAIL", detail,
        ))
    return rows


def _score_operation(payload, error, pending: dict, cycle_no: int) -> list[dict]:
    slot = pending["slot"]
    counter = pending["counter"]
    before = pending["before"]
    drive = pending["drive_error"]
    snap = pending["snap_error"]
    prefix = slot
    if drive is not None:
        prefix = f"{slot} drive failed: {drive}"
    elif snap is not None:
        prefix = f"{slot} snapshot failed: {snap}"
    rows = [_event_row(payload, error, pending, cycle_no, prefix)]
    for name in COUNTERS:
        rows.append(_counter_row(
            payload, error, name, before, name == counter, cycle_no, prefix,
        ))
    for name in QUIET:
        rows.append(_counter_row(
            payload, error, name, before, False, cycle_no, prefix,
        ))
    return rows


def _event_row(payload, error, pending, cycle_no, prefix) -> dict:
    slot = pending["slot"]
    before_ts = pending["before_ts"]
    if payload is None or before_ts is None:
        why = str(error or pending["snap_error"] or "no live payload")
        return _result(cycle_no, slot, "event", slot, "", "", slot, slot, "FAIL", f"{prefix}; {why}")
    fresh = [
        row for row in _rows_after(payload, before_ts)
        if str(row.get("rec_slot") or "").lower() != "backup_trip"
    ]
    slots = [str(row.get("rec_slot") or "").lower() for row in fresh]
    unwanted = [name for name in slots if name in UNWANTED_SLOTS]
    ok = len(fresh) == 1 and slots == [slot] and not unwanted and pending["drive_error"] is None
    shown = ",".join(slots) if slots else ""
    note = f"{prefix}; new rec_slot [{shown or 'none'}]"
    if unwanted:
        note = f"iph/imtr event {', '.join(unwanted)}; {note}"
    elif len(fresh) != 1 or slots != [slot]:
        note = f"expected one {slot} row; {note}"
    return _result(
        cycle_no, slot, "event", slot, "", shown, slot, slot,
        "PASS" if ok else "FAIL", note,
    )


def _counter_row(payload, error, name, before, increase, cycle_no, prefix) -> dict:
    if not isinstance(before, dict) or name not in before or payload is None:
        why = str(error or "counter was not read before the command")
        return _result(cycle_no, "b7", name, "", "", "", "", "", "FAIL", f"{prefix}; {why}")
    base = before[name]
    expect = base + 1 if increase else base
    try:
        got = int(b7_value(payload, name))
    except (LiveError, TypeError, ValueError) as exc:
        return _result(
            cycle_no, "b7", name, str(expect), "", "", str(expect), str(expect),
            "FAIL", f"{prefix}; {exc}",
        )
    ok = got == expect
    note = f"{prefix}; {name}"
    if not ok:
        note = f"expected {expect}, got {got}; {note}"
    return _result(
        cycle_no, "b7", name, str(expect), "", str(got), str(expect), str(expect),
        "PASS" if ok else "FAIL", note,
    )


def _snapshot(fetch, board):
    try:
        payload = fetch(board)
        before = {}
        for name in COUNTERS + QUIET:
            before[name] = int(b7_value(payload, name))
        return max_recent_ts(payload), before, None
    except (LiveError, TypeError, ValueError) as exc:
        return None, None, exc


def _rows_after(payload: dict, before_ts: int) -> list[dict]:
    fresh = []
    for row in operations_recent(payload):
        try:
            ts = int(row.get("ts_ms") or 0)
        except (TypeError, ValueError):
            continue
        if ts > before_ts:
            fresh.append(row)
    return fresh


def _read_live(fetch, board):
    try:
        return fetch(board), None
    except LiveError as exc:
        return None, exc


def _digital_note(payload) -> str:
    if not isinstance(payload, dict):
        return "digital unread"
    digital = payload.get("digital")
    if not isinstance(digital, list):
        return "digital missing"
    parts = []
    for entry in digital:
        if not isinstance(entry, dict):
            continue
        signal = entry.get("signal") or entry.get("id") or "?"
        active = entry.get("active")
        flag = "1" if active in (True, 1, "1", "true", "True") else "0"
        parts.append(f"{signal}={flag}")
    return "digital " + (",".join(parts) if parts else "none")


def _result(cycle, signal, phase, stimulus, unit, measured, low, high, result, note) -> dict:
    return {
        "group": "emc",
        "test_id": "EMC",
        "step": str(cycle),
        "signal": signal,
        "phase": phase,
        "stimulus_value": stimulus,
        "unit": unit,
        "measured": measured,
        "field": phase,
        "min": low,
        "max": high,
        "result": result,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "note": note,
    }


def _open_digital(channels, module, controller_factory):
    _ensure_daq_path()
    resolved = channels or default_trip_channels()
    if module is None:
        module = find_digital_module(device_name_from_channels(resolved))
    resolved = resolve_channels(module, resolved)
    factory = controller_factory or open_breaker_controller
    return resolved, factory(module), module


def _resolve_vctrl(module, resolved: dict[str, str]) -> str:
    known = list(getattr(module, "do_channels", None) or [])
    device = device_name_from_channels(resolved) or getattr(module, "name", "")
    hit = _match_line(known, device, VCTRL_LINE)
    if not hit:
        raise RuntimeError(f"Could not resolve Vctrl enable {VCTRL_LINE} on {device}")
    return hit


def _prime_closed(controller, channels: dict[str, str]) -> None:
    for key, state in _CLOSED.items():
        channel = channels.get(key)
        if not channel:
            continue
        _set_line(controller, channel, state, key)


def _set_line(controller, channel: str, state: bool, label: str = "vctrl") -> None:
    if controller.set_output_line(channel, state):
        return
    detail = ""
    getter = getattr(controller, "task_manager", None)
    if getter is not None and hasattr(getter, "get_last_error"):
        detail = (getter.get_last_error() or "").strip()
    raise RuntimeError(detail or f"Could not set {label}")


def _wait_until(deadline: float, should_stop, stop_event: threading.Event) -> None:
    while True:
        if stop_event.is_set() or (should_stop is not None and should_stop()):
            stop_event.set()
            raise MvbcmStopped("stopped by user")
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        if remaining > 0.02:
            time.sleep(min(0.01, remaining * 0.5))
