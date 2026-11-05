"""
ATS Test System - Web Application Server

A Flask-based web interface for the ATS Test System that exposes
all functionality through a REST API and serves a browser-based UI.

Run this to access the controller via localhost in your browser.

Usage:
    python web_app.py

Then open http://localhost:5000 in your browser.

Supports:
    - Device discovery and management
    - Analog input (single + continuous)
    - Analog output (DC + AC with waveforms)
    - Digital output (True/False per line, All True/All False)
    - System log viewing
"""

import sys
import os
import json
import time
import threading
from datetime import datetime
from pathlib import Path

# Ensure package directory is in path
PACKAGE_DIR = Path(__file__).parent.absolute()
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from flask import Flask, jsonify, request, render_template, send_from_directory, make_response

from app_logging.logger import initialize_logging, get_logger
from utils.config import global_config
from core.task_manager import TaskManager
from core.device_manager import DeviceManager, ModuleInfo
from daq_io.input import AnalogInputController, TerminalConfig
from daq_io.output import (
    AnalogOutputController, SignalType, WaveformType, OutputMode,
    convert_to_output_level,
)
from daq_io.digital_io import DigitalIOController
from daq_io.di_waveform import build_close_waveform, build_trip_waveform
from constants import (
    DEFAULT_HOST, DEFAULT_LOG_LEVEL, DEFAULT_SAMPLE_RATE,
    DEFAULT_SAMPLES_PER_CHANNEL, MAX_LOG_ENTRIES, DEFAULT_PORT,
    NETWORK_DEVICE_ADD_TIMEOUT, MILLIAMPS_TO_AMPS, OPERATION_INTERVAL_S,
)

# Initialize
initialize_logging(level=DEFAULT_LOG_LEVEL)
log = get_logger(__name__)
# Config is loaded lazily on first access; avoid blocking the network stack here.

# Create global instances
task_manager = TaskManager()
device_manager = DeviceManager()

# Active task tracking
_active_ai_tasks: dict = {}
_active_ao_tasks: dict = {}
_ao_controllers: dict = {}
_ao_module_locks: dict = {}
_ao_locks_guard = threading.Lock()
_dio_controllers: dict = {}
_mvbcm_lock = threading.Lock()
_mvbcm_stop = threading.Event()
_live_check_lock = threading.Lock()
_live_check_stop = threading.Event()
_live_check_state = {
    'running': False,
    'done': False,
    'stopped': False,
    'error': '',
    'rows': [],
    'board': '',
}
_mvbcm_state = {
    'running': False,
    'done': False,
    'stopped': False,
    'error': '',
    'rows': [],
    'passed': 0,
    'failed': 0,
    'device_idx': None,
    'module_idx': None,
    'board': '',
    'device': '',
}


def _ao_task_key(device_idx: int, module_idx: int, channel: str) -> str:
    safe = channel.replace('/', '_').replace(':', '_')
    return f"ao_{device_idx}_{module_idx}_{safe}"
_log_entries: list = []
_max_log_entries = MAX_LOG_ENTRIES

app = Flask(__name__, template_folder=str(PACKAGE_DIR.parent.parent / 'ui' / 'templates'),
            static_folder=str(PACKAGE_DIR.parent.parent / 'ui' / 'static'))
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


def add_log(message: str, level: str = 'INFO') -> None:
    """Add a log entry to the in-memory log buffer."""
    entry = {
        'time': datetime.now().strftime('%H:%M:%S'),
        'message': message,
        'level': level,
        'timestamp': time.time()
    }
    _log_entries.append(entry)
    if len(_log_entries) > _max_log_entries:
        _log_entries.pop(0)
    
    # Also log to the file logger
    getattr(log, level.lower(), log.info)(message)


def _parse_json():
    """Return (data dict, error_response) — error_response is set on bad JSON."""
    data = request.get_json(silent=True)
    if data is None:
        return None, (jsonify({'error': 'Invalid or missing JSON body'}), 400)
    return data, None


def _require_indices(data: dict):
    """Return (device_idx, module_idx, error_response) from request data."""
    if 'device_idx' not in data or 'module_idx' not in data:
        return None, None, (jsonify({'error': 'device_idx and module_idx are required'}), 400)
    try:
        return int(data['device_idx']), int(data['module_idx']), None
    except (TypeError, ValueError):
        return None, None, (jsonify({'error': 'device_idx and module_idx must be integers'}), 400)


def _ni_error_detail(fallback: str) -> str:
    """Append the last NI-DAQmx error to a user-facing message."""
    ni_err = (task_manager.get_last_error() or '').strip()
    if ni_err and ni_err not in fallback:
        return f"{fallback} ({ni_err})"
    return fallback or ni_err or 'Operation failed'


def _api_error(endpoint: str, exc: Exception, *, di=None, mi=None, channel=None) -> tuple:
    """Log, record in UI log, and return a JSON 500 response."""
    log.exception(
        "%s failed device_idx=%s module_idx=%s channel=%s",
        endpoint, di, mi, channel,
    )
    msg = _ni_error_detail(str(exc))
    add_log(f"{endpoint}: {msg}", 'ERROR')
    return jsonify({'error': msg}), 500


def stop_all_ai_tasks() -> None:
    """Stop all continuous analog-input background tasks."""
    for task_key, info in list(_active_ai_tasks.items()):
        try:
            info['controller'].stop_acquisition(info['task_name'])
        except Exception as e:
            log.warning("Failed to stop AI task %s: %s", task_key, e)
    _active_ai_tasks.clear()


def get_controller(device_idx: int, module_idx: int):
    """Get a device and module by index from the discovered list."""
    devices = device_manager.get_all_devices()
    if not devices or device_idx < 0 or device_idx >= len(devices):
        return None, None, f"Device not found (index {device_idx} out of range)"
    
    device = devices[device_idx]
    if module_idx < 0 or module_idx >= len(device.modules):
        return None, None, f"Module not found (index {module_idx} out of range)"
    
    return device, device.modules[module_idx], None


def get_dio_controller(device_idx: int, module_idx: int, module):
    """Reuse DigitalIOController per module so port output tasks persist."""
    key = f"{device_idx}_{module_idx}"
    if key not in _dio_controllers:
        _dio_controllers[key] = DigitalIOController(task_manager, module)
    return _dio_controllers[key]


def _find_trip_dio_indices(prefer_name: str | None = None):
    """Locate USB trip/close DO module indices (same device Apply trip uses).

    Prefers ``prefer_name`` (e.g. Dev9), then any USB-6509 with DO lines.
    Returns ``(device_idx, module_idx, module)``.
    """
    devices = device_manager.get_all_devices()
    if not devices:
        raise RuntimeError("No NI devices discovered for trip/close")
    candidates = []
    for di, device in enumerate(devices):
        modules = list(device.modules or [])
        if not modules and list(getattr(device, "do_channels", None) or []):
            # Standalone USB DO device (e.g. USB-6509) with no child modules.
            modules = [device]
        for mi, module in enumerate(modules):
            do_channels = list(getattr(module, "do_channels", None) or [])
            if not do_channels:
                continue
            name = getattr(module, "name", "") or ""
            product = (getattr(module, "product_type", "") or "").upper()
            if prefer_name and name != prefer_name and device.name != prefer_name:
                continue
            score = len(do_channels)
            if "6509" in product or "6509" in name.upper():
                score += 1000
            if prefer_name and (name == prefer_name or device.name == prefer_name):
                score += 5000
            candidates.append((score, di, mi, module))
    if prefer_name and not candidates:
        raise RuntimeError(f"No digital module matching {prefer_name!r} for trip/close")
    if not candidates:
        raise RuntimeError("No digital-output module found for trip/close (need USB-6509)")
    candidates.sort(key=lambda item: item[0], reverse=True)
    _score, di, mi, module = candidates[0]
    return di, mi, module


def get_ao_controller(device_idx: int, module_idx: int, module):
    """Reuse AnalogOutputController per module so AO tasks can be stopped."""
    key = f"{device_idx}_{module_idx}"
    if key not in _ao_controllers:
        _ao_controllers[key] = AnalogOutputController(task_manager, module)
    return _ao_controllers[key]


def _ao_module_lock(device_idx: int, module_idx: int) -> threading.Lock:
    """Get or create a per-module lock for AO operations (thread-safe)."""
    key = f"{device_idx}_{module_idx}"
    with _ao_locks_guard:
        if key not in _ao_module_locks:
            _ao_module_locks[key] = threading.Lock()
        return _ao_module_locks[key]


def release_module_ao(device_idx: int, module_idx: int, module) -> None:
    """Stop tracked AO tasks and clear NI-DAQmx tasks on one module."""
    controller = get_ao_controller(device_idx, module_idx, module)
    controller.stop_all_outputs()
    prefix = f"ao_{device_idx}_{module_idx}_"
    for task_key in list(_active_ao_tasks.keys()):
        if task_key.startswith(prefix):
            del _active_ao_tasks[task_key]
    task_manager.clear_tasks_for_device(module.name)


def _ao_error_message() -> str:
    return task_manager.get_last_error() or 'Failed to start output'


def _clear_ao_channel_tasks(module_name: str, channel: str) -> None:
    """Release only AO tasks for one channel on one module."""
    task_manager.clear_ao_tasks_for_channel(module_name, channel)


def _remove_ao_task_refs(task_name: str) -> None:
    """Remove all tracked AO channel keys that reference the same task."""
    for task_key in list(_active_ao_tasks.keys()):
        if _active_ao_tasks.get(task_key) == task_name:
            del _active_ao_tasks[task_key]


def _stop_ao_tasks_for_channels(device_idx: int, module_idx: int, module,
                                controller, channel_list: list) -> None:
    """Stop AO tasks that touch any of the given channels."""
    task_names = set()
    for channel in channel_list:
        resolved = controller._resolve_channel(channel)
        if not resolved:
            continue
        task_key = _ao_task_key(device_idx, module_idx, resolved)
        if task_key in _active_ao_tasks:
            task_names.add(_active_ao_tasks[task_key])
        task_manager.clear_ao_tasks_for_channel(module.name, resolved)

    for task_name in task_names:
        controller.stop_output(task_name)
        _remove_ao_task_refs(task_name)


def reset_all_dio_controllers() -> None:
    """Release digital I/O tasks and cached state (call on device refresh)."""
    stop_all_ai_tasks()
    for controller in list(_dio_controllers.values()):
        controller.cleanup()
    _dio_controllers.clear()
    for controller in list(_ao_controllers.values()):
        controller.stop_all_outputs()
    _ao_controllers.clear()
    _active_ao_tasks.clear()


def _build_device_details(devices) -> list:
    """Serialize DeviceInfo list for the JSON API."""
    return [_serialize_device(device) for device in devices]


def _serialize_device(device) -> dict:
    """Serialize a single DeviceInfo for the JSON API."""
    modules_data = []
    for mod in device.modules:
        modules_data.append({
            'name': mod.name,
            'slot_number': mod.slot_number,
            'product_type': mod.product_type,
            'serial_number': mod.serial_number,
            'supported_operations': mod.supported_operations,
            'ai_channels': mod.ai_channels,
            'ao_channels': mod.ao_channels,
            'di_channels': mod.di_channels,
            'do_channels': mod.do_channels,
            'ci_channels': mod.ci_channels,
            'co_channels': mod.co_channels,
            'voltage_ranges': [f"{r[0]:.1f} to {r[1]:.1f} V" for r in mod.voltage_ranges],
            'max_sample_rate': mod.max_sample_rate,
            'is_simulated': mod.is_simulated,
            'ao_output_unit': (
                'current' if '9266' in (mod.product_type or '').upper() else 'voltage'
            ),
        })

    return {
        'name': device.name,
        'product_type': device.product_type,
        'serial_number': device.serial_number,
        'connection_type': device.connection_type.value,
        'ip_address': device.ip_address,
        'status': device.status.value,
        'module_count': len(device.modules),
        'ai_channels': len(device.ai_channels),
        'ao_channels': len(device.ao_channels),
        'di_channels': len(device.di_channels),
        'do_channels': len(device.do_channels),
        'is_simulated': device.is_simulated,
        'modules': modules_data,
    }


def _device_api_payload(devices, *, scanned: bool) -> dict:
    """Build /api/devices JSON body from a device list."""
    device_details = _build_device_details(devices)
    connected = len(device_details)
    total = connected

    if scanned:
        add_log(f"Device discovery: {connected} device(s) found")

    hints = []
    if scanned:
        discovery_error = device_manager.get_last_discovery_error()
        if discovery_error:
            hints.append(f"NI-DAQmx scan error: {discovery_error}")
    if not device_details and scanned:
        hints.append(
            "USB DAQ devices appear automatically when plugged in and visible in NI MAX. "
            "For Ethernet cDAQ, add and reserve the chassis in NI MAX (or use Add Network "
            "Device below), then click Refresh."
        )

    connection_summary = device_manager.get_connection_summary()

    status = f"{connected} device(s) connected"
    if connection_summary:
        breakdown = ", ".join(
            f"{n} {t}" for t, n in sorted(connection_summary.items())
        )
        status = f"{connected} connected ({breakdown})"
    elif total == 0 and not scanned:
        status = "No devices cached — click Refresh to scan"

    payload = {
        'devices': device_details,
        'status': status,
        'connection_summary': connection_summary,
        'cached': not scanned,
        'log': _log_entries[-30:],
        'hints': hints,
    }
    return payload


# ===================== API Routes =====================

@app.errorhandler(Exception)
def handle_unhandled_exception(e):
    """Log unexpected errors; pass HTTP exceptions through to Flask."""
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return e
    log.exception("Unhandled server error")
    add_log(f"Unhandled error: {e}", 'ERROR')
    return jsonify({'error': str(e)}), 500


@app.route('/')
def index():
    """Serve the main web interface."""
    resp = make_response(render_template('index.html'))
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    return resp


@app.route('/api/devices')
def api_devices():
    """API: Return cached devices. Pass ?refresh=1 to rescan hardware."""
    try:
        refresh = request.args.get('refresh', '').lower() in ('1', 'true', 'yes')
        if refresh:
            devices = device_manager.discover_devices()
            reset_all_dio_controllers()
        else:
            devices = device_manager.get_all_devices()

        return jsonify(_device_api_payload(devices, scanned=refresh))
    except Exception as e:
        log.exception("Device API error")
        add_log(f"Device API error: {e}", 'ERROR')
        return jsonify({'error': str(e), 'devices': [], 'log': _log_entries[-10:]})


@app.route('/api/devices/add-network', methods=['POST'])
def api_add_network_device():
    """API: Add and reserve an Ethernet cDAQ chassis by IP or hostname."""
    try:
        data = request.json or {}
        ip_or_hostname = (
            data.get('ip_or_hostname')
            or data.get('ip')
            or data.get('hostname')
        )
        if not ip_or_hostname:
            return jsonify({'error': 'ip_or_hostname is required'}), 400

        device_name = (data.get('device_name') or '').strip()
        attempt_reservation = data.get('attempt_reservation', True)
        timeout = float(data.get('timeout', NETWORK_DEVICE_ADD_TIMEOUT))

        added_name = device_manager.add_network_device(
            ip_or_hostname.strip(),
            device_name=device_name,
            attempt_reservation=attempt_reservation,
            timeout=timeout,
        )
        devices = device_manager.discover_devices()
        add_log(f"Added network device: {added_name} ({ip_or_hostname})")

        return jsonify({
            'success': True,
            'device_name': added_name,
            'device_count': len(devices),
            **_device_api_payload(devices, scanned=True),
        })
    except Exception as e:
        log.exception("Add network device error")
        add_log(f"Failed to add network device: {e}", 'ERROR')
        return jsonify({'error': str(e)}), 400


@app.route('/api/ai/read', methods=['POST'])
def api_ai_read():
    """API: Read analog input (single shot)."""
    di = mi = None
    try:
        data, err = _parse_json()
        if err:
            return err
        di, mi, err = _require_indices(data)
        if err:
            return err
        channels = data.get('channels', [])
        rate = float(data.get('rate', DEFAULT_SAMPLE_RATE))
        samples = int(data.get('samples', DEFAULT_SAMPLES_PER_CHANNEL))
        terminal_config = data.get('terminal_config', 'RSE')

        device, module, error = get_controller(di, mi)
        if error:
            return jsonify({'error': error}), 400

        controller = AnalogInputController(task_manager, module)
        task_name = controller.start_single_acquisition(
            channels, rate, samples, terminal_config=terminal_config
        )
        if not task_name:
            err_msg = _ni_error_detail('Failed to start acquisition')
            add_log(f"AI read failed: {err_msg}", 'ERROR')
            return jsonify({'error': err_msg}), 500

        task_manager.start_task(task_name)
        result = controller.read_data(task_name)
        task_manager.clear_task(task_name)

        if result and result.success:
            values = []
            for ch, ch_data in result.data.items():
                if len(ch_data) > 0:
                    values.append(f"{ch_data[-1]:.4f} V")
            add_log(f"AI read: {device.name}/{module.name} channels={channels} terminal={terminal_config}")
            return jsonify({'values': values, 'success': True, 'terminal_config': terminal_config})
        
        err_msg = _ni_error_detail('No data returned')
        add_log(f"AI read failed: {err_msg}", 'ERROR')
        return jsonify({'error': err_msg}), 500
    except (TypeError, ValueError) as e:
        return jsonify({'error': f'Invalid request: {e}'}), 400
    except Exception as e:
        return _api_error('AI read', e, di=di, mi=mi)


@app.route('/api/ai/start', methods=['POST'])
def api_ai_start():
    """API: Start continuous analog input acquisition."""
    di = mi = None
    try:
        data, err = _parse_json()
        if err:
            return err
        di, mi, err = _require_indices(data)
        if err:
            return err
        channels = data.get('channels', [])
        rate = float(data.get('rate', DEFAULT_SAMPLE_RATE))
        samples = int(data.get('samples', DEFAULT_SAMPLES_PER_CHANNEL))
        terminal_config = data.get('terminal_config', 'RSE')

        device, module, error = get_controller(di, mi)
        if error:
            return jsonify({'error': error}), 400

        controller = AnalogInputController(task_manager, module)

        # Stop any existing continuous task for this module
        task_key = f"{di}_{mi}"
        if task_key in _active_ai_tasks:
            old_ctrl = _active_ai_tasks[task_key]['controller']
            old_ctrl.stop_acquisition(_active_ai_tasks[task_key]['task_name'])

        def data_callback(result):
            """Callback for continuous acquisition updates."""
            try:
                if not result.success:
                    return
                if task_key not in _active_ai_tasks:
                    return
                values = []
                for ch, ch_data in result.data.items():
                    if len(ch_data) > 0:
                        values.append(f"{ch_data[-1]:.4f} V")
                _active_ai_tasks[task_key]['latest'] = values
            except Exception as cb_e:
                log.error("AI data callback error: %s", cb_e)

        task_name = controller.start_continuous_acquisition(
            channels, rate, samples, terminal_config=terminal_config,
            data_callback=data_callback
        )

        if task_name:
            _active_ai_tasks[task_key] = {
                'task_name': task_name,
                'controller': controller,
                'latest': ['Starting...'],
                'started': time.time(),
                'terminal_config': terminal_config,
            }
            add_log(f"Continuous AI started: {device.name}/{module.name} terminal={terminal_config}")
            return jsonify({'task_name': task_name, 'success': True, 'terminal_config': terminal_config})
        
        err_msg = _ni_error_detail('Failed to start continuous acquisition')
        add_log(f"AI start failed: {err_msg}", 'ERROR')
        return jsonify({'error': err_msg}), 500
    except (TypeError, ValueError) as e:
        return jsonify({'error': f'Invalid request: {e}'}), 400
    except Exception as e:
        return _api_error('AI start', e, di=di, mi=mi)


@app.route('/api/ai/stop', methods=['POST'])
def api_ai_stop():
    """API: Stop continuous analog input acquisition."""
    di = mi = None
    try:
        data, err = _parse_json()
        if err:
            return err
        di, mi, err = _require_indices(data)
        if err:
            return err
        task_key = f"{di}_{mi}"
        
        if task_key in _active_ai_tasks:
            info = _active_ai_tasks.pop(task_key)
            info['controller'].stop_acquisition(info['task_name'])
            add_log("Continuous AI stopped")
            return jsonify({'success': True})
        
        return jsonify({'success': True, 'message': 'No active task'})
    except Exception as e:
        return _api_error('AI stop', e, di=di, mi=mi)


@app.route('/api/ao/start', methods=['POST'])
def api_ao_start():
    """API: Start analog output."""
    di = mi = channel = None
    try:
        data, err = _parse_json()
        if err:
            return err
        di, mi, err = _require_indices(data)
        if err:
            return err
        channel = data.get('channel')
        if not channel:
            return jsonify({'error': 'channel is required'}), 400
        signal = data.get('signal', 'DC')
        value_mode = data.get('value_mode', 'direct')
        value = data.get('value', 1.0)
        frequency = float(data.get('frequency', 50.0))
        waveform = data.get('waveform', 'Sine')
        amplitude = data.get('amplitude', 1.0)

        device, module, error = get_controller(di, mi)
        if error:
            return jsonify({'error': error}), 400

        try:
            wf_type = WaveformType(waveform)
        except ValueError:
            return jsonify({'error': f'Invalid waveform: {waveform}'}), 400

        with _ao_module_lock(di, mi):
            task_manager.clear_last_error()
            controller = get_ao_controller(di, mi, module)
            resolved = controller._resolve_channel(channel)
            if not resolved:
                return jsonify({'error': f'Invalid AO channel: {channel}'}), 400
            channel = resolved
            is_current = '9266' in (module.product_type or '').upper()
            output_mode = OutputMode.CURRENT if is_current else OutputMode.VOLTAGE

            if signal == 'DC':
                value = convert_to_output_level(
                    float(value), SignalType.DC, value_mode, wf_type
                )
                applied_display = float(value)
            else:
                amplitude = convert_to_output_level(
                    float(amplitude), SignalType.AC, value_mode, wf_type
                )
                applied_display = float(amplitude)

            if is_current:
                value = value / MILLIAMPS_TO_AMPS
                amplitude = amplitude / MILLIAMPS_TO_AMPS
                applied_display = round(applied_display, 4)

            display_unit = 'mA' if is_current else 'V'
            applied_out = round(applied_display, 4)

            task_key = _ao_task_key(di, mi, channel)
            if signal == 'DC' and task_key in _active_ao_tasks:
                task_name = _active_ao_tasks[task_key]
                if controller.update_dc_value(task_name, value):
                    add_log(
                        f"AO updated: {device.name}/{module.name} DC channel={channel} "
                        f"value={applied_out} {display_unit}"
                    )
                    return jsonify({
                        'task_name': task_name,
                        'success': True,
                        'updated': True,
                        'applied_value': applied_out,
                        'unit': display_unit,
                        'value_mode': value_mode,
                    })

            # Stop all existing AO tasks on this module — cDAQ modules share
            # the analog output subsystem, so a task on one channel reserves
            # the entire module.
            _prefix = f"ao_{di}_{mi}_"
            _task_names_to_stop = set()
            for _existing_key in list(_active_ao_tasks.keys()):
                if _existing_key.startswith(_prefix):
                    _task_names_to_stop.add(_active_ao_tasks[_existing_key])
            for _task_name in _task_names_to_stop:
                controller.stop_output(_task_name)
                _remove_ao_task_refs(_task_name)

            _clear_ao_channel_tasks(module.name, channel)

            if signal == 'DC':
                task_name = controller.start_dc_output(
                    channel, value, output_mode=output_mode
                )
            else:
                task_name = controller.start_ac_output(
                    channel, wf_type, frequency, amplitude, output_mode=output_mode
                )

            if task_name:
                _active_ao_tasks[task_key] = task_name
                add_log(f"AO started: {device.name}/{module.name} {signal} channel={channel}")
                return jsonify({
                    'task_name': task_name,
                    'success': True,
                    'value_mode': value_mode,
                    'applied_value': applied_out,
                    'unit': display_unit,
                })

            err_msg = _ao_error_message()
            add_log(f"AO start failed: {err_msg}", 'ERROR')
            return jsonify({'error': err_msg}), 500
    except (TypeError, ValueError) as e:
        return jsonify({'error': f'Invalid request: {e}'}), 400
    except Exception as e:
        return _api_error('AO start', e, di=di, mi=mi, channel=channel)


@app.route('/api/ao/write-config-module', methods=['POST'])
def api_ao_write_config_module():
    """API: Write the same signal configuration to selected analog output channels."""
    di = mi = None
    try:
        data, err = _parse_json()
        if err:
            return err
        di, mi, err = _require_indices(data)
        if err:
            return err
        signal = data.get('signal', 'DC')
        value_mode = data.get('value_mode', 'direct')
        value = data.get('value', 1.0)
        frequency = float(data.get('frequency', 50.0))
        waveform = data.get('waveform', 'Sine')
        amplitude = data.get('amplitude', 1.0)
        requested_channels = data.get('channels')

        device, module, error = get_controller(di, mi)
        if error:
            return jsonify({'error': error}), 400

        try:
            wf_type = WaveformType(waveform)
        except ValueError:
            return jsonify({'error': f'Invalid waveform: {waveform}'}), 400

        if not module.ao_channels:
            return jsonify({'error': 'No analog output channels on this module'}), 400

        if requested_channels is not None:
            if not isinstance(requested_channels, list):
                return jsonify({'error': 'channels must be a list'}), 400
            if not requested_channels:
                return jsonify({'error': 'Select at least one channel'}), 400
            module_channel_set = set(module.ao_channels)
            target_channels = [
                ch for ch in requested_channels if ch in module_channel_set
            ]
            if not target_channels:
                return jsonify({'error': 'No valid channels selected'}), 400
        else:
            target_channels = list(module.ao_channels)

        with _ao_module_lock(di, mi):
            task_manager.clear_last_error()
            controller = get_ao_controller(di, mi, module)
            is_current = '9266' in (module.product_type or '').upper()
            output_mode = OutputMode.CURRENT if is_current else OutputMode.VOLTAGE
            display_unit = 'mA' if is_current else 'V'

            # Stop all existing AO tasks on this module — cDAQ modules share
            # the analog output subsystem, so a task on one channel reserves
            # the entire module.
            _prefix = f"ao_{di}_{mi}_"
            _task_names_to_stop = set()
            for _existing_key in list(_active_ao_tasks.keys()):
                if _existing_key.startswith(_prefix):
                    _task_names_to_stop.add(_active_ao_tasks[_existing_key])
            for _task_name in _task_names_to_stop:
                controller.stop_output(_task_name)
                _remove_ao_task_refs(_task_name)

            _stop_ao_tasks_for_channels(di, mi, module, controller, target_channels)

            results = []
            applied_out = None

            if signal == 'AC':
                amplitude_val = convert_to_output_level(
                    float(amplitude), SignalType.AC, value_mode, wf_type
                )
                applied_display = float(amplitude_val)
                actual_amplitude = amplitude_val / MILLIAMPS_TO_AMPS if is_current else amplitude_val
                if is_current:
                    applied_display = round(applied_display, 4)
                applied_out = round(applied_display, 4)

                channel_pairs = []
                for channel in target_channels:
                    resolved = controller._resolve_channel(channel)
                    if resolved:
                        channel_pairs.append((channel, resolved))
                    else:
                        results.append({'channel': channel, 'success': False, 'error': 'Invalid channel'})

                if channel_pairs:
                    resolved_channels = [resolved for _, resolved in channel_pairs]
                    task_name = controller.start_ac_output_multi(
                        resolved_channels, wf_type, frequency, actual_amplitude,
                        output_mode=output_mode
                    )
                    if task_name:
                        for channel, resolved in channel_pairs:
                            task_key = _ao_task_key(di, mi, resolved)
                            _active_ao_tasks[task_key] = task_name
                            results.append({
                                'channel': channel,
                                'success': True,
                                'applied_value': applied_out,
                            })
                    else:
                        err = _ao_error_message()
                        for channel, _ in channel_pairs:
                            results.append({
                                'channel': channel,
                                'success': False,
                                'error': err or 'Failed to start output',
                            })
            else:
                for channel in target_channels:
                    try:
                        resolved = controller._resolve_channel(channel)
                        if not resolved:
                            results.append({'channel': channel, 'success': False, 'error': 'Invalid channel'})
                            continue

                        output_value = convert_to_output_level(
                            float(value), SignalType.DC, value_mode, wf_type
                        )
                        applied_display = float(output_value)

                        if is_current:
                            actual_value = output_value / MILLIAMPS_TO_AMPS
                            applied_display = round(applied_display, 4)
                        else:
                            actual_value = output_value

                        applied_out = round(applied_display, 4)

                        _clear_ao_channel_tasks(module.name, resolved)

                        task_name = controller.start_dc_output(
                            resolved, actual_value, output_mode=output_mode
                        )

                        if task_name:
                            task_key = _ao_task_key(di, mi, resolved)
                            _active_ao_tasks[task_key] = task_name
                            results.append({'channel': channel, 'success': True, 'applied_value': applied_out})
                        else:
                            results.append({'channel': channel, 'success': False, 'error': 'Failed to start output'})
                    except Exception as ch_e:
                        results.append({'channel': channel, 'success': False, 'error': str(ch_e)})

            success_count = sum(1 for r in results if r['success'])
            add_log(
                f"AO write-config module={module.name} {signal}: "
                f"{success_count}/{len(results)} channel(s) written"
            )
            return jsonify({
                'success': success_count > 0,
                'results': results,
                'signal': signal,
                'applied_value': round(applied_out, 4) if results else None,
                'unit': display_unit,
            })

    except (TypeError, ValueError) as e:
        return jsonify({'error': f'Invalid request: {e}'}), 400
    except Exception as e:
        return _api_error('AO write-config-module', e, di=di, mi=mi)


@app.route('/api/ao/stop', methods=['POST'])
def api_ao_stop():
    """API: Stop analog output."""
    di = mi = None
    try:
        data, err = _parse_json()
        if err:
            return err
        di, mi, err = _require_indices(data)
        if err:
            return err
        channel = data.get('channel')

        device, module, error = get_controller(di, mi)
        if error:
            return jsonify({'error': error}), 400

        controller = get_ao_controller(di, mi, module)
        with _ao_module_lock(di, mi):
            if channel:
                task_key = _ao_task_key(di, mi, channel)
                if task_key in _active_ao_tasks:
                    task_name = _active_ao_tasks[task_key]
                    controller.stop_output(task_name)
                    _remove_ao_task_refs(task_name)
                    add_log(f"AO stopped: {channel}")
            else:
                prefix = f"ao_{di}_{mi}_"
                for task_key in list(_active_ao_tasks.keys()):
                    if task_key.startswith(prefix):
                        controller.stop_output(_active_ao_tasks[task_key])
                        del _active_ao_tasks[task_key]
                task_manager.clear_tasks_for_device(module.name)
                add_log("AO stopped (all channels on module)")

        return jsonify({'success': True})
    except Exception as e:
        return _api_error('AO stop', e, di=di, mi=mi)


def _board_url(text: str) -> str:
    """Accept an IP, host:port, or full URL. Bare hosts get port 8080."""
    raw = (text or '').strip().rstrip('/')
    if not raw:
        raise ValueError('Board IP is required')
    if '://' not in raw:
        if ':' not in raw:
            raw = f'{raw}:8080'
        raw = 'http://' + raw
    return raw


def _stop_module_ao(di: int, mi: int) -> str:
    """Release this module's analog output so the test task can take it."""
    _device, module, error = get_controller(di, mi)
    if error:
        raise RuntimeError(error)
    controller = get_ao_controller(di, mi, module)
    with _ao_module_lock(di, mi):
        prefix = f"ao_{di}_{mi}_"
        for task_key in list(_active_ao_tasks.keys()):
            if task_key.startswith(prefix):
                controller.stop_output(_active_ao_tasks[task_key])
                del _active_ao_tasks[task_key]
        task_manager.clear_tasks_for_device(module.name)
    return module.name


def _stop_modules_by_product(token: str) -> None:
    """Release every module of this product so a second AO task can open."""
    needle = token.upper()
    for di, device in enumerate(device_manager.get_all_devices()):
        for mi, module in enumerate(device.modules):
            if needle in (module.product_type or '').upper():
                _stop_module_ao(di, mi)


def _mvbcm_counts(rows: list) -> tuple:
    passed = sum(1 for row in rows if row.get('result') == 'PASS')
    return passed, len(rows) - passed


@app.route('/api/mvbcm/status')
def api_mvbcm_status():
    """API: Current AI-01 run, including rows scored so far."""
    with _mvbcm_lock:
        passed, failed = _mvbcm_counts(_mvbcm_state['rows'])
        return jsonify({
            'running': _mvbcm_state['running'],
            'done': _mvbcm_state['done'],
            'stopped': _mvbcm_state.get('stopped', False),
            'error': _mvbcm_state['error'],
            'rows': list(_mvbcm_state['rows']),
            'passed': passed,
            'failed': failed,
            'device_idx': _mvbcm_state['device_idx'],
            'module_idx': _mvbcm_state['module_idx'],
            'board': _mvbcm_state['board'],
            'device': _mvbcm_state['device'],
        })


@app.route('/api/mvbcm/stop', methods=['POST'])
def api_mvbcm_stop():
    """API: Request stop of the in-flight MV BCM run."""
    with _mvbcm_lock:
        running = _mvbcm_state['running']
    if not running:
        return jsonify({'stopped': False, 'message': 'No test running'})
    _mvbcm_stop.set()
    add_log('MV BCM stop requested')
    return jsonify({'stopped': True})


@app.route('/api/mvbcm/run', methods=['POST'])
def api_mvbcm_run():
    """API: Run the enabled MV BCM group against the board live API."""
    di = mi = None
    try:
        data, err = _parse_json()
        if err:
            return err
        di, mi, err = _require_indices(data)
        if err:
            return err
        try:
            board = _board_url(data.get('board', ''))
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400

        # Independent sections: analog / CT / trip_close / di_check.
        if 'analog' in data:
            run_analog = bool(data.get('analog'))
        else:
            run_analog = True
        run_ct = bool(data.get('ct', False))
        trip_close = bool(data.get('trip_close', True))
        run_di = bool(data.get('di_check', False))
        run_fault = bool(data.get('fault_trip_close', False))
        if not run_analog and not run_ct and not trip_close and not run_di and not run_fault:
            return jsonify({
                'error': 'Select Analog, CT, Trip/close, DI check, and/or Fault trip/close',
            }), 400
        try:
            trip_close_cycles = int(data.get('trip_close_cycles', 1))
        except (TypeError, ValueError):
            return jsonify({'error': 'trip_close_cycles must be an integer'}), 400
        if trip_close and trip_close_cycles < 1:
            return jsonify({'error': 'trip_close_cycles must be >= 1'}), 400
        try:
            di_cycles = int(data.get('di_cycles', 1))
        except (TypeError, ValueError):
            return jsonify({'error': 'di_cycles must be an integer'}), 400
        if run_di and di_cycles < 1:
            return jsonify({'error': 'di_cycles must be >= 1'}), 400
        try:
            fault_cycles = int(data.get('fault_cycles', 1))
        except (TypeError, ValueError):
            return jsonify({'error': 'fault_cycles must be an integer'}), 400
        if run_fault and fault_cycles < 1:
            return jsonify({'error': 'fault_cycles must be >= 1'}), 400
        try:
            harmonic_h3_pct = float(data.get('harmonic_h3_pct', 10.0))
        except (TypeError, ValueError):
            return jsonify({'error': 'harmonic_h3_pct must be a number'}), 400
        harmonic_map = {}
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.append(str(repo_root))
        try:
            from nidaq_controller.mvbcm.harmonic_ops import normalize_harmonics
        except ModuleNotFoundError:
            from mvbcm.harmonic_ops import normalize_harmonics
        try:
            harmonic_map = normalize_harmonics(
                data.get('harmonics'),
                fallback_h3_pct=harmonic_h3_pct,
                require=False,
            )
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400
        run_iph_h = bool(run_analog and harmonic_map)
        run_imtr_h = bool(run_ct and harmonic_map)

        _device, module, error = get_controller(di, mi)
        if error:
            return jsonify({'error': error}), 400
        if '9266' not in (module.product_type or '').upper():
            return jsonify({'error': 'This test runs on an NI-9266'}), 400

        with _mvbcm_lock:
            if _mvbcm_state['running']:
                return jsonify({'error': 'A test is already running'}), 409
            _mvbcm_stop.clear()
            _mvbcm_state.update({
                'running': True,
                'done': False,
                'stopped': False,
                'error': '',
                'rows': [],
                'passed': 0,
                'failed': 0,
                'device_idx': di,
                'module_idx': mi,
                'board': board,
                'device': module.name,
            })

        device_name = module.name
        trip_device = (data.get('trip_device') or '').strip() or None

        def _should_stop() -> bool:
            return _mvbcm_stop.is_set()

        def _worker():
            rows = []
            stopped = False
            try:
                repo_root = Path(__file__).resolve().parents[2]
                if str(repo_root) not in sys.path:
                    sys.path.append(str(repo_root))
                from nidaq_controller.mvbcm.run_mvbcm import (
                    append_results,
                    run_groups,
                    DEFAULT_RESULTS,
                )
                from nidaq_controller.mvbcm.sheet import load_steps
                from nidaq_controller.mvbcm.channel_map import enabled_groups
                from nidaq_controller.mvbcm.breaker_ops import (
                    EVENT_SETTLE_S,
                    MvbcmStopped,
                    default_trip_channels,
                    device_name_from_channels,
                    run_trip_close_test,
                )
                from nidaq_controller.mvbcm.ct_ops import run_ct_peak_test
                from nidaq_controller.mvbcm.di_ops import DI_SETTLE_S, run_di_check_test
                from nidaq_controller.mvbcm.fault_ops import (
                    FAULT_EVENT_WAIT_S,
                    run_fault_trip_close_test,
                )
                from nidaq_controller.mvbcm.harmonic_ops import run_harmonic_test

                _stop_module_ao(di, mi)
                _stop_modules_by_product('9264')
                parts = []
                if run_analog:
                    parts.append('analog')
                if run_ct:
                    parts.append('CT')
                if run_iph_h or run_imtr_h:
                    hn = '+'.join(
                        f'H{n} {pct:g}%' for n, pct in harmonic_map.items()
                    )
                    who = []
                    if run_iph_h:
                        who.append('iph')
                    if run_imtr_h:
                        who.append('imtr')
                    parts.append(f'harmonic {"+".join(who)} {hn}')
                if run_di:
                    parts.append(f'DI check×{di_cycles}')
                if run_fault:
                    parts.append(f'fault trip/close×{fault_cycles}')
                if trip_close:
                    parts.append(f'trip/close×{trip_close_cycles}')
                add_log(
                    f"MV BCM run started on {device_name} -> {board} "
                    f"({', '.join(parts)})"
                )

                def _on_row(row):
                    with _mvbcm_lock:
                        _mvbcm_state['rows'].append(row)

                if run_analog:
                    rows = run_groups(
                        enabled_groups(),
                        load_steps(),
                        board=board,
                        device=device_name,
                        settle=3.0,
                        timeout=5.0,
                        on_row=_on_row,
                        should_stop=_should_stop,
                    )

                if run_ct and not _should_stop():
                    from nidaq_controller.mvbcm.stimuli.voltage import find_9264
                    ct_device = find_9264()
                    add_log(
                        f"MV BCM CT (MTR-03 peak + AI-02 RMS, 11 each) "
                        f"on {ct_device} ao0"
                    )
                    _stop_module_ao(di, mi)
                    _stop_modules_by_product('9264')
                    ct_rows = run_ct_peak_test(
                        board,
                        device=ct_device,
                        settle=3.0,
                        timeout=5.0,
                        on_row=_on_row,
                        should_stop=_should_stop,
                        include_peak=True,
                        include_rms=True,
                    )
                    rows.extend(ct_rows)

                if (run_iph_h or run_imtr_h) and not _should_stop():
                    from nidaq_controller.mvbcm.stimuli.voltage import find_9264
                    harm_device = find_9264()
                    hn = '+'.join(
                        f'H{n} {pct:g}%' for n, pct in harmonic_map.items()
                    )
                    harm_targets = []
                    if run_iph_h:
                        harm_targets.append("iph")
                    if run_imtr_h:
                        harm_targets.append("imtr")
                    if harm_targets:
                        add_log(
                            f"MV BCM harmonic {'+'.join(harm_targets)} {hn} "
                            f"of live peak_eng on {harm_device} ao0"
                        )
                        _stop_module_ao(di, mi)
                        _stop_modules_by_product('9264')
                        harm_rows = run_harmonic_test(
                            board,
                            targets=tuple(harm_targets),
                            device=harm_device,
                            settle=3.0,
                            timeout=5.0,
                            harmonics=harmonic_map,
                            on_row=_on_row,
                            should_stop=_should_stop,
                        )
                        rows.extend(harm_rows)

                if run_di and not _should_stop():
                    channels = default_trip_channels()
                    prefer = (
                        trip_device
                        or device_name_from_channels(channels or {})
                        or "Dev9"
                    )
                    trip_di, trip_mi, trip_mod = _find_trip_dio_indices(prefer)
                    dio = get_dio_controller(trip_di, trip_mi, trip_mod)
                    add_log(
                        f"MV BCM DI check USB2 wet Ch1/2/3/4/6 on "
                        f"{trip_mod.name} ({di_cycles} cycle(s), "
                        f"settle {DI_SETTLE_S:.0f}s)"
                    )
                    di_rows = run_di_check_test(
                        board,
                        device=trip_mod.name,
                        module=trip_mod,
                        channels=channels,
                        settle=DI_SETTLE_S,
                        timeout=5.0,
                        cycles=di_cycles,
                        on_row=_on_row,
                        controller_factory=lambda _m: dio,
                        cleanup=False,
                        should_stop=_should_stop,
                    )
                    rows.extend(di_rows)

                if run_fault and not _should_stop():
                    channels = default_trip_channels()
                    prefer = (
                        trip_device
                        or device_name_from_channels(channels or {})
                        or "Dev9"
                    )
                    trip_di, trip_mi, trip_mod = _find_trip_dio_indices(prefer)
                    dio = get_dio_controller(trip_di, trip_mi, trip_mod)
                    add_log(
                        f"MV BCM fault trip/close on {trip_mod.name} "
                        f"({fault_cycles} cycle(s), counter wait "
                        f"{FAULT_EVENT_WAIT_S:.0f}s)"
                    )
                    fault_rows = run_fault_trip_close_test(
                        board,
                        device=trip_mod.name,
                        module=trip_mod,
                        channels=channels,
                        timeout=30.0,
                        cycles=fault_cycles,
                        on_row=_on_row,
                        controller_factory=lambda _m: dio,
                        cleanup=False,
                        should_stop=_should_stop,
                    )
                    rows.extend(fault_rows)

                if trip_close and not _should_stop():
                    # Same path as Apply trip: shared get_dio_controller +
                    # apply_breaker_operation on the USB wet-DO module.
                    channels = default_trip_channels()
                    prefer = (
                        trip_device
                        or device_name_from_channels(channels or {})
                        or "Dev9"
                    )
                    trip_di, trip_mi, trip_mod = _find_trip_dio_indices(prefer)
                    dio = get_dio_controller(trip_di, trip_mi, trip_mod)
                    add_log(
                        f"MV BCM trip/close via Apply-trip path on "
                        f"{trip_mod.name} ({trip_close_cycles} cycle(s))"
                    )
                    add_log(
                        f"MV BCM event wait {EVENT_SETTLE_S:.0f}s after trip/close"
                    )
                    breaker_rows = run_trip_close_test(
                        board,
                        device=trip_mod.name,
                        module=trip_mod,
                        channels=channels,
                        settle=EVENT_SETTLE_S,
                        timeout=5.0,
                        cycles=trip_close_cycles,
                        on_row=_on_row,
                        controller_factory=lambda _m: dio,
                        cleanup=False,
                        should_stop=_should_stop,
                    )
                    rows.extend(breaker_rows)

                if _should_stop():
                    stopped = True
                    raise MvbcmStopped('stopped by user')

                append_results(DEFAULT_RESULTS, rows)
                passed, failed = _mvbcm_counts(rows)
                add_log(f"MV BCM run finished: {passed} pass, {failed} fail")
            except Exception as exc:
                from nidaq_controller.mvbcm.breaker_ops import MvbcmStopped as _Stopped
                stopped = isinstance(exc, _Stopped) or _should_stop()
                if stopped:
                    try:
                        _stop_module_ao(di, mi)
                        _stop_modules_by_product('9264')
                    except Exception:
                        pass
                    with _mvbcm_lock:
                        kept = list(_mvbcm_state['rows'])
                        _mvbcm_state['error'] = 'stopped by user'
                        _mvbcm_state['stopped'] = True
                    add_log(f"MV BCM run stopped ({len(kept)} result(s) kept)")
                    if kept:
                        try:
                            from nidaq_controller.mvbcm.run_mvbcm import (
                                append_results,
                                DEFAULT_RESULTS,
                            )
                            append_results(DEFAULT_RESULTS, kept)
                        except Exception:
                            pass
                else:
                    add_log(f"MV BCM run failed: {exc}", 'ERROR')
                    with _mvbcm_lock:
                        _mvbcm_state['error'] = str(exc)
            finally:
                with _mvbcm_lock:
                    _mvbcm_state['running'] = False
                    _mvbcm_state['done'] = True
                    if stopped:
                        _mvbcm_state['stopped'] = True

        threading.Thread(target=_worker, name='mvbcm-run', daemon=True).start()
        return jsonify({
            'started': True,
            'board': board,
            'device': device_name,
            'analog': run_analog,
            'ct': run_ct,
            'harmonic': bool(run_iph_h or run_imtr_h),
            'harmonic_h3_pct': harmonic_map.get(3, 0),
            'harmonics': [
                {'order': n, 'pct': pct} for n, pct in harmonic_map.items()
            ],
            'di_check': run_di,
            'di_cycles': di_cycles if run_di else 0,
            'fault_trip_close': run_fault,
            'fault_cycles': fault_cycles if run_fault else 0,
            'trip_close': trip_close,
            'trip_close_cycles': trip_close_cycles if trip_close else 0,
        })
    except Exception as e:
        with _mvbcm_lock:
            if _mvbcm_state['running'] and not _mvbcm_state['rows']:
                _mvbcm_state['running'] = False
                _mvbcm_state['done'] = True
                _mvbcm_state['error'] = str(e)
        return _api_error('MV BCM run', e, di=di, mi=mi)


def _launch_mvbcm_event(di, mi, board, device_name, label, runner_name):
    """Background runner for Fault current and Close event. Not the main Run path."""
    def _should_stop() -> bool:
        return _mvbcm_stop.is_set()

    def _worker():
        rows = []
        stopped = False
        try:
            repo_root = Path(__file__).resolve().parents[2]
            if str(repo_root) not in sys.path:
                sys.path.append(str(repo_root))
            from nidaq_controller.mvbcm.breaker_ops import (
                MvbcmStopped,
                default_trip_channels,
                device_name_from_channels,
            )
            from nidaq_controller.mvbcm.run_mvbcm import DEFAULT_RESULTS, append_results
            from nidaq_controller.mvbcm.stimuli.voltage import find_9264
            if runner_name == 'close':
                from nidaq_controller.mvbcm.close_event import run_close_event as run
            elif runner_name == 'emc':
                from nidaq_controller.mvbcm.emc import run_emc as run
            else:
                from nidaq_controller.mvbcm.fault_current import run_fault_current as run

            _stop_module_ao(di, mi)
            _stop_modules_by_product('9264')
            voltage_device = find_9264()
            channels = default_trip_channels()
            prefer = device_name_from_channels(channels or {}) or "Dev9"
            trip_di, trip_mi, trip_mod = _find_trip_dio_indices(prefer)
            dio = get_dio_controller(trip_di, trip_mi, trip_mod)
            add_log(f"MV BCM {label} on {device_name} -> {board}")

            def _on_row(row):
                with _mvbcm_lock:
                    _mvbcm_state['rows'].append(row)

            rows = run(
                board,
                voltage_device=voltage_device,
                current_device=device_name,
                channels=channels,
                module=trip_mod,
                controller_factory=lambda _m: dio,
                timeout=5.0,
                on_row=_on_row,
                should_stop=_should_stop,
            )
            if _should_stop():
                stopped = True
                raise MvbcmStopped('stopped by user')
            append_results(DEFAULT_RESULTS, rows)
            passed, failed = _mvbcm_counts(rows)
            add_log(f"MV BCM {label} finished: {passed} pass, {failed} fail")
        except Exception as exc:
            from nidaq_controller.mvbcm.breaker_ops import MvbcmStopped as _Stopped
            stopped = isinstance(exc, _Stopped) or _should_stop()
            if stopped:
                try:
                    _stop_module_ao(di, mi)
                    _stop_modules_by_product('9264')
                except Exception:
                    pass
                with _mvbcm_lock:
                    kept = list(_mvbcm_state['rows'])
                    _mvbcm_state['error'] = 'stopped by user'
                    _mvbcm_state['stopped'] = True
                add_log(f"MV BCM {label} stopped ({len(kept)} result(s) kept)")
                if kept:
                    try:
                        from nidaq_controller.mvbcm.run_mvbcm import DEFAULT_RESULTS, append_results
                        append_results(DEFAULT_RESULTS, kept)
                    except Exception:
                        pass
            else:
                add_log(f"MV BCM {label} failed: {exc}", 'ERROR')
                with _mvbcm_lock:
                    _mvbcm_state['error'] = str(exc)
        finally:
            with _mvbcm_lock:
                _mvbcm_state['running'] = False
                _mvbcm_state['done'] = True
                if stopped:
                    _mvbcm_state['stopped'] = True

    threading.Thread(target=_worker, name=f'mvbcm-{label}', daemon=True).start()


def _begin_mvbcm_event(data):
    di = mi = None
    di, mi, err = _require_indices(data)
    if err:
        return None, err
    try:
        board = _board_url(data.get('board', ''))
    except ValueError as exc:
        return None, (jsonify({'error': str(exc)}), 400)
    _device, module, error = get_controller(di, mi)
    if error:
        return None, (jsonify({'error': error}), 400)
    if '9266' not in (module.product_type or '').upper():
        return None, (jsonify({'error': 'This test runs on an NI-9266'}), 400)
    with _mvbcm_lock:
        if _mvbcm_state['running']:
            return None, (jsonify({'error': 'A test is already running'}), 409)
        _mvbcm_stop.clear()
        _mvbcm_state.update({
            'running': True,
            'done': False,
            'stopped': False,
            'error': '',
            'rows': [],
            'passed': 0,
            'failed': 0,
            'device_idx': di,
            'module_idx': mi,
            'board': board,
            'device': module.name,
        })
    return (di, mi, board, module.name), None


@app.route('/api/mvbcm/fault-current', methods=['POST'])
def api_mvbcm_fault_current():
    """API: Fault-current trip. Scores the stored event, not the main Run path."""
    di = mi = None
    try:
        data, err = _parse_json()
        if err:
            return err
        started, err = _begin_mvbcm_event(data)
        if err:
            return err
        di, mi, board, device_name = started
        _launch_mvbcm_event(di, mi, board, device_name, 'fault current', 'fault')
        return jsonify({'started': True, 'fault_current': True, 'board': board, 'device': device_name})
    except Exception as e:
        with _mvbcm_lock:
            if _mvbcm_state['running'] and not _mvbcm_state['rows']:
                _mvbcm_state['running'] = False
                _mvbcm_state['done'] = True
                _mvbcm_state['error'] = str(e)
        return _api_error('MV BCM fault current', e, di=di, mi=mi)


@app.route('/api/mvbcm/close-event', methods=['POST'])
def api_mvbcm_close_event():
    """API: Close event. Scores rms_eng and the close event row."""
    di = mi = None
    try:
        data, err = _parse_json()
        if err:
            return err
        started, err = _begin_mvbcm_event(data)
        if err:
            return err
        di, mi, board, device_name = started
        _launch_mvbcm_event(di, mi, board, device_name, 'close event', 'close')
        return jsonify({'started': True, 'close_event': True, 'board': board, 'device': device_name})
    except Exception as e:
        with _mvbcm_lock:
            if _mvbcm_state['running'] and not _mvbcm_state['rows']:
                _mvbcm_state['running'] = False
                _mvbcm_state['done'] = True
                _mvbcm_state['error'] = str(e)
        return _api_error('MV BCM close event', e, di=di, mi=mi)


@app.route('/api/mvbcm/emc', methods=['POST'])
def api_mvbcm_emc():
    """API: EMC hold. RMS every 5 s, event and counters after each command."""
    di = mi = None
    try:
        data, err = _parse_json()
        if err:
            return err
        started, err = _begin_mvbcm_event(data)
        if err:
            return err
        di, mi, board, device_name = started
        _launch_mvbcm_event(di, mi, board, device_name, 'emc', 'emc')
        return jsonify({'started': True, 'emc': True, 'board': board, 'device': device_name})
    except Exception as e:
        with _mvbcm_lock:
            if _mvbcm_state['running'] and not _mvbcm_state['rows']:
                _mvbcm_state['running'] = False
                _mvbcm_state['done'] = True
                _mvbcm_state['error'] = str(e)
        return _api_error('MV BCM emc', e, di=di, mi=mi)


@app.route('/api/do/write', methods=['POST'])
def api_do_write():
    """API: Set digital output to true or false."""
    di = mi = channel = None
    try:
        data, err = _parse_json()
        if err:
            return err
        di, mi, err = _require_indices(data)
        if err:
            return err
        channel = data.get('channel')
        if not channel:
            return jsonify({'error': 'channel is required'}), 400
        state = bool(data.get('state', False))

        device, module, error = get_controller(di, mi)
        if error:
            return jsonify({'error': error}), 400

        controller = get_dio_controller(di, mi, module)
        if controller.set_output_line(channel, state):
            add_log(f"DO write: {channel} -> {state}")
            return jsonify({'state': state, 'success': True})

        failed = controller.get_last_failed_ports()
        err = _ni_error_detail(
            'Digital write failed. In NI MAX, set the port direction to '
            'Output for lines you want to drive.'
        )
        if failed:
            err += f' Failed port(s): {failed}.'
        add_log(f"DO write failed: {channel} — {err}", 'ERROR')
        return jsonify({'error': err, 'failed_ports': failed}), 500
    except Exception as e:
        return _api_error('DO write', e, di=di, mi=mi, channel=channel)


@app.route('/api/do/read', methods=['POST'])
def api_do_read():
    """API: Read one digital output / relay line from hardware."""
    di = mi = channel = None
    try:
        data, err = _parse_json()
        if err:
            return err
        di, mi, err = _require_indices(data)
        if err:
            return err
        channel = data.get('channel')
        if not channel:
            return jsonify({'error': 'channel is required'}), 400

        device, module, error = get_controller(di, mi)
        if error:
            return jsonify({'error': error}), 400

        controller = get_dio_controller(di, mi, module)
        state = controller.read_output_line(channel)
        if state is None:
            err_msg = _ni_error_detail('Digital output read failed')
            add_log(f"DO read failed: {channel} — {err_msg}", 'ERROR')
            return jsonify({'error': err_msg}), 500

        add_log(f"DO read: {channel} -> {state}")
        return jsonify({'state': state, 'success': True})
    except Exception as e:
        return _api_error('DO read', e, di=di, mi=mi, channel=channel)


@app.route('/api/di/read', methods=['POST'])
def api_di_read():
    """API: Read one digital input line."""
    di = mi = channel = None
    try:
        data, err = _parse_json()
        if err:
            return err
        di, mi, err = _require_indices(data)
        if err:
            return err
        channel = data.get('channel')
        if not channel:
            return jsonify({'error': 'channel is required'}), 400

        device, module, error = get_controller(di, mi)
        if error:
            return jsonify({'error': error}), 400

        controller = get_dio_controller(di, mi, module)
        values = controller.read_digital_input([channel])
        if not values or channel not in values:
            err_msg = _ni_error_detail('Digital read failed')
            add_log(f"DI read failed: {channel} — {err_msg}", 'ERROR')
            return jsonify({'error': err_msg}), 500

        state = bool(values[channel])
        add_log(f"DI read: {channel} -> {state}")
        return jsonify({'state': state, 'success': True})
    except Exception as e:
        return _api_error('DI read', e, di=di, mi=mi, channel=channel)


@app.route('/api/di/trip-waveform', methods=['POST'])
def api_di_trip_waveform():
    """API: Expected or captured 4-channel wet DI trip waveform."""
    di = mi = None
    try:
        data, err = _parse_json()
        if err:
            return err
        mode = (data.get('mode') or 'preview').strip().lower()
        operation = (data.get('operation') or 'trip').strip().lower()
        if operation not in ('trip', 'close'):
            return jsonify({'error': 'operation must be trip or close'}), 400
        if mode == 'apply':
            di, mi, err = _require_indices(data)
            if err:
                return err
            channels = data.get('channels') or {}
            if not isinstance(channels, dict):
                return jsonify({'error': 'channels must be an object'}), 400
            device, module, error = get_controller(di, mi)
            if error:
                return jsonify({'error': error}), 400
            controller = get_dio_controller(di, mi, module)
            debounce = bool(data.get('debounce', False))
            payload = controller.apply_breaker_operation(
                channels, operation, debounce=debounce,
            )
            used = payload.get('channels_used') or {}
            debounce_note = " with 1 ms debounce ×2" if debounce else ""
            add_log(
                f"{operation.upper()} APPLIED{debounce_note} on outputs: "
                + ", ".join(f"{key}={line}" for key, line in used.items())
            )
            payload['applied'] = True
            return jsonify(payload)

        if mode == 'preview':
            payload = build_close_waveform() if operation == 'close' else build_trip_waveform()
            payload['applied'] = False
            add_log(
                f"{operation.capitalize()} plot only — outputs were NOT written"
            )
            return jsonify(payload)

        if mode != 'capture':
            return jsonify({'error': 'mode must be preview or capture'}), 400

        di, mi, err = _require_indices(data)
        if err:
            return err
        channels = data.get('channels') or {}
        if not isinstance(channels, dict):
            return jsonify({'error': 'channels must be an object'}), 400

        device, module, error = get_controller(di, mi)
        if error:
            return jsonify({'error': error}), 400

        arm_timeout_s = float(data.get('arm_timeout_s', 20.0))
        controller = get_dio_controller(di, mi, module)
        if operation == 'close':
            payload = controller.capture_close_waveform(channels, arm_timeout_s=arm_timeout_s)
        else:
            payload = controller.capture_trip_waveform(channels, arm_timeout_s=arm_timeout_s)
        timing = payload.get('timing') or {}
        add_log(
            f"{operation.capitalize()} waveform capture: "
            f"pulse={timing.get('pulse_ms')} ms, "
            f"travel={timing.get('travel_ms')} ms, "
            f"rate={payload.get('sample_rate_hz')} Hz"
        )
        return jsonify(payload)
    except (ValueError, TimeoutError, RuntimeError) as e:
        add_log(f"Trip waveform failed: {e}", 'ERROR')
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return _api_error('Trip waveform', e, di=di, mi=mi)


def _operation_loop_controller():
    """Resolve the DI controller for a loop start/stop/status call."""
    if request.method == 'GET':
        data = request.args
    else:
        data, err = _parse_json()
        if err:
            return None, None, None, err
    if 'device_idx' not in data or 'module_idx' not in data:
        return None, None, None, (jsonify({'error': 'device_idx and module_idx are required'}), 400)
    try:
        di = int(data['device_idx'])
        mi = int(data['module_idx'])
    except (TypeError, ValueError):
        return None, None, None, (jsonify({'error': 'device_idx and module_idx must be integers'}), 400)
    device, module, error = get_controller(di, mi)
    if error:
        return None, None, None, (jsonify({'error': error}), 400)
    return di, mi, get_dio_controller(di, mi, module), data if request.method != 'GET' else None


@app.route('/api/di/operation-loop', methods=['GET'])
def api_operation_loop_status():
    """API: Latest trip/close records and loop phase."""
    try:
        di, mi, controller, err = _operation_loop_controller()
        if err:
            return err
        return jsonify(controller.operation_loop_status())
    except Exception as e:
        return _api_error('Operation loop status', e)


@app.route('/api/di/operation-loop/start', methods=['POST'])
def api_operation_loop_start():
    """API: Loop trip then close, ``interval_s`` apart (default OPERATION_INTERVAL_S)."""
    di = mi = None
    try:
        di, mi, controller, data = _operation_loop_controller()
        if controller is None:
            return data
        channels = (data or {}).get('channels') or {}
        if not isinstance(channels, dict):
            return jsonify({'error': 'channels must be an object'}), 400
        interval_s = float((data or {}).get('interval_s', OPERATION_INTERVAL_S))
        if interval_s < 0:
            return jsonify({'error': 'interval_s must be >= 0'}), 400
        debounce_trip = bool((data or {}).get('debounce_trip', False))
        debounce_close = bool((data or {}).get('debounce_close', False))
        status = controller.start_operation_loop(
            channels,
            interval_s=interval_s,
            debounce_trip=debounce_trip,
            debounce_close=debounce_close,
        )
        extra = []
        if debounce_trip:
            extra.append('debounce trip')
        if debounce_close:
            extra.append('debounce close')
        extra_s = f" ({', '.join(extra)})" if extra else ""
        add_log(f"Trip/close loop started, {interval_s:.0f}s apart{extra_s}")
        return jsonify(status)
    except (ValueError, RuntimeError) as e:
        add_log(f"Trip/close loop failed: {e}", 'ERROR')
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return _api_error('Operation loop start', e, di=di, mi=mi)


@app.route('/api/di/operation-loop/stop', methods=['POST'])
def api_operation_loop_stop():
    """API: Stop the trip/close loop."""
    try:
        di, mi, controller, data = _operation_loop_controller()
        if controller is None:
            return data
        status = controller.stop_operation_loop()
        add_log("Trip/close loop stopped")
        return jsonify(status)
    except Exception as e:
        return _api_error('Operation loop stop', e)


@app.route('/api/do/write-module', methods=['POST'])
def api_do_write_module():
    """API: Write all digital lines on one module to the same state."""
    di = mi = None
    try:
        data, err = _parse_json()
        if err:
            return err
        di, mi, err = _require_indices(data)
        if err:
            return err

        device, module, error = get_controller(di, mi)
        if error:
            return jsonify({'error': error}), 400

        states = data.get('states')
        if data.get('uniform_state') is not None:
            uniform = bool(data['uniform_state'])
            states = {ch: uniform for ch in (module.do_channels or [])}

        if not states:
            return jsonify({'error': 'No channels to write'}), 400

        controller = get_dio_controller(di, mi, module)
        if controller.write_digital_output(states):
            display = controller.get_all_output_states()
            failed_ports = controller.get_last_failed_ports()
            add_log(f"DO write-all: {len(states)} channel(s) on {module.name}")
            payload = {
                'states': {ch: bool(v) for ch, v in display.items()},
                'success': True,
                'failed_ports': failed_ports,
            }
            if failed_ports:
                payload['warning'] = (
                    f"Port(s) {failed_ports} not written — set to Output in NI MAX"
                )
            return jsonify(payload)

        failed_ports = controller.get_last_failed_ports()
        err_msg = _ni_error_detail(
            'Digital write-all failed. In NI MAX, set port direction to '
            'Output for lines you want to drive.'
        )
        add_log(f"DO write-all failed on {module.name}: {err_msg}", 'ERROR')
        return jsonify({
            'error': err_msg,
            'failed_ports': failed_ports,
        }), 500
    except Exception as e:
        return _api_error('DO write-module', e, di=di, mi=mi)


@app.route('/api/do/toggle', methods=['POST'])
def api_do_toggle():
    """API: Toggle digital output."""
    di = mi = channel = None
    try:
        data, err = _parse_json()
        if err:
            return err
        di, mi, err = _require_indices(data)
        if err:
            return err
        channel = data.get('channel')
        if not channel:
            return jsonify({'error': 'channel is required'}), 400
        
        device, module, error = get_controller(di, mi)
        if error:
            return jsonify({'error': error}), 400
        
        controller = get_dio_controller(di, mi, module)
        state = controller.toggle_output(channel)
        
        if state is not None:
            add_log(f"DO toggle: {channel} -> {'HIGH' if state else 'LOW'}")
            return jsonify({'state': state, 'success': True})
        
        err_msg = _ni_error_detail('Toggle failed')
        add_log(f"DO toggle failed: {channel} — {err_msg}", 'ERROR')
        return jsonify({'error': err_msg}), 500
    except Exception as e:
        return _api_error('DO toggle', e, di=di, mi=mi, channel=channel)


@app.route('/api/do/test-line', methods=['POST'])
def api_do_test_line():
    """API: Test one digital line (True then False)."""
    di = mi = None
    try:
        data, err = _parse_json()
        if err:
            return err
        di, mi, err = _require_indices(data)
        if err:
            return err
        port = int(data['port'])
        line = int(data['line'])

        device, module, error = get_controller(di, mi)
        if error:
            return jsonify({'error': error}), 400

        if not module.do_channels:
            return jsonify({'error': 'No digital output channels'}), 400

        controller = get_dio_controller(di, mi, module)
        result = controller.test_one_line(port, line)
        add_log(f"Line test P{port}.{line} on {module.name}: {'OK' if result.get('ok') else 'FAIL'}")
        return jsonify({'success': True, **result})
    except (TypeError, ValueError, KeyError) as e:
        return jsonify({'error': f'Invalid request: {e}'}), 400
    except Exception as e:
        return _api_error('DO test-line', e, di=di, mi=mi)


@app.route('/api/do/test-ports', methods=['POST'])
def api_do_test_ports():
    """API: Test each digital line sequentially (batch; use test-line for live UI)."""
    di = mi = None
    try:
        data, err = _parse_json()
        if err:
            return err
        di, mi, err = _require_indices(data)
        if err:
            return err

        device, module, error = get_controller(di, mi)
        if error:
            return jsonify({'error': error}), 400

        if not module.do_channels:
            return jsonify({'error': 'No digital output channels'}), 400

        controller = get_dio_controller(di, mi, module)
        line_results = controller.test_lines()
        port_results = {}
        by_port = {}
        for key, info in line_results.items():
            pn = info.get('port', -1)
            if not info.get('found', True):
                continue
            by_port.setdefault(pn, []).append(bool(info.get('ok')))
        for pn, oks in by_port.items():
            port_results[pn] = {
                'ok': all(oks) if oks else False,
                'lines': len(oks),
                'lines_ok': sum(1 for x in oks if x),
                'message': 'OK' if all(oks) else 'One or more lines failed',
            }
        ok_ports = sum(1 for v in port_results.values() if v.get('ok'))
        ok_lines = sum(1 for v in line_results.values() if v.get('ok'))
        add_log(f"Line test on {module.name}: {ok_lines}/{len(line_results)} OK")
        return jsonify({
            'success': True,
            'ports': port_results,
            'lines': line_results,
            'summary': f'{ok_lines}/{len(line_results)} lines OK, {ok_ports}/{len(port_results)} ports OK',
        })
    except Exception as e:
        return _api_error('DO test-ports', e, di=di, mi=mi)


def _live_check_snapshot() -> dict:
    with _live_check_lock:
        rows = list(_live_check_state['rows'])
        passed, failed = _mvbcm_counts(rows)
        return {
            'running': _live_check_state['running'],
            'done': _live_check_state['done'],
            'stopped': _live_check_state['stopped'],
            'error': _live_check_state['error'],
            'rows': rows,
            'passed': passed,
            'failed': failed,
            'board': _live_check_state['board'],
        }


@app.route('/api/live-check/status')
def api_live_check_status():
    """API: Live-check rows scored so far."""
    return jsonify(_live_check_snapshot())


@app.route('/api/live-check/stop', methods=['POST'])
def api_live_check_stop():
    """API: Stop the live-check poll and append rows to the MV BCM CSV."""
    with _live_check_lock:
        running = _live_check_state['running']
    if not running:
        return jsonify({'stopped': False, 'message': 'No live check running'})
    _live_check_stop.set()
    add_log('Live check stop requested')
    return jsonify({'stopped': True})


@app.route('/api/live-check/start', methods=['POST'])
def api_live_check_start():
    """API: Poll GET /api/live and score every analog and digital channel."""
    data, err = _parse_json()
    if err:
        return err
    try:
        board = _board_url(data.get('board', ''))
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    try:
        try:
            from nidaq_controller.mvbcm.live_check import parse_inputs
        except ModuleNotFoundError:
            from mvbcm.live_check import parse_inputs
        analog, digital = parse_inputs(data.get('analog'), data.get('digital'))
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    with _live_check_lock:
        if _live_check_state['running']:
            return jsonify({'error': 'A live check is already running'}), 409
        _live_check_stop.clear()
        _live_check_state.update({
            'running': True,
            'done': False,
            'stopped': False,
            'error': '',
            'rows': [],
            'board': board,
        })

    def _worker():
        try:
            from nidaq_controller.mvbcm.live import LiveError
            from nidaq_controller.mvbcm.live_check import (
                DEFAULT_RESULTS,
                append_results,
                fetch_and_score,
                live_error_row,
            )
        except ModuleNotFoundError:
            from mvbcm.live import LiveError
            from mvbcm.live_check import (
                DEFAULT_RESULTS,
                append_results,
                fetch_and_score,
                live_error_row,
            )

        step = 0
        try:
            while not _live_check_stop.is_set():
                step += 1
                try:
                    batch = fetch_and_score(board, analog, digital, step)
                except LiveError as exc:
                    batch = [live_error_row(step, str(exc))]
                except Exception as exc:
                    add_log(f'Live check failed: {exc}', 'ERROR')
                    with _live_check_lock:
                        _live_check_state['error'] = str(exc)
                    break
                with _live_check_lock:
                    _live_check_state['rows'].extend(batch)
                if _live_check_stop.wait(5.0):
                    break
        finally:
            with _live_check_lock:
                kept = list(_live_check_state['rows'])
                _live_check_state['running'] = False
                _live_check_state['done'] = True
                _live_check_state['stopped'] = _live_check_stop.is_set()
            if kept:
                try:
                    with _mvbcm_lock:
                        append_results(DEFAULT_RESULTS, kept)
                except Exception as exc:
                    add_log(f'Live check CSV save failed: {exc}', 'ERROR')
                    with _live_check_lock:
                        if not _live_check_state['error']:
                            _live_check_state['error'] = f'CSV save failed: {exc}'
            passed, failed = _mvbcm_counts(kept)
            add_log(f'Live check finished: {passed} pass, {failed} fail')

    add_log(f'Live check started -> {board}')
    threading.Thread(target=_worker, name='live-check', daemon=True).start()
    return jsonify({'started': True, 'board': board})


@app.route('/api/log')
def api_log():
    """API: Get recent log entries."""
    return jsonify({'log': _log_entries[-50:]})


@app.route('/api/status')
def api_status():
    """API: Get system status summary."""
    devices = device_manager.get_all_devices()
    connected = device_manager.get_connected_devices()
    active_tasks = task_manager.get_active_tasks()
    
    return jsonify({
        'total_devices': len(devices),
        'connected_devices': len(connected),
        'active_tasks': len(active_tasks),
        'active_ai': len(_active_ai_tasks),
        'active_ao': len(_active_ao_tasks),
        'uptime': time.time() - _start_time
    })


# ===================== Main Entry Point =====================

_start_time = time.time()


def main() -> None:
    """Start the ATS Test System web server."""
    import logging
    # Flask's built-in server is for local/dev use only — suppress the banner noise.
    logging.getLogger('werkzeug').setLevel(logging.ERROR)

    print()
    print("=" * 60)
    print("  ATS Test System - Web Interface")
    print("=" * 60)
    print()
    print("  Starting server...")
    print()

    add_log("Starting ATS Test System web server...")
    devices = device_manager.discover_devices()
    add_log(f"Detected {len(devices)} device(s)")

    print(f"  OK Device discovery: {len(devices)} device(s)")
    print("  OK Web server running on http://localhost:5000")
    print()
    print("  Open in your browser:  http://localhost:5000")
    print()
    print("  Press Ctrl+C to stop")
    print("=" * 60)
    print()

    try:
        app.run(
            host=DEFAULT_HOST,
            port=DEFAULT_PORT,
            debug=False,
            use_reloader=False,
        )
    except Exception as e:
        log.error("Server error: %s", e)
        print(f"\n  ERROR: {e}\n")


if __name__ == '__main__':
    main()
else:
    # When imported, do initial discovery
    add_log("ATS Test System module loaded")
    try:
        device_manager.discover_devices()
    except Exception as e:
        log.warning("Initial discovery failed: %s", e)
