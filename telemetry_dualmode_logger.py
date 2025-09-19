#!/usr/bin/env python3
"""
Telemetry Logger (Windows/Linux)
- Single Start/Stop toggle (green ↔ red)
- Profile Mode (1s) toggle button (for inference bursts)
- "View Log File" button opens today's active CSV
- Logs every 15s (or 1s in Profile Mode) to ./logs and rolls daily
- Multi-GPU via NVML (pynvml) with nvidia-smi fallback
- CPU/RAM/Disk/Net via psutil
- CPU package + motherboard temps via LibreHardwareMonitor (LHM) or WMI (Windows)
- Windows convenience: auto-copy nvml.dll into NVSMI if missing

Setup:
  pip install psutil nvidia-ml-py3
  # Optional (Windows sensors):
  pip install wmi pywin32

Optional (Windows): Start LibreHardwareMonitor → Options → Remote Web Server → Start
Override LHM endpoint with env var LHM_URL (default http://localhost:8085/data.json)

Run:
  python telemetry_dualmode_logger.py
"""

import csv
import os
import sys
import time
import datetime
import subprocess
import shutil
import ctypes
import platform
import threading
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import re

# ---------- Config ----------
DEFAULT_INTERVAL_S = 15.0
PROFILE_INTERVAL_S = 1.0
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# LibreHardwareMonitor JSON endpoint (HTTP)
LHM_URL = os.environ.get("LHM_URL", "http://localhost:8085/data.json").strip()

# Optional imports
try:
    import psutil
except ImportError:
    print("Missing dependency: psutil. Install with: pip install psutil", file=sys.stderr)
    raise

# NVIDIA NVML (pynvml)
try:
    import pynvml
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

# Windows WMI (optional for motherboard temp)
try:
    import wmi as _wmi
    WMI_AVAILABLE = True
except Exception:
    WMI_AVAILABLE = False

# ---------- NVML pre-load helper (Windows) ----------
def _ensure_nvml_loaded():
    """On Windows, try to explicitly load nvml.dll from known locations."""
    if platform.system().lower() != "windows":
        return
    candidates = []
    env_path = os.environ.get("NVML_DLL_PATH", "").strip()
    if env_path:
        candidates.append(env_path)
    candidates.append(r"C:\Windows\System32\nvml.dll")
    for cand in candidates:
        try:
            dll_dir = os.path.dirname(cand)
            if hasattr(os, "add_dll_directory") and os.path.isdir(dll_dir):
                os.add_dll_directory(dll_dir)
            if os.path.isfile(cand):
                ctypes.WinDLL(cand)
                return
        except Exception:
            continue

# ---------- Windows NVML helper: ensure nvml.dll exists in NVSMI ----------
def _nvml_src_candidates_windows():
    cands = []
    envp = os.environ.get("NVML_DLL_PATH", "").strip()
    if envp:
        cands.append(envp)
    windir = os.environ.get("WINDIR", r"C:\Windows")
    cands.append(os.path.join(windir, "System32", "nvml.dll"))
    cands.append(os.path.join(windir, "SysWOW64", "nvml.dll"))  # rare 32-bit Python
    return [c for c in cands if c and os.path.isfile(c)]

def _ensure_nvml_in_nvsm_windows(show_message: bool = False):
    """Try to copy nvml.dll into NVSMI so pynvml can find it. Returns (ok: bool, msg: str)."""
    if platform.system().lower() != "windows":
        return (False, "Not Windows")
    dest_dir = r"C:\Program Files\NVIDIA Corporation\NVSMI"
    dest = os.path.join(dest_dir, "nvml.dll")
    if os.path.isfile(dest):
        try:
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(dest_dir)
        except Exception:
            pass
        return (True, f"NVML present at {dest}")
    srcs = _nvml_src_candidates_windows()
    if not srcs:
        return (False, "No nvml.dll source discovered (check NVIDIA driver install)")
    src = srcs[0]
    try:
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(src, dest)
        try:
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(dest_dir)
        except Exception:
            pass
        return (True, f"Copied {src} -> {dest}")
    except PermissionError as e:
        msg = ("Permission denied copying nvml.dll to NVSMI.\n"
               "Run this app as Administrator once or copy manually.\n"
               f"Source: {src}\nDest: {dest}\nError: {e}")
        if show_message:
            try:
                from tkinter import messagebox
                messagebox.showwarning("NVML copy requires elevation", msg)
            except Exception:
                pass
        return (False, msg)
    except Exception as e:
        msg = f"Failed to copy nvml.dll: {e}"
        if show_message:
            try:
                from tkinter import messagebox
                messagebox.showwarning("NVML copy failed", msg)
            except Exception:
                pass
        return (False, msg)

# ---------- Helpers ----------
def now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")

@dataclass
class RateTracker:
    last_time: Optional[float] = None
    last_val: Optional[int] = None

    def rate_per_sec(self, current_val: int) -> Optional[float]:
        t = time.time()
        if self.last_time is None or self.last_val is None:
            self.last_time = t
            self.last_val = current_val
            return None
        dt = t - self.last_time
        dv = current_val - self.last_val
        self.last_time = t
        self.last_val = current_val
        if dt <= 0:
            return None
        return dv / dt

class CSVLogger:
    """Daily CSV writer with schema-aware rollover (creates ..._2.csv if header changes mid-day)."""
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.current_date: Optional[datetime.date] = None
        self.file = None
        self.writer = None
        self.header_written = False
        self.lock = threading.Lock()
        self.roll_index = 1
        self.active_header: Optional[List[str]] = None
        self.current_path: Optional[str] = None

    def _open_new_file(self, header: List[str]):
        if self.file:
            try:
                self.file.flush()
                self.file.close()
            except Exception:
                pass
        ts = datetime.datetime.now().strftime("%Y-%m-%d")
        base = os.path.join(self.base_dir, f"telemetry_{ts}.csv")
        fname = base
        if self.active_header is not None:
            self.roll_index += 1
            fname = os.path.join(self.base_dir, f"telemetry_{ts}_{self.roll_index}.csv")
        self.file = open(fname, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(header)
        self.header_written = True
        self.active_header = list(header)
        self.current_path = fname

    def write_row(self, header: List[str], row: List[Any]):
        with self.lock:
            today = datetime.date.today()
            if self.current_date != today or self.file is None:
                self.current_date = today
                self.active_header = None
                self.roll_index = 1
                self._open_new_file(header)
            if self.active_header is not None and set(header) != set(self.active_header):
                self._open_new_file(header)
            elif not self.header_written:
                self.writer.writerow(header)
                self.header_written = True
                self.active_header = list(header)
            self.writer.writerow(row)
            self.file.flush()

    def current_file_path(self) -> Optional[str]:
        """Return the path of the current CSV being written (if any)."""
        return self.current_path

class NvidiaSMIHelper:
    """Fallback GPU metrics via `nvidia-smi` when NVML is unavailable."""
    def __init__(self):
        self.available = shutil.which("nvidia-smi") is not None

    def query(self):
        if not self.available:
            return []
        fields = [
            "name","uuid","utilization.gpu","temperature.gpu","temperature.memory",
            "clocks.sm","clocks.mem","power.draw","memory.used","memory.total","fan.speed"
        ]
        cmd = ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"]
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=3.0)
        except Exception:
            return []
        gpus = []
        for line in out.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < len(fields):
                parts += [None] * (len(fields) - len(parts))
            try:
                name, uuid, util, t_gpu, t_mem, sm_clk, mem_clk, pwr, v_used, v_tot, fan = parts[:11]
                gpus.append({
                    "name": name or None,
                    "uuid": uuid or None,
                    "gpu_util_percent": float(util) if util not in (None, "", "N/A") else None,
                    "mem_util_percent": None,
                    "vram_used_GB": round(float(v_used) / 1024.0, 3) if v_used not in (None, "", "N/A") else None,
                    "vram_total_GB": round(float(v_tot) / 1024.0, 3) if v_tot not in (None, "", "N/A") else None,
                    "temp_gpu_c": float(t_gpu) if t_gpu not in (None, "", "N/A") else None,
                    "temp_mem_c": float(t_mem) if t_mem not in (None, "", "N/A") else None,
                    "power_w": float(pwr) if pwr not in (None, "", "N/A") else None,
                    "fan_percent": float(fan) if fan not in (None, "", "N/A") else None,
                    "sm_clock_mhz": int(float(sm_clk)) if sm_clk not in (None, "", "N/A") else None,
                    "mem_clock_mhz": int(float(mem_clk)) if mem_clk not in (None, "", "N/A") else None,
                    "temp_mem_supported": (t_mem not in (None, "", "N/A")),
                })
            except Exception:
                continue
        return gpus

# ---------- LibreHardwareMonitor (LHM) integration ----------
def _lhm_fetch_json(url: str, timeout: float = 0.8) -> Optional[dict]:
    if not url or not url.lower().startswith("http"):
        return None
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "telemetry-logger/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if getattr(resp, "status", 200) != 200:
                return None
            data = resp.read()
            return json.loads(data.decode("utf-8", "ignore"))
    except Exception:
        return None

def _lhm_flatten_sensors(node: dict, path: List[str], out: Dict[str, float]):
    name = str(node.get("Text") or node.get("text") or node.get("Name") or node.get("name") or "")
    if name:
        path = path + [name]
    val = node.get("Value", node.get("value", None))
    stype = str(node.get("SensorType") or node.get("sensortype") or "").lower()
    def _parse_value(v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        m = re.search(r'(-?\d+(?:\.\d+)?)', str(v))
        try:
            return float(m.group(1)) if m else None
        except Exception:
            return None
    pv = _parse_value(val)
    if pv is not None and (stype in ("temperature", "") or "temp" in name.lower()):
        out[" / ".join(path)] = float(pv)
    for key in ("Children", "children", "Nodes", "nodes", "Sensors", "sensors"):
        kids = node.get(key)
        if isinstance(kids, list):
            for ch in kids:
                if isinstance(ch, dict):
                    _lhm_flatten_sensors(ch, path, out)
    return out

def _lhm_read_temps() -> Dict[str, Optional[float]]:
    root = _lhm_fetch_json(LHM_URL)
    if not root:
        return {}
    flat = _lhm_flatten_sensors(root, [], {})
    cpu_pkg = None
    cpu_any = None
    mb_temp = None
    cpu_pkg_keywords = ("cpu package", "package", "tctl/tdie", "tctl", "tdie", "cpu die", "socket")
    for k, v in flat.items():
        kl = k.lower()
        if "cpu" in kl:
            if any(w in kl for w in cpu_pkg_keywords):
                cpu_pkg = v if cpu_pkg is None else max(cpu_pkg, v)
            if ("temp" in kl) or ("core" in kl) or ("cpu" in kl):
                cpu_any = v if cpu_any is None else max(cpu_any, v)
    for k, v in flat.items():
        kl = k.lower()
        if any(tag in kl for tag in ("mainboard", "motherboard", "system", "pch", "vrm", "/lpc/", "superio", "nct")):
            if 5.0 <= v <= 120.0:
                mb_temp = v if (mb_temp is None or v > mb_temp) else mb_temp
    out = {}
    if cpu_pkg is not None:
        out["cpu_pkg_temp_c"] = round(cpu_pkg, 1)
    if mb_temp is not None:
        out["mb_temp_c"] = round(mb_temp, 1)
    if cpu_any is not None:
        out["cpu_temp_c"] = round(cpu_any, 1)
    return out

# ---------- Platform-specific helpers ----------
def _motherboard_temp_windows_wmi():
    try:
        if platform.system().lower() != "windows" or not WMI_AVAILABLE:
            return None
        c = _wmi.WMI(namespace="root\\WMI")
        temps = []
        for tz in c.MSAcpi_ThermalZoneTemperature():
            raw = getattr(tz, "CurrentTemperature", None)  # tenths of Kelvin
            if raw is None:
                continue
            celsius = (float(raw) / 10.0) - 273.15
            if 0.0 < celsius < 125.0:
                temps.append(celsius)
        if temps:
            return max(temps)
    except Exception:
        return None
    return None

class TelemetrySampler:
    def __init__(self):
        self.net_rx_tracker = RateTracker()
        self.net_tx_tracker = RateTracker()
        self.disk_read_tracker = RateTracker()
        self.disk_write_tracker = RateTracker()
        self.gpu_count = 0
        self.using_fallback = False
        self.smi = NvidiaSMIHelper()

        if NVML_AVAILABLE and platform.system().lower() == "windows":
            try:
                _ensure_nvml_loaded()
            except Exception:
                pass

        if NVML_AVAILABLE:
            # Try to ensure nvml.dll is available where pynvml expects it (Windows)
            if platform.system().lower() == 'windows':
                _ensure_nvml_in_nvsm_windows(show_message=False)
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
            except Exception:
                try:
                    _ensure_nvml_loaded()
                    # Retry after explicit load; also attempt NVSMI copy once on Windows
                    if platform.system().lower() == 'windows':
                        ok,_ = _ensure_nvml_in_nvsm_windows(show_message=True)
                    pynvml.nvmlInit()
                    self.gpu_count = pynvml.nvmlDeviceGetCount()
                except Exception as e2:
                    print(f"NVML init failed: {e2}", file=sys.stderr)
                    self.using_fallback = self.smi.available
                    if self.using_fallback:
                        self.gpu_count = len(self.smi.query())
        else:
            self.using_fallback = self.smi.available
            if self.using_fallback:
                self.gpu_count = len(self.smi.query())

    def _cpu_metrics(self) -> Dict[str, Any]:
        vm = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_temp = None
        cpu_pkg = None
        try:
            temps = psutil.sensors_temperatures(fahrenheit=False) or {}
            for key in ["coretemp", "cpu-thermal", "acpitz"]:
                arr = temps.get(key)
                if arr:
                    cpu_temp = max([getattr(t, "current", None) for t in arr if hasattr(t, "current")], default=None)
                    break
            if cpu_temp is None and temps:
                first = next(iter(temps.values()))
                if first and len(first) > 0 and hasattr(first[0], "current"):
                    cpu_temp = first[0].current
            try:
                for arr in temps.values():
                    for s in arr:
                        label = (getattr(s, 'label', '') or getattr(s, 'sensor', '') or '').lower()
                        if ('package' in label) or ('tctl' in label) or ('socket' in label):
                            v = getattr(s, 'current', None)
                            if v is not None:
                                cpu_pkg = v if cpu_pkg is None else max(cpu_pkg, v)
            except Exception:
                cpu_pkg = None
        except Exception:
            cpu_temp = None
            cpu_pkg = None

        mb_temp = _motherboard_temp_windows_wmi()

        try:
            lhm = _lhm_read_temps()
        except Exception:
            lhm = {}
        if lhm:
            if cpu_pkg is None and "cpu_pkg_temp_c" in lhm:
                cpu_pkg = lhm["cpu_pkg_temp_c"]
            if mb_temp is None and "mb_temp_c" in lhm:
                mb_temp = lhm["mb_temp_c"]
            if cpu_temp is None and "cpu_temp_c" in lhm:
                cpu_temp = lhm["cpu_temp_c"]

        return {
            "cpu_percent": round(cpu_percent, 2) if cpu_percent is not None else None,
            "ram_used_gb": round((vm.total - vm.available) / (1024**3), 3),
            "ram_total_gb": round(vm.total / (1024**3), 1),
            "ram_percent": round(vm.percent, 2),
            "cpu_temp_c": round(cpu_temp, 1) if cpu_temp is not None else None,
            "cpu_pkg_temp_c": round(cpu_pkg, 1) if cpu_pkg is not None else None,
            "mb_temp_c": round(mb_temp, 1) if mb_temp is not None else None,
        }

    def _disk_net_metrics(self) -> Dict[str, Any]:
        dio = psutil.disk_io_counters(nowrap=True)
        d_read_bps = self.disk_read_tracker.rate_per_sec(dio.read_bytes) if dio else None
        d_write_bps = self.disk_write_tracker.rate_per_sec(dio.write_bytes) if dio else None
        nio = psutil.net_io_counters(nowrap=True)
        n_rx_bps = self.net_rx_tracker.rate_per_sec(nio.bytes_recv) if nio else None
        n_tx_bps = self.net_tx_tracker.rate_per_sec(nio.bytes_sent) if nio else None
        return {
            "disk_read_MBps": round((d_read_bps or 0) / (1024**2), 3) if d_read_bps is not None else None,
            "disk_write_MBps": round((d_write_bps or 0) / (1024**2), 3) if d_write_bps is not None else None,
            "net_rx_Mbps": round((n_rx_bps or 0) * 8 / (1024**2), 3) if n_rx_bps is not None else None,
            "net_tx_Mbps": round((n_tx_bps or 0) * 8 / (1024**2), 3) if n_tx_bps is not None else None,
        }

    def _gpu_metrics(self) -> List[Dict[str, Any]]:
        gpus: List[Dict[str, Any]] = []
        if self.using_fallback:
            data = self.smi.query()
            for i, g in enumerate(data):
                g2 = g.copy()
                g2["index"] = i
                gpus.append(g2)
            return gpus
        if not NVML_AVAILABLE or self.gpu_count <= 0:
            return gpus
        for i in range(self.gpu_count):
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(h).decode("utf-8", "ignore")
                uuid = pynvml.nvmlDeviceGetUUID(h).decode("utf-8", "ignore")
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                temp_gpu = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                temp_mem = None
                temp_mem_supported = False
                try:
                    temp_mem = pynvml.nvmlDeviceGetTemperature(h, getattr(pynvml, "NVML_TEMPERATURE_MEMORY", 1))
                    temp_mem_supported = (temp_mem is not None and temp_mem > 0)
                except Exception:
                    pass
                try:
                    power_w = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                except Exception:
                    power_w = None
                try:
                    fan_pct = pynvml.nvmlDeviceGetFanSpeed_v2(h).speed
                except Exception:
                    try:
                        fan_pct = pynvml.nvmlDeviceGetFanSpeed(h)
                    except Exception:
                        fan_pct = None
                try:
                    sm_clock = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
                    mem_clock = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM)
                except Exception:
                    sm_clock = None
                    mem_clock = None

                gpus.append({
                    "index": i,
                    "name": name,
                    "uuid": uuid,
                    "gpu_util_percent": util.gpu if util else None,
                    "mem_util_percent": util.memory if util else None,
                    "vram_used_GB": round(mem.used / (1024**3), 3) if mem else None,
                    "vram_total_GB": round(mem.total / (1024**3), 3) if mem else None,
                    "temp_gpu_c": temp_gpu,
                    "temp_mem_c": temp_mem,
                    "temp_mem_supported": temp_mem_supported,
                    "power_w": round(power_w, 2) if power_w is not None else None,
                    "fan_percent": fan_pct,
                    "sm_clock_mhz": sm_clock,
                    "mem_clock_mhz": mem_clock,
                })
            except Exception as e:
                gpus.append({
                    "index": i,
                    "name": None,
                    "uuid": None,
                    "gpu_util_percent": None,
                    "mem_util_percent": None,
                    "vram_used_GB": None,
                    "vram_total_GB": None,
                    "temp_gpu_c": None,
                    "temp_mem_c": None,
                    "temp_mem_supported": False,
                    "power_w": None,
                    "fan_percent": None,
                    "sm_clock_mhz": None,
                    "mem_clock_mhz": None,
                    "error": str(e),
                })
        return gpus

    def sample(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"timestamp": now_iso()}
        data.update(self._cpu_metrics())
        data.update(self._disk_net_metrics())
        for g in self._gpu_metrics():
            idx = g.get("index", 0)
            prefix = f"gpu{idx}_"
            for k, v in g.items():
                if k in ("index", "error"):
                    continue
                data[prefix + k] = v
        return data

class TelemetryController:
    def __init__(self, interval_s: float = DEFAULT_INTERVAL_S):
        self.interval_s = interval_s
        self.running = threading.Event()
        self.profile_mode = threading.Event()
        self.sampler = TelemetrySampler()
        self.csv = CSVLogger(LOG_DIR)
        self.thread = None
        self.header_cache: Optional[List[str]] = None
        self.last_sample: Optional[Dict[str, Any]] = None

    def _build_header(self) -> List[str]:
        sample = self.sampler.sample()
        header = list(sample.keys())
        wanted = ['timestamp','cpu_percent','ram_used_gb','ram_total_gb','ram_percent','cpu_temp_c','cpu_pkg_temp_c','mb_temp_c']
        pref = [c for c in wanted if c in header]
        rest = [c for c in header if c not in pref]
        return pref + rest

    def _loop(self):
        if self.header_cache is None:
            self.header_cache = self._build_header()
        while self.running.is_set():
            try:
                sample = self.sampler.sample()
                self.last_sample = sample
                keys = list(sample.keys())
                if set(keys) != set(self.header_cache):
                    merged = list(dict.fromkeys(self.header_cache + [k for k in keys if k not in self.header_cache]))
                    self.header_cache = merged
                row = [sample.get(k, None) for k in self.header_cache]
                self.csv.write_row(self.header_cache, row)
            except Exception as e:
                sys.stderr.write(f"[WARN] Sample/log error: {e}\n")
            time.sleep(PROFILE_INTERVAL_S if self.profile_mode.is_set() else self.interval_s)

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.running.set()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running.clear()
        if self.thread:
            self.thread.join(timeout=3)

    def set_profile_mode(self, enabled: bool):
        if enabled:
            self.profile_mode.set()
        else:
            self.profile_mode.clear()

# ---------- Minimal Tkinter UI (toggle + profile + view log) ----------
import tkinter as tk
from tkinter import ttk, messagebox

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Telemetry Logger")
        self.geometry("640x220")
        self.resizable(False, False)

        self.controller = TelemetryController()

        pad = {'padx': 12, 'pady': 8}
        info = ttk.Frame(self)
        info.pack(fill='x', **pad)

        self.status_var = tk.StringVar(value="Status: Stopped")
        self.mode_var = tk.StringVar(value="Mode: Continuous (15s)")
        self.logdir_var = tk.StringVar(value=f"Log dir: {LOG_DIR}")
        self.backend_var = tk.StringVar(value=self._backend_text())

        ttk.Label(info, textvariable=self.status_var).pack(anchor='w')
        ttk.Label(info, textvariable=self.mode_var).pack(anchor='w')
        ttk.Label(info, textvariable=self.logdir_var).pack(anchor='w')
        ttk.Label(info, textvariable=self.backend_var, foreground="#666").pack(anchor='w')

        # Buttons row
        row = ttk.Frame(self)
        row.pack(fill='x', **pad)

        # Start/Stop toggle
        self.toggle_btn = tk.Button(row, text="Start logging", command=self.on_toggle,
                                    width=18, fg="white", bg="#2e7d32", activebackground="#1b5e20")
        self.toggle_btn.pack(side='left', padx=6)

        # Profile Mode toggle (1s)
        self.profile_btn = tk.Button(row, text="Enable Profile Mode (1s)", command=self.on_profile_toggle,
                                     width=22, fg="white", bg="#1565c0", activebackground="#0d47a1")
        self.profile_btn.pack(side='left', padx=6)

        # View log file
        self.view_btn = tk.Button(row, text="View Log File", command=self.on_view_log,
                                  width=16, fg="white", bg="#455a64", activebackground="#263238")
        self.view_btn.pack(side='left', padx=6)

        # Shortcuts
        self.bind("<Control-s>", lambda e: self.on_toggle())
        self.bind("<Control-p>", lambda e: self.on_profile_toggle())
        self.bind("<Control-l>", lambda e: self.on_view_log())
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _backend_text(self):
        if self.controller.sampler.using_fallback and self.controller.smi.available:
            return "GPU telemetry via nvidia-smi fallback"
        elif NVML_AVAILABLE:
            return "GPU telemetry via NVML"
        else:
            return "No GPU telemetry (install drivers & nvidia-ml-py3)"

    def on_toggle(self):
        try:
            if not self.controller.running.is_set():
                self.controller.start()
                self.status_var.set("Status: Running")
                self.toggle_btn.configure(text="Stop logging", bg="#c62828", activebackground="#8e0000")
            else:
                self.controller.stop()
                self.status_var.set("Status: Stopped")
                self.toggle_btn.configure(text="Start logging", bg="#2e7d32", activebackground="#1b5e20")
        except Exception as e:
            messagebox.showerror("Error", f"Operation failed:\n{e}")

        if self.controller.profile_mode.is_set():
            self.mode_var.set("Mode: PROFILE (1s)")
        else:
            self.mode_var.set(f"Mode: Continuous ({int(DEFAULT_INTERVAL_S)}s)")

    def on_profile_toggle(self):
        try:
            enabled = not self.controller.profile_mode.is_set()
            self.controller.set_profile_mode(enabled)
            if enabled:
                self.profile_btn.configure(text="Disable Profile Mode", bg="#ef6c00", activebackground="#e65100")
                if self.controller.running.is_set():
                    self.status_var.set("Status: Running")
                self.mode_var.set("Mode: PROFILE (1s)")
            else:
                self.profile_btn.configure(text="Enable Profile Mode (1s)", bg="#1565c0", activebackground="#0d47a1")
                self.mode_var.set(f"Mode: Continuous ({int(DEFAULT_INTERVAL_S)}s)")
        except Exception as e:
            messagebox.showerror("Error", f"Operation failed:\n{e}")

    def on_view_log(self):
        path = self.controller.csv.current_file_path()
        if not path or not os.path.exists(path):
            messagebox.showinfo("Log File", "No log file yet. Start logging to create today's CSV.")
            return
        try:
            if platform.system().lower() == "windows":
                os.startfile(path)
            elif platform.system().lower() == "darwin":
                subprocess.call(["open", path])
            else:
                subprocess.call(["xdg-open", path])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open log file:\n{e}\nPath: {path}")

    def on_close(self):
        try:
            self.controller.stop()
        finally:
            self.destroy()

if __name__ == "__main__":
    # Proactively ensure NVML dll is accessible for pynvml on Windows
    if platform.system().lower() == 'windows':
        try:
            _ensure_nvml_in_nvsm_windows(show_message=False)
        except Exception:
            pass
    # Ensure NVML preload attempt on Windows before Tk loop (just in case)
    if platform.system().lower() == "windows":
        try:
            _ensure_nvml_loaded()
        except Exception:
            pass
    App().mainloop()
