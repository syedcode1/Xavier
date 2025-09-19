# Xavier

A cross-platform system telemetry logger with GUI that captures hardware metrics to CSV files for performance analysis.

## Features

- **Real-time monitoring** of CPU, RAM, GPU, disk I/O, and network
- **Dual-mode operation**: Standard (15s) or Profile mode (1s) for high-frequency sampling
- **Multi-GPU support** via NVML with automatic nvidia-smi fallback
- **Temperature monitoring** including CPU package and motherboard (requires LibreHardwareMonitor on Windows)
- **Daily rolling CSV logs** with automatic schema adaptation
- **Simple GUI** with Start/Stop toggle and quick log file access

## Xavier UI 
Simply click on Enable Profile Mode (1 sec) to capture the telemetr to a csv file every second.
<img width="1120" height="445" alt="Screenshot 2025-09-19 224742" src="https://github.com/user-attachments/assets/d5ebc227-f55f-4a15-9f8f-57fa088b4791" />

## Analyze Data
Xavier helps you visualize the problem. For example, below we are using 2D Chart in Microsoft Excel to analyze the captured data. It shows us during the AI workload, the temperature on GPU reached dangerously high to 91 celcius, however the GPU fan didn't keep up and was still running at 60%. 

<img width="945" height="524" alt="Screenshot 2025-09-19 230522" src="https://github.com/user-attachments/assets/965e63a3-8619-463b-aff5-4bede244db03" />


## Installation

### Clone the repository
```bash
git clone https://github.com/syedcode1/Xavier.git
cd Xavier
```

### Install dependencies
```bash
pip install psutil nvidia-ml-py3
```

### Optional (Windows)
```bash
pip install wmi pywin32  # For additional temperature sensors
```

## Usage

```bash
python telemetry_dualmode_logger.py
```

Xavier will open with a simple control panel.

### GUI Controls
- **Green/Red Toggle**: Start/Stop logging
- **Profile Mode**: Toggle 1-second sampling for burst analysis
- **View Log**: Open today's CSV file

### Keyboard Shortcuts
- `Ctrl+S`: Start/Stop logging
- `Ctrl+P`: Toggle Profile Mode
- `Ctrl+L`: View log file

## Output

Logs are saved to `./logs/telemetry_YYYY-MM-DD.csv` with columns including:
- System: `timestamp`, `cpu_percent`, `ram_used_gb`, `cpu_temp_c`
- GPU (per device): `gpu0_util_percent`, `gpu0_vram_used_GB`, `gpu0_temp_gpu_c`, `gpu0_power_w`
- I/O: `disk_read_MBps`, `disk_write_MBps`, `net_rx_Mbps`, `net_tx_Mbps`

## Temperature Monitoring (Windows)

### LibreHardwareMonitor Setup
**Required for CPU temperature capture:**
1. Download and run LibreHardwareMonitor as Administrator
2. Enable: Options → Remote Web Server → Start
3. Keep LibreHardwareMonitor running while using Xavier
4. (Optional) Set custom endpoint: `export LHM_URL=http://localhost:8085/data.json`

### Custom NVML Path (Windows)
```bash
export NVML_DLL_PATH=C:\Windows\System32\nvml.dll
```

## Notes

- Xavier will auto-detect and use nvidia-smi if NVML is unavailable
- First run on Windows may require Administrator privileges to copy nvml.dll
- Logs roll daily at midnight with automatic header updates for new metrics
- Profile Mode (1s) is useful for capturing GPU inference bursts or performance spikes
- CPU temperatures require LibreHardwareMonitor to be running on Windows

## License

Apache 2.0
