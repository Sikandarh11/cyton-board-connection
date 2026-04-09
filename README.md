# cyton-board-connection

EEG processing project for OpenBCI Cyton with:
- real-time acquisition and streaming inference,
- complete offline inference pipeline,
- utility checks for device diagnostics.

## Project Layout

- `1. real-time-pipeline/`: live board connection, preprocessing, EDF writing, and chunk/session inference
- `2. offline_pipeline/`: two-stage offline inference from EDF or NumPy arrays
- `3. checks/`: utility scripts for Cyton serial diagnostics and quick live signal UI checks
- `outputs/`: generated recordings, logs, and result files

## Setup (CPU, Windows)

Run commands from repository root:

`D:/5. FYP/2. Code/4. Cyton board code/cyton-board-connection`

### 1) Create environment

Conda:

```powershell
conda create -n cyton-cpu python=3.10 -y
conda activate cyton-cpu
```

Or venv:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2) Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3) Verify model files

Required checkpoints:
- `1. real-time-pipeline/models/ShallowFBCSPNet_best_control_vs_ds.pt`
- `1. real-time-pipeline/models/ShallowFBCSPNet_best_ds_abnormal.pt`
- `2. offline_pipeline/models/ShallowFBCSPNet_best_control_vs_ds.pt`
- `2. offline_pipeline/models/ShallowFBCSPNet_best_ds_abnormal.pt`

## Real-Time Pipeline

### Run with real Cyton

1. Edit `1. real-time-pipeline/config_realtime.json`.
2. Set `board.board_id` to `0` and `board.serial_port` to your COM port (for example `COM10`).
3. Start:

```powershell
python "1. real-time-pipeline/main.py"
```

Optional explicit config:

```powershell
python "1. real-time-pipeline/main.py" --config "1. real-time-pipeline/config_realtime.json"
```

### Run without hardware (synthetic board)

In `1. real-time-pipeline/config_realtime.json` set:
- `board.board_id: -1`
- `board.serial_port: ""`

Then run the same `main.py` command.

## Offline Pipeline

The offline pipeline is now complete and supports:
- single EDF inference,
- folder-level batch inference,
- direct NumPy-array inference.

Core files:
- `2. offline_pipeline/handler_combined.py`
- `2. offline_pipeline/config_combined.json`
- `2. offline_pipeline/example_usage_combined.py`

### Quick start (example script)

```powershell
python "2. offline_pipeline/example_usage_combined.py"
```

### Minimal usage pattern

```python
from handler_combined import CombinedEEGHandler

handler = CombinedEEGHandler("2. offline_pipeline/config_combined.json")

# Single EDF
result = handler.predict_file("path/to/file.edf")
print(result["final_label"])

# Folder of EDF files
df = handler.predict_folder("path/to/folder")
print(df[["subject_id", "final_label"]])
```

## Check Scripts

Serial/BrainFlow diagnose:

```powershell
python "3. checks/cyton_manual_diagnose.py"
python "3. checks/cyton_manual_diagnose.py" --port COM10
```

Live signal UI:

```powershell
python "3. checks/get_sig_with_ui.py"
```

## Outputs

Common output locations:
- `outputs/recordings/record_N.edf`
- `outputs/results/session_results.json`
- `outputs/results/summary.csv`
- `outputs/inference/` (offline inference results)
- `outputs/logs/`

Stop running pipelines with `Ctrl+C`.