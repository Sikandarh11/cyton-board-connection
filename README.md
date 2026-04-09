# cyton-board-connection

Real-time EEG acquisition from OpenBCI Cyton, EDF recording, and two-stage inference.

## Project Layout

- `1. real-time-pipeline/`: main acquisition + preprocessing + inference pipeline
- `2. checks/`: utility scripts for Cyton serial diagnostics and live signal plotting
- `outputs/`: generated recordings, logs, and prediction outputs

## CPU-Only Setup (Windows)

Run all commands from the repository root:

`D:/5. FYP/2. Code/4. Cyton board code/cyton-board-connection`

### 1) Create and activate environment

Conda (recommended):

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
python -m pip install -r req.txt
```

Notes:
- `req.txt` is prepared for CPU usage.
- Device selection in configs is set to CPU (`"device": "cpu"`).

### 3) Verify model/config paths

Check these files:
- `1. real-time-pipeline/config_realtime.json`
- `1. real-time-pipeline/config_combined.json`

Make sure model checkpoints exist at:
- `1. real-time-pipeline/models/ShallowFBCSPNet_best_control_vs_ds.pt`
- `1. real-time-pipeline/models/ShallowFBCSPNet_best_ds_abnormal.pt`

## How To Run

### Option A: Real Cyton board

1. Edit `1. real-time-pipeline/config_realtime.json`:
- Set `board.board_id` to `0` (Cyton).
- Set correct `board.serial_port` (for example `COM10`).

2. Start pipeline:

```powershell
python "1. real-time-pipeline/main.py"
```

Or with explicit config path:

```powershell
python "1. real-time-pipeline/main.py" --config "1. real-time-pipeline/config_realtime.json"
```

### Option B: Test without hardware (synthetic board)

1. In `1. real-time-pipeline/config_realtime.json` set:
- `board.board_id` to `-1`
- `board.serial_port` to empty string `""`

2. Run the same command:

```powershell
python "1. real-time-pipeline/main.py"
```

## Optional Diagnostic Scripts

### 1) Serial/BrainFlow diagnose

```powershell
python "2. checks/cyton_manual_diagnose.py"
```

For a specific COM port:

```powershell
python "2. checks/cyton_manual_diagnose.py" --port COM10
```

### 2) Live signal UI check

```powershell
python "2. checks/get_sig_with_ui.py"
```

## Outputs

After running, files are written to:

- `outputs/recordings/record_N.edf`
- `outputs/results/session_results.json`
- `outputs/results/summary.csv`
- `outputs/logs/`

## Stop Run

Press `Ctrl+C` to stop cleanly.