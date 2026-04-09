# Real-time EEG Pipeline

Modular real-time EEG acquisition → EDF storage → two-stage DS/Control/Abnormal inference.

---

## File structure

```
realtime_eeg_pipeline/
│
├── config_realtime.json      ← All settings (board, signal, recording, model, output)
│
├── main.py                   ← Orchestrator — run this
│
├── board_interface.py        ← BrainFlow connection / streaming / read
├── recorder.py               ← 10-second buffer + channel remapping
├── preprocessor.py           ← Detrend · notch · bandpass (BrainFlow)
├── edf_writer.py             ← Save windows as record_N.edf (pyEDFlib)
├── inference_engine.py       ← Adapter to CombinedEEGHandler
└── result_store.py           ← JSON session log + CSV summary
│
├── combined_handler_ds_abnormal_control/
│   ├── config_combined.json  ← CombinedEEGHandler model config
│   └── handler_combined.py   ← Your existing two-stage handler
│
└── outputs/
    ├── recordings/           ← record_1.edf, record_2.edf, …
    ├── results/
    │   ├── session_results.json
    │   └── summary.csv
    └── logs/
```

---

## Setup

```bash
pip install brainflow pyEDFlib mne torch braindecode scikit-learn pandas
```

> **Python 3.9–3.11** recommended (BrainFlow + braindecode constraint).

---

## Run

```bash
# Default config
python main.py

# Custom config
python main.py --config path/to/config_realtime.json
```

Press **Ctrl-C** to stop cleanly.

---

## Electrode → Cyton channel mapping

The Cyton board has **8 EEG input pins** (N1P–N8P on the bottom header).
All channels are referenced to **Cz** (SRB2 pin = reference electrode placed at Cz).

| Cyton pin | Config key | Electrode | Location |
|-----------|------------|-----------|----------|
| N1P       | `"1"`      | EEG F3-Cz | Left frontal |
| N2P       | `"2"`      | EEG Fz-Cz | Midline frontal |
| N3P       | `"3"`      | EEG F4-Cz | Right frontal |
| N4P       | `"4"`      | EEG C3-Cz | Left central |
| N5P       | `"5"`      | EEG Cz-Cz | Midline central (= reference → expect ~0 µV) |
| N6P       | `"6"`      | EEG C4-Cz | Right central |
| N7P       | `"7"`      | EEG P3-Cz | Left parietal |
| N8P       | `"8"`      | EEG Pz-Cz | Midline parietal |
| *(none)*  | `"9_fallback"` | EEG P4-Cz | Right parietal — **synthesised** from ch8 |

### Channel 9 (P4-Cz)

The model was trained on **9 channels**.  The standard Cyton provides 8.
Two options:

**Option A — Cyton Daisy module** (hardware)
Gives 16 channels total; map Daisy ch1 to P4-Cz.
Set `"use_ch9_fallback": false` and add `"9": "EEG P4-Cz"` mapping the Daisy channel.

**Option B — Software fallback** (default)
Set `"use_ch9_fallback": true`.
`recorder.py` copies Pz-Cz into the P4-Cz slot.
This is a reasonable approximation for P-region activity when Daisy is unavailable.

### Reference electrode wiring

- **SRB2** pin → electrode at **Cz** (top of head)
- **BIAS** pin → ear lobe or mastoid (common ground / DRL)
- All N*P channels then record the differential against Cz, matching the EDF channel labels exactly.

---

## Output files

### `outputs/recordings/record_N.edf`

Standard EDF+ file.  Open in:
- **MNE-Python**: `mne.io.read_raw_edf("record_1.edf", preload=True)`
- **EDFbrowser** (free GUI)
- Any other EDF-compatible tool

Metadata stored in each EDF:
| Field | Value |
|-------|-------|
| Fs | 250 Hz |
| Duration | 10 s (configurable) |
| Channels | 9 × Cz-referenced labels |
| Physical unit | µV |
| Physical range | −500 to +500 µV |
| Prefilter | 0.5–45 Hz Butterworth; notch |

### `outputs/results/session_results.json`

Grows incrementally — one JSON object per window:

```json
[
  {
    "record_index":        1,
    "timestamp":           "2025-04-09T14:23:01",
    "edf_path":            "outputs/recordings/record_1.edf",
    "subject_id":          "record_1",
    "final_label":         "DS",
    "n_windows":           10,
    "stage1_prediction":   "DS",
    "stage1_confidence":   0.87,
    "stage1_votes":        {"DS": 8, "Control": 2},
    "stage1_mean_probs":   {"DS": 0.87, "Control": 0.13},
    "stage2_prediction":   "DS",
    "stage2_confidence":   0.91,
    "stage2_votes":        {"DS": 9, "Abnormal": 1},
    "stage2_mean_probs":   {"DS": 0.91, "Abnormal": 0.09}
  },
  …
]
```

### `outputs/results/summary.csv`

Flat CSV — one row per recording.  Useful for batch analysis in Excel or pandas.

---

## Testing without hardware

Set `"board_id": -1` in `config_realtime.json` to use BrainFlow's **SYNTHETIC_BOARD**.
No serial port needed — generates plausible random EEG for pipeline testing.

```json
"board": {
    "board_id": -1,
    "serial_port": ""
}
```

---

## Tuning the config

| Key | Effect |
|-----|--------|
| `recording.window_sec` | Length of each segment. Must match the model's expected input duration. |
| `signal.notch_freq` | 50 Hz (EU/Pakistan) or 60 Hz (US). |
| `signal.bandpass_low/high` | Filter edges. Default 0.5–45 Hz covers all standard EEG bands. |
| `recording.max_records` | Set to an integer to auto-stop. `null` = run forever. |
| `recording.resume` | `true` = continue record numbering from last run. |
| `channel_map.use_ch9_fallback` | `true` = synthesise P4-Cz from Pz-Cz if no Daisy. |
