"""
main.py
=======
Real-time EEG acquisition + two-stage inference pipeline.

Loop
----
    1.  BoardInterface streams data continuously.
    2.  Recorder accumulates samples into a 10-second buffer.
    3.  When a full window is ready:
        a.  EDFWriter saves the raw window to record_N.edf
        b.  Preprocessor applies detrend / notch / bandpass
        c.  InferenceEngine runs CombinedEEGHandler on the clean window
        d.  ResultStore appends and flushes results to JSON + CSV
    4.  Repeat until KeyboardInterrupt or max_records reached.

Run
---
    python main.py                           # uses config_realtime.json
    python main.py --config my_config.json   # custom config path
"""

import argparse
import json
import os
import time
import sys

import numpy as np

from board_interface import BoardInterface
from recorder        import Recorder
from preprocessor    import Preprocessor
from edf_writer      import EDFWriter
from inference_engine import InferenceEngine
from result_store    import ResultStore


# ──────────────────────────────────────────────────────────────────────
# Config loader
# ──────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: '{path}'")
    with open(path) as fh:
        return json.load(fh)


# ──────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────

def run(config_path: str) -> None:
    cfg = load_config(config_path)

    verbose          = cfg["run"].get("verbose", True)
    print_per_window = cfg["run"].get("print_per_window", True)
    max_records      = cfg["recording"].get("max_records", None)
    resume           = cfg["recording"].get("resume", False)

    # ── Initialise modules ────────────────────────────────────────────
    board     = BoardInterface(cfg["board"])
    recorder  = Recorder(cfg["channel_map"], cfg["signal"], cfg["recording"])
    preproc   = Preprocessor(cfg["signal"])
    writer    = EDFWriter(
        cfg["recording"], cfg["edf"], cfg["signal"],
        channel_names=recorder.channel_names,
    )
    engine    = InferenceEngine(cfg["model"])
    store     = ResultStore(cfg["output"])

    # Determine starting record index
    record_idx = EDFWriter.load_counter(
        cfg["recording"]["output_dir"],
        cfg["recording"].get("filename_prefix", "record"),
        resume,
    )

    fs          = board.sampling_rate           # from board object
    chunk_size  = max(1, fs // 10)             # pull ~100 ms at a time
    window_sec  = cfg["recording"]["window_sec"]

    if verbose:
        print("\n" + "═" * 52)
        print("  Real-time EEG Pipeline")
        print(f"  Fs={fs} Hz  |  window={window_sec} s  |  record prefix='"
              f"{cfg['recording'].get('filename_prefix','record')}'")
        print(f"  Channels: {recorder.channel_names}")
        print("  Press Ctrl-C to stop.")
        print("═" * 52 + "\n")

    # ── Main loop ─────────────────────────────────────────────────────
    try:
        board.connect()

        # Brief warm-up so BrainFlow ring buffer fills
        print("[main] Warming up (2 s) ...")
        time.sleep(2)

        while True:
            # 1. Read a small chunk from the board
            raw_chunk = board.read(chunk_size)   # (n_raw_channels, chunk_size)

            # 2. Push into recorder
            recorder.push(raw_chunk)

            # 3. Check if a full 10-second window is ready
            if not recorder.window_ready():
                time.sleep(0.01)  # yield CPU
                continue

            window_raw = recorder.pop_window()   # (9, window_samples) float32

            if verbose:
                print(f"\n[main] Window ready — record #{record_idx}")

            # 4a. Save raw EDF
            edf_path = writer.save(window_raw, record_index=record_idx)
            if verbose:
                print(f"[main] EDF saved -> {edf_path}")

            # 4b. Preprocess
            window_clean = preproc.process(window_raw)  # (9, window_samples)

            # 4c. Inference
            subject_id = f"{cfg['recording'].get('filename_prefix','record')}_{record_idx}"
            result = engine.predict(window_clean, subject_id=subject_id)

            # 4d. Store results
            store.append(result, record_index=record_idx, edf_path=edf_path)
            store.flush()

            if print_per_window:
                store.print_latest()

            # Persist counter
            record_idx += 1
            EDFWriter.save_counter(
                cfg["recording"]["output_dir"],
                cfg["recording"].get("filename_prefix", "record"),
                record_idx,
            )

            # Stop condition
            if max_records is not None and record_idx > max_records:
                print(f"[main] Reached max_records={max_records}. Stopping.")
                break

    except KeyboardInterrupt:
        print("\n[main] Interrupted by user.")

    finally:
        board.disconnect()
        print(f"\n[main] Session complete. {record_idx - 1} windows recorded.")
        print(f"[main] Results -> {cfg['output']['session_json']}")
        print(f"[main] Summary -> {cfg['output']['summary_csv']}")


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time EEG pipeline")
    parser.add_argument(
        "--config",
        default="1. real-time-pipeline\\config_realtime.json",
        help="Path to config_realtime.json (default: config_realtime.json)",
    )
    args = parser.parse_args()
    run(args.config)
