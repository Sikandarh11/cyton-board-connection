"""Session-oriented EEG backend.

One start/stop session writes one EDF file while still emitting chunk-by-chunk
inference results in sequence.
"""

import argparse
import json
import os
import time
from datetime import datetime

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

class SessionPipeline:
    """Backend-friendly session controller.

    Integration flow:
    1) start_session()
    2) call poll_once() in a loop (or run_until_stopped())
    3) stop_session()
    """

    def __init__(self, config_path: str):
        self._cfg = load_config(config_path)

        self._verbose = self._cfg["run"].get("verbose", True)
        self._print_per_window = self._cfg["run"].get("print_per_window", True)
        self._resume = self._cfg["recording"].get("resume", False)

        self._board = BoardInterface(self._cfg["board"])
        self._recorder = Recorder(self._cfg["channel_map"], self._cfg["signal"], self._cfg["recording"])
        self._preproc = Preprocessor(self._cfg["signal"])
        self._writer = EDFWriter(
            self._cfg["recording"],
            self._cfg["edf"],
            self._cfg["signal"],
            channel_names=self._recorder.channel_names,
        )
        self._engine = InferenceEngine(self._cfg["model"])
        self._store = ResultStore(self._cfg["output"])

        self._record_idx = EDFWriter.load_counter(
            self._cfg["recording"]["output_dir"],
            self._cfg["recording"].get("filename_prefix", "record"),
            self._resume,
        )

        self._chunk_size = max(1, self._board.sampling_rate // 10)
        self._window_sec = self._cfg["recording"]["window_sec"]

        self._session_active = False
        self._session_started_at = None
        self._session_subject_id = None
        self._session_chunks = []
        self._session_raw_windows = []

    def start_session(self) -> dict:
        if self._session_active:
            raise RuntimeError("Session is already active.")

        # Ensure every session starts with an empty recording buffer.
        self._recorder = Recorder(self._cfg["channel_map"], self._cfg["signal"], self._cfg["recording"])

        prefix = self._cfg["recording"].get("filename_prefix", "record")
        self._session_subject_id = f"{prefix}_{self._record_idx}.edf"
        self._session_chunks = []
        self._session_raw_windows = []
        self._session_started_at = datetime.now().isoformat(timespec="seconds")

        self._board.connect()
        if self._verbose:
            print("[main] Warming up (2 s) ...")
        time.sleep(2)

        self._session_active = True
        return {
            "session_id": self._session_subject_id,
            "window_sec": self._window_sec,
            "sampling_rate": self._board.sampling_rate,
            "status": "started",
        }

    def poll_once(self) -> list:
        if not self._session_active:
            raise RuntimeError("No active session. Call start_session() first.")

        emitted = []
        raw_chunk = self._board.read(self._chunk_size)
        self._recorder.push(raw_chunk)

        while self._recorder.window_ready():
            window_raw = self._recorder.pop_window()
            self._session_raw_windows.append(window_raw)

            window_clean = self._preproc.process(window_raw)
            result = self._engine.predict(window_clean, subject_id=self._session_subject_id)

            chunk_idx = len(self._session_chunks) + 1
            chunk_entry = self._build_chunk_entry(result, chunk_idx)
            self._session_chunks.append(chunk_entry)
            emitted.append(chunk_entry)

            if self._print_per_window:
                print(f"[main] chunk#{chunk_idx} -> {chunk_entry['final_label']}")

        return emitted

    def stop_session(self) -> dict:
        if not self._session_active:
            raise RuntimeError("No active session to stop.")

        ended_at = datetime.now().isoformat(timespec="seconds")

        try:
            self._board.disconnect()
        finally:
            self._session_active = False

        if not self._session_raw_windows:
            session_summary = {
                "session_id": self._session_subject_id,
                "started_at": self._session_started_at,
                "ended_at": ended_at,
                "total_chunks": 0,
                "chunk_results": [],
                "final_subject_stage": None,
                "status": "stopped_no_data",
            }
            self._store.append_session(session_summary)
            self._store.flush()
            return session_summary

        session_raw = np.concatenate(self._session_raw_windows, axis=1)
        edf_path = self._writer.save(session_raw, record_index=self._record_idx)

        session_summary = self._store.build_session_summary(
            session_id=self._session_subject_id,
            record_index=self._record_idx,
            edf_path=edf_path,
            started_at=self._session_started_at,
            ended_at=ended_at,
            chunk_results=self._session_chunks,
        )

        self._store.append_session(session_summary)
        self._store.flush()

        self._record_idx += 1
        EDFWriter.save_counter(
            self._cfg["recording"]["output_dir"],
            self._cfg["recording"].get("filename_prefix", "record"),
            self._record_idx,
        )

        if self._verbose:
            print(f"[main] Saved session EDF -> {edf_path}")
            print(f"[main] Final subject stage -> {session_summary['final_subject_stage']}")

        return session_summary

    def run_until_stopped(self) -> dict:
        self.start_session()
        try:
            while True:
                self.poll_once()
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\n[main] Interrupted by user.")
        return self.stop_session()

    @staticmethod
    def _build_chunk_entry(result: dict, chunk_index: int) -> dict:
        return {
            "chunk_index": chunk_index,
            "subject_id": result.get("subject_id"),
            "final_label": result.get("final_label"),
            "n_windows": result.get("n_windows"),
            "stage1_prediction": result.get("stage1_prediction"),
            "stage1_confidence": result.get("stage1_confidence"),
            "stage1_votes": result.get("stage1_votes"),
            "stage1_mean_probs": result.get("stage1_mean_probs"),
            "stage2_prediction": result.get("stage2_prediction"),
            "stage2_confidence": result.get("stage2_confidence"),
            "stage2_votes": result.get("stage2_votes"),
            "stage2_mean_probs": result.get("stage2_mean_probs"),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }


def run(config_path: str) -> None:
    pipeline = SessionPipeline(config_path)
    summary = pipeline.run_until_stopped()
    print(f"[main] Session chunks: {summary.get('total_chunks', 0)}")
    print(f"[main] Results -> {pipeline._cfg['output']['session_json']}")
    print(f"[main] Summary -> {pipeline._cfg['output']['summary_csv']}")


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
