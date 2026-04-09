"""
result_store.py
===============
Persists inference results:

1. **Session JSON** (``outputs/results/session_results.json``)
   Grows incrementally — one entry per 10-second window.
   Reload-safe: if the file already exists it is extended, not overwritten.

2. **Summary CSV** (``outputs/results/summary.csv``)
   One row per recording, flat columns, easy to open in Excel / pandas.

Result dict schema (written verbatim, minus the ``window_results`` DataFrame)
--------------
{
    "record_index"      : int,       # 1-based recording counter
    "edf_path"          : str,       # path to the saved EDF
    "subject_id"        : str,
    "final_label"       : str,
    "n_windows"         : int,
    "stage1_prediction" : str,
    "stage1_confidence" : float,
    "stage1_votes"      : dict,
    "stage1_mean_probs" : dict,
    "stage2_prediction" : str | None,
    "stage2_confidence" : float | None,
    "stage2_votes"      : dict | None,
    "stage2_mean_probs" : dict | None,
}

Usage
-----
    from result_store import ResultStore
    store = ResultStore(cfg["output"])

    store.append(result, record_index=1, edf_path="/path/record_1.edf")
    store.flush()   # write session JSON and summary CSV to disk
"""

import os
import json
import csv
from datetime import datetime
from copy import deepcopy


# CSV columns in display order
_CSV_COLUMNS = [
    "record_index",
    "timestamp",
    "edf_path",
    "subject_id",
    "final_label",
    "n_windows",
    "stage1_prediction",
    "stage1_confidence",
    "stage2_prediction",
    "stage2_confidence",
    "stage1_votes_DS",
    "stage1_votes_Control",
    "stage2_votes_DS",
    "stage2_votes_Abnormal",
    "stage1_prob_DS",
    "stage1_prob_Control",
    "stage2_prob_DS",
    "stage2_prob_Abnormal",
]


class ResultStore:
    def __init__(self, output_cfg: dict):
        self._json_path = output_cfg.get("session_json", "outputs/results/session_results.json")
        self._csv_path  = output_cfg.get("summary_csv",  "outputs/results/summary.csv")

        os.makedirs(os.path.dirname(self._json_path), exist_ok=True)
        os.makedirs(os.path.dirname(self._csv_path),  exist_ok=True)

        # In-memory list of result dicts (grows each window)
        self._records: list = []

        # Load existing session if file already exists (resume mode)
        if os.path.exists(self._json_path):
            try:
                with open(self._json_path) as fh:
                    self._records = json.load(fh)
                print(f"[ResultStore] Resumed — {len(self._records)} existing records loaded.")
            except Exception:
                self._records = []

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def append(self, result: dict, record_index: int, edf_path: str) -> None:
        """
        Add one inference result to the in-memory store.

        Parameters
        ----------
        result       : dict from InferenceEngine.predict()
        record_index : 1-based counter
        edf_path     : path to the corresponding .edf file
        """
        # Drop the DataFrame — not JSON-serialisable
        entry = {k: v for k, v in result.items() if k != "window_results"}
        entry["record_index"] = record_index
        entry["edf_path"]     = edf_path
        entry["timestamp"]    = datetime.now().isoformat(timespec="seconds")
        self._records.append(entry)

    def flush(self) -> None:
        """Write / overwrite both the JSON and CSV files."""
        self._write_json()
        self._write_csv()

    def print_latest(self) -> None:
        """Pretty-print the most recent result to stdout."""
        if not self._records:
            return
        r = self._records[-1]
        print("\n" + "─" * 52)
        print(f"  Record #{r['record_index']}  ->  {r['final_label']}")
        print(f"  Stage 1 : {r['stage1_prediction']}"
              f"  (conf={r['stage1_confidence']:.2f})"
              f"  votes={r['stage1_votes']}")
        s2p = r.get("stage2_prediction")
        if s2p:
            print(f"  Stage 2 : {s2p}"
                  f"  (conf={r['stage2_confidence']:.2f})"
                  f"  votes={r['stage2_votes']}")
        print(f"  EDF     : {r['edf_path']}")
        print("─" * 52 + "\n")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_json(self) -> None:
        with open(self._json_path, "w") as fh:
            json.dump(self._records, fh, indent=2, default=str)

    def _write_csv(self) -> None:
        write_header = not os.path.exists(self._csv_path) or len(self._records) == 1
        with open(self._csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for r in self._records:
                row = {
                    "record_index"        : r.get("record_index"),
                    "timestamp"           : r.get("timestamp"),
                    "edf_path"            : r.get("edf_path"),
                    "subject_id"          : r.get("subject_id"),
                    "final_label"         : r.get("final_label"),
                    "n_windows"           : r.get("n_windows"),
                    "stage1_prediction"   : r.get("stage1_prediction"),
                    "stage1_confidence"   : r.get("stage1_confidence"),
                    "stage2_prediction"   : r.get("stage2_prediction"),
                    "stage2_confidence"   : r.get("stage2_confidence"),
                    "stage1_votes_DS"     : (r.get("stage1_votes") or {}).get("DS"),
                    "stage1_votes_Control": (r.get("stage1_votes") or {}).get("Control"),
                    "stage2_votes_DS"     : (r.get("stage2_votes") or {}).get("DS"),
                    "stage2_votes_Abnormal": (r.get("stage2_votes") or {}).get("Abnormal"),
                    "stage1_prob_DS"      : (r.get("stage1_mean_probs") or {}).get("DS"),
                    "stage1_prob_Control" : (r.get("stage1_mean_probs") or {}).get("Control"),
                    "stage2_prob_DS"      : (r.get("stage2_mean_probs") or {}).get("DS"),
                    "stage2_prob_Abnormal": (r.get("stage2_mean_probs") or {}).get("Abnormal"),
                }
                writer.writerow(row)
