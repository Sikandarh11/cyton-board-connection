"""Session-level result persistence.

Stores one JSON object per start/stop session. Each session object contains:
- one EDF path for the full session
- sequential chunk-level predictions
- final subject stage aggregation across all chunks
"""

import os
import json
import csv
from datetime import datetime
from collections import Counter


# CSV columns in display order
_CSV_COLUMNS = [
    "record_index",
    "session_id",
    "started_at",
    "ended_at",
    "edf_path",
    "total_chunks",
    "final_subject_stage",
    "final_votes_json",
    "avg_stage1_mean_probs_json",
    "avg_stage2_mean_probs_json",
]


class ResultStore:
    def __init__(self, output_cfg: dict):
        self._json_path = output_cfg.get("session_json", "outputs/results/session_results.json")
        self._csv_path  = output_cfg.get("summary_csv",  "outputs/results/summary.csv")

        os.makedirs(os.path.dirname(self._json_path), exist_ok=True)
        os.makedirs(os.path.dirname(self._csv_path),  exist_ok=True)

        self._sessions = []

        # Load existing session if file already exists (resume mode)
        if os.path.exists(self._json_path):
            try:
                with open(self._json_path) as fh:
                    data = json.load(fh)
                self._sessions = data if isinstance(data, list) else []
                print(f"[ResultStore] Resumed — {len(self._sessions)} existing sessions loaded.")
            except Exception:
                self._sessions = []

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def append_session(self, session_summary: dict) -> None:
        self._sessions.append(session_summary)

    def build_session_summary(
        self,
        session_id: str,
        record_index: int,
        edf_path: str,
        started_at: str,
        ended_at: str,
        chunk_results: list,
    ) -> dict:
        votes = Counter()
        stage1_prob_keys = ["DS", "Control"]
        stage2_prob_keys = ["DS", "Abnormal"]

        stage1_sums = {k: 0.0 for k in stage1_prob_keys}
        stage2_sums = {k: 0.0 for k in stage2_prob_keys}
        stage2_count = 0

        for chunk in chunk_results:
            label = chunk.get("final_label")
            if label:
                votes[label] += 1

            s1_probs = chunk.get("stage1_mean_probs") or {}
            for key in stage1_prob_keys:
                stage1_sums[key] += float(s1_probs.get(key, 0.0))

            s2_probs = chunk.get("stage2_mean_probs") or {}
            if s2_probs:
                stage2_count += 1
                for key in stage2_prob_keys:
                    stage2_sums[key] += float(s2_probs.get(key, 0.0))

        n_chunks = len(chunk_results)
        avg_stage1 = {
            key: round(stage1_sums[key] / n_chunks, 4) if n_chunks else None
            for key in stage1_prob_keys
        }
        avg_stage2 = {
            key: round(stage2_sums[key] / stage2_count, 4) if stage2_count else None
            for key in stage2_prob_keys
        }

        final_subject_stage = None
        if votes:
            final_subject_stage = votes.most_common(1)[0][0]

        return {
            "session_id": session_id,
            "subject_id": os.path.basename(edf_path),
            "record_index": record_index,
            "edf_path": edf_path,
            "started_at": started_at,
            "ended_at": ended_at,
            "total_chunks": n_chunks,
            "chunk_results": chunk_results,
            "final_subject_stage": final_subject_stage,
            "final_votes": dict(votes),
            "avg_stage1_mean_probs": avg_stage1,
            "avg_stage2_mean_probs": avg_stage2,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    def flush(self) -> None:
        """Write / overwrite both the JSON and CSV files."""
        self._write_json()
        self._write_csv()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_json(self) -> None:
        with open(self._json_path, "w") as fh:
            json.dump(self._sessions, fh, indent=2, default=str)

    def _write_csv(self) -> None:
        with open(self._csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for r in self._sessions:
                row = {
                    "record_index": r.get("record_index"),
                    "session_id": r.get("session_id"),
                    "started_at": r.get("started_at"),
                    "ended_at": r.get("ended_at"),
                    "edf_path": r.get("edf_path"),
                    "total_chunks": r.get("total_chunks"),
                    "final_subject_stage": r.get("final_subject_stage"),
                    "final_votes_json": json.dumps(r.get("final_votes") or {}),
                    "avg_stage1_mean_probs_json": json.dumps(r.get("avg_stage1_mean_probs") or {}),
                    "avg_stage2_mean_probs_json": json.dumps(r.get("avg_stage2_mean_probs") or {}),
                }
                writer.writerow(row)
