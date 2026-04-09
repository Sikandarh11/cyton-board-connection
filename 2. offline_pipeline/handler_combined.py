# ============================================================
# handler_combined.py
# Two-stage EEG classification pipeline
#
# Stage 1: DS vs Control  → if Control  → result = "Control (Non-DS)"
#                         → if DS       → go to Stage 2
# Stage 2: DS vs Abnormal → if DS       → result = "DS"
#                         → if Abnormal → result = "Abnormal (Non-DS)"
#
# Usage:
#   from handler_combined import CombinedEEGHandler
#   handler = CombinedEEGHandler("config_combined.json")
#   result  = handler.predict_file("/data/subject01.edf")
#   result  = handler.predict_array(eeg_array, subject_id="subj_01")
#   df      = handler.predict_folder("/data/edf_folder/")
# ============================================================

import os
import json
import logging
import warnings
import contextlib
import inspect
from io import StringIO
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
#handler path
path = "2. offline_pipeline\\config_combined.json"

# ============================================================
# LOGGING SETUP
# ============================================================

def _setup_logger(log_to_file: bool = True, log_dir: str = "outputs/logs") -> logging.Logger:
    logger = logging.getLogger("CombinedEEGHandler")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler — always on
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler — optional
    if log_to_file:
        os.makedirs(log_dir, exist_ok=True)
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"inference_{ts}.log")
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.info(f"Logging to file: {log_path}")

    return logger


# ============================================================
# FINAL LABEL MAPPING
# ============================================================

# Maps (stage1_pred, stage2_pred_or_None) → human-readable final label
#
#   Stage 1 = "Control"  → skip stage 2 → "Control (Non-DS)"
#   Stage 1 = "DS"       → run stage 2
#       Stage 2 = "DS"       → "DS"
#       Stage 2 = "Abnormal" → "Abnormal (Non-DS)"

def resolve_final_label(stage1_pred: str, stage2_pred: str = None) -> str:
    if stage1_pred != "DS":
        return f"{stage1_pred} (Non-DS)"
    if stage2_pred is None:
        return "DS"
    if stage2_pred == "DS":
        return "DS"
    return f"{stage2_pred} (Non-DS)"


# ============================================================
# INTERNAL DATASET
# ============================================================

class _EEGDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])


# ============================================================
# COMBINED HANDLER
# ============================================================

class CombinedEEGHandler:
    """
    Two-stage EEG classification handler.

    Stage 1 — DS vs Control
        Control → final = "Control (Non-DS)"   [pipeline stops]
        DS      → proceed to Stage 2

    Stage 2 — DS vs Abnormal
        DS       → final = "DS"
        Abnormal → final = "Abnormal (Non-DS)"

    Parameters
    ----------
    config_path : str
        Path to config_combined.json

    Example
    -------
    handler = CombinedEEGHandler("config_combined.json")

    result = handler.predict_file("data/subject01.edf")
    print(result["final_label"])        # "DS" / "Control (Non-DS)" / "Abnormal (Non-DS)"
    print(result["stage1_prediction"])
    print(result["stage2_prediction"])  # None if stage 1 was not DS

    df = handler.predict_folder("data/")
    """

    def __init__(self, config_path: str):
        self._cfg    = self._load_config(config_path)
        self._device = self._resolve_device(self._cfg.get("device", "auto"))

        log_cfg      = self._cfg.get("logging", {})
        self._logger = _setup_logger(
            log_to_file=log_cfg.get("log_to_file", True),
            log_dir    =log_cfg.get("log_dir", "outputs/logs"),
        )

        # Stage models (lazy-loaded on first predict call)
        self._stage1_model      = None
        self._stage1_n_channels = None
        self._stage1_n_times    = None
        self._stage1_l2c        = None   # label → class name

        self._stage2_model      = None
        self._stage2_n_channels = None
        self._stage2_n_times    = None
        self._stage2_l2c        = None

        self._logger.info("CombinedEEGHandler initialised")
        self._logger.info(f"Device: {self._device}")
        self._logger.info(
            f"Stage 1 model: {self._cfg['stage1']['model_name']}  "
            f"({self._cfg['stage1']['model_path']})"
        )
        self._logger.info(
            f"Stage 2 model: {self._cfg['stage2']['model_name']}  "
            f"({self._cfg['stage2']['model_path']})"
        )

    # ----------------------------------------------------------
    # PUBLIC API
    # ----------------------------------------------------------

    def predict_file(self, edf_path: str, true_label: str = None) -> dict:
        """
        Run two-stage inference on a single EDF file.

        Parameters
        ----------
        edf_path    : path to .edf file
        true_label  : optional ground-truth label for accuracy reporting

        Returns
        -------
        dict — see _build_combined_result() for all keys
        """
        self._ensure_models_loaded()
        subject_id = os.path.basename(edf_path)

        self._logger.info(f"{'='*55}")
        self._logger.info(f"Subject: {subject_id}")

        # Load raw EDF → windows using stage1 signal config
        # (both stages must share the same fs/channels/window_sec)
        chunks = self._load_edf(edf_path)

        if not chunks:
            self._logger.error(
                f"{subject_id}: No windows extracted. "
                "Check channel names and recording length."
            )
            raise RuntimeError(
                f"No windows extracted from '{edf_path}'. "
                "Check channels / window_sec / recording length."
            )

        self._logger.info(f"Windows extracted: {len(chunks)}")
        return self._run_pipeline(subject_id, chunks, true_label)

    def predict_array(self, eeg_array: np.ndarray,
                      subject_id: str = "unknown",
                      true_label: str = None) -> dict:
        """
        Run two-stage inference on a raw numpy array.

        Parameters
        ----------
        eeg_array  : np.ndarray of shape (n_channels, n_samples)
        subject_id : identifier string for logging / results
        true_label : optional ground-truth label

        Returns
        -------
        dict — same structure as predict_file()
        """
        self._ensure_models_loaded()

        self._logger.info(f"{'='*55}")
        self._logger.info(f"Subject (array): {subject_id}")

        chunks = self._window_array(
            eeg_array,
            fs         = self._cfg["signal"]["fs"],
            window_sec = self._cfg["signal"]["window_sec"],
            overlap    = self._cfg["signal"]["overlap"],
        )

        if not chunks:
            raise RuntimeError(
                "No windows extracted from array. "
                "Check that n_samples >= window_sec * fs."
            )

        self._logger.info(f"Windows extracted: {len(chunks)}")
        return self._run_pipeline(subject_id, chunks, true_label)

    def predict_folder(self, folder_path: str,
                       true_label: str = None,
                       max_subjects: int = None) -> pd.DataFrame:
        """
        Run two-stage inference on all .edf files in a folder.

        Parameters
        ----------
        folder_path  : folder containing .edf files
        true_label   : optional ground-truth class for all files in folder
        max_subjects : optional cap on number of subjects

        Returns
        -------
        pd.DataFrame — one row per subject, all result fields flattened
        """
        self._ensure_models_loaded()

        edf_files = sorted([
            f for f in os.listdir(folder_path) if f.endswith(".edf")
        ])

        if max_subjects is not None:
            edf_files = edf_files[:max_subjects]

        if not edf_files:
            raise RuntimeError(f"No .edf files found in '{folder_path}'")

        self._logger.info(
            f"predict_folder: {len(edf_files)} subjects in '{folder_path}'"
        )

        rows = []
        for fname in edf_files:
            fpath = os.path.join(folder_path, fname)
            try:
                result = self.predict_file(fpath, true_label=true_label)
                rows.append(self._flatten_result(result))
            except Exception as e:
                self._logger.error(f"{fname}: {e}")
                rows.append({
                    "subject_id"         : fname,
                    "true_label"         : true_label,
                    "final_label"        : "ERROR",
                    "stage1_prediction"  : None,
                    "stage2_prediction"  : None,
                    "correct"            : None,
                    "n_windows"          : 0,
                    "error"              : str(e),
                })

        df = pd.DataFrame(rows)

        # Save batch CSV
        out_dir = self._cfg.get("output_dir", "outputs/inference")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "combined_batch_predictions.csv")
        df.to_csv(out_path, index=False)
        self._logger.info(f"Batch results saved → {out_path}")

        # Print summary
        self._logger.info(f"{'='*55}")
        self._logger.info("BATCH SUMMARY")
        for label, count in df["final_label"].value_counts().items():
            self._logger.info(f"  {label:30s}: {count}")
        if true_label is not None:
            acc = df["correct"].mean() * 100
            self._logger.info(f"  Accuracy: {acc:.2f}%")

        return df

    def save_result(self, result: dict, output_dir: str = None) -> str:
        """Save per-window results for a single subject to CSV."""
        out = output_dir or self._cfg.get("output_dir", "outputs/inference")
        os.makedirs(out, exist_ok=True)
        sid  = result["subject_id"].replace(".edf", "")
        path = os.path.join(out, f"{sid}_window_results.csv")
        result["window_results"].to_csv(path, index=False)
        self._logger.info(f"Window results saved → {path}")
        return path

    def model_info(self) -> dict:
        """Return metadata for both loaded models."""
        self._ensure_models_loaded()
        sig = self._cfg["signal"]
        return {
            "stage1_model"    : self._cfg["stage1"]["model_name"],
            "stage1_path"     : self._cfg["stage1"]["model_path"],
            "stage1_classes"  : self._stage1_l2c,
            "stage2_model"    : self._cfg["stage2"]["model_name"],
            "stage2_path"     : self._cfg["stage2"]["model_path"],
            "stage2_classes"  : self._stage2_l2c,
            "n_channels"      : self._stage1_n_channels,
            "n_times"         : self._stage1_n_times,
            "window_sec"      : sig.get("window_sec"),
            "fs"              : sig["fs"],
            "device"          : str(self._device),
        }

    # ----------------------------------------------------------
    # INTERNAL — TWO-STAGE PIPELINE
    # ----------------------------------------------------------

    def _run_pipeline(self, subject_id: str,
                      chunks: list,
                      true_label: str) -> dict:
        """Core two-stage logic."""

        # ── Stage 1: DS vs Control ─────────────────────────────────────
        self._logger.info(f"[Stage 1] DS vs Control  — {len(chunks)} windows")

        s1_preds, s1_probs = self._infer(
            chunks,
            model      = self._stage1_model,
            n_channels = self._stage1_n_channels,
            n_times    = self._stage1_n_times,
        )

        s1_majority    = int(np.bincount(s1_preds).argmax())
        s1_pred_class  = self._stage1_l2c[s1_majority]
        s1_confidence  = float(s1_probs[:, s1_majority].mean())
        s1_votes       = {
            self._stage1_l2c[lbl]: int((s1_preds == lbl).sum())
            for lbl in self._stage1_l2c
        }
        s1_mean_probs  = {
            self._stage1_l2c[c]: round(float(s1_probs[:, c].mean()), 4)
            for c in range(s1_probs.shape[1])
        }

        self._logger.info(
            f"[Stage 1] Prediction : {s1_pred_class}  "
            f"(confidence={s1_confidence:.4f})"
        )
        self._logger.info(f"[Stage 1] Votes       : {s1_votes}")
        self._logger.info(f"[Stage 1] Mean probs  : {s1_mean_probs}")

        # ── Early exit if not DS ───────────────────────────────────────
        if s1_pred_class != "DS":
            final_label = resolve_final_label(s1_pred_class)
            self._logger.info(
                f"[Stage 1] Not DS → pipeline stops. "
                f"Final label: '{final_label}'"
            )

            result = self._build_combined_result(
                subject_id     = subject_id,
                true_label     = true_label,
                final_label    = final_label,
                s1_pred        = s1_pred_class,
                s1_confidence  = s1_confidence,
                s1_votes       = s1_votes,
                s1_mean_probs  = s1_mean_probs,
                s1_preds_arr   = s1_preds,
                s1_probs_arr   = s1_probs,
                s2_pred        = None,
                s2_confidence  = None,
                s2_votes       = None,
                s2_mean_probs  = None,
                s2_preds_arr   = None,
                s2_probs_arr   = None,
            )
            self._log_final(result)
            return result

        # ── Stage 2: DS vs Abnormal ────────────────────────────────────
        self._logger.info(f"[Stage 2] DS vs Abnormal — running on same windows")

        s2_preds, s2_probs = self._infer(
            chunks,
            model      = self._stage2_model,
            n_channels = self._stage2_n_channels,
            n_times    = self._stage2_n_times,
        )

        s2_majority    = int(np.bincount(s2_preds).argmax())
        s2_pred_class  = self._stage2_l2c[s2_majority]
        s2_confidence  = float(s2_probs[:, s2_majority].mean())
        s2_votes       = {
            self._stage2_l2c[lbl]: int((s2_preds == lbl).sum())
            for lbl in self._stage2_l2c
        }
        s2_mean_probs  = {
            self._stage2_l2c[c]: round(float(s2_probs[:, c].mean()), 4)
            for c in range(s2_probs.shape[1])
        }

        self._logger.info(
            f"[Stage 2] Prediction : {s2_pred_class}  "
            f"(confidence={s2_confidence:.4f})"
        )
        self._logger.info(f"[Stage 2] Votes       : {s2_votes}")
        self._logger.info(f"[Stage 2] Mean probs  : {s2_mean_probs}")

        final_label = resolve_final_label(s1_pred_class, s2_pred_class)

        result = self._build_combined_result(
            subject_id    = subject_id,
            true_label    = true_label,
            final_label   = final_label,
            s1_pred       = s1_pred_class,
            s1_confidence = s1_confidence,
            s1_votes      = s1_votes,
            s1_mean_probs = s1_mean_probs,
            s1_preds_arr  = s1_preds,
            s1_probs_arr  = s1_probs,
            s2_pred       = s2_pred_class,
            s2_confidence = s2_confidence,
            s2_votes      = s2_votes,
            s2_mean_probs = s2_mean_probs,
            s2_preds_arr  = s2_preds,
            s2_probs_arr  = s2_probs,
        )
        self._log_final(result)
        return result

    # ----------------------------------------------------------
    # INTERNAL — INFERENCE ENGINE
    # ----------------------------------------------------------

    def _infer(self, chunks: list, model, n_channels: int,
               n_times: int) -> tuple:
        """Normalize + forward pass for one model. Returns (preds, probs)."""
        X      = np.array(chunks)
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(
                     X.reshape(len(X), -1)
                 ).reshape(-1, n_channels, n_times)

        loader = DataLoader(
            _EEGDataset(X_norm),
            batch_size = self._cfg.get("batch_size", 32),
            shuffle    = False,
        )

        all_preds, all_probs = [], []
        with torch.no_grad():
            for batch_x in loader:
                batch_x = batch_x.to(self._device)
                logits  = model(batch_x)
                probs   = torch.softmax(logits, dim=1)
                preds   = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        return np.array(all_preds), np.array(all_probs)

    # ----------------------------------------------------------
    # INTERNAL — RESULT BUILDING
    # ----------------------------------------------------------

    def _build_combined_result(
        self,
        subject_id, true_label, final_label,
        s1_pred, s1_confidence, s1_votes, s1_mean_probs,
        s1_preds_arr, s1_probs_arr,
        s2_pred, s2_confidence, s2_votes, s2_mean_probs,
        s2_preds_arr, s2_probs_arr,
    ) -> dict:

        correct = (final_label == true_label) if true_label is not None else None

        # Per-window DataFrame — stage 1 columns always present,
        # stage 2 columns present only if stage 2 ran
        win_data = {
            "window_idx"       : np.arange(len(s1_preds_arr)),
            "s1_predicted"     : [self._stage1_l2c[p] for p in s1_preds_arr],
        }
        for c in range(s1_probs_arr.shape[1]):
            cname = self._stage1_l2c[c]
            win_data[f"s1_prob_{cname}"] = s1_probs_arr[:, c].round(4)

        if s2_preds_arr is not None:
            win_data["s2_predicted"] = [self._stage2_l2c[p] for p in s2_preds_arr]
            for c in range(s2_probs_arr.shape[1]):
                cname = self._stage2_l2c[c]
                win_data[f"s2_prob_{cname}"] = s2_probs_arr[:, c].round(4)

        window_df = pd.DataFrame(win_data)

        return {
            "subject_id"        : subject_id,
            "true_label"        : true_label,
            "final_label"       : final_label,
            "correct"           : correct,
            "n_windows"         : len(s1_preds_arr),
            "stage1_prediction" : s1_pred,
            "stage1_confidence" : round(s1_confidence, 4),
            "stage1_votes"      : s1_votes,
            "stage1_mean_probs" : s1_mean_probs,
            "stage2_prediction" : s2_pred,
            "stage2_confidence" : round(s2_confidence, 4) if s2_confidence else None,
            "stage2_votes"      : s2_votes,
            "stage2_mean_probs" : s2_mean_probs,
            "window_results"    : window_df,
        }

    def _flatten_result(self, result: dict) -> dict:
        """Flatten a result dict to one CSV row."""
        row = {
            "subject_id"         : result["subject_id"],
            "true_label"         : result["true_label"],
            "final_label"        : result["final_label"],
            "correct"            : result["correct"],
            "n_windows"          : result["n_windows"],
            "stage1_prediction"  : result["stage1_prediction"],
            "stage1_confidence"  : result["stage1_confidence"],
            "stage2_prediction"  : result["stage2_prediction"],
            "stage2_confidence"  : result["stage2_confidence"],
        }
        for k, v in (result["stage1_votes"] or {}).items():
            row[f"s1_votes_{k}"] = v
        for k, v in (result["stage2_votes"] or {}).items():
            row[f"s2_votes_{k}"] = v
        for k, v in (result["stage1_mean_probs"] or {}).items():
            row[f"s1_prob_{k}"] = v
        for k, v in (result["stage2_mean_probs"] or {}).items():
            row[f"s2_prob_{k}"] = v
        return row

    def _log_final(self, result: dict):
        self._logger.info(
            f"FINAL  {result['subject_id']:30s}  →  {result['final_label']}"
            + (f"  (true={result['true_label']}, "
               f"correct={result['correct']})"
               if result["true_label"] else "")
        )

    # ----------------------------------------------------------
    # INTERNAL — MODEL LOADING
    # ----------------------------------------------------------

    def _ensure_models_loaded(self):
        if self._stage1_model is None:
            self._logger.info("Loading Stage 1 model (DS vs Control)...")
            (self._stage1_model,
             self._stage1_n_channels,
             self._stage1_n_times,
             self._stage1_l2c) = self._load_model(
                 model_name = self._cfg["stage1"]["model_name"],
                 model_path = self._cfg["stage1"]["model_path"],
                 classes    = self._cfg["stage1"]["classes"],
             )

        if self._stage2_model is None:
            self._logger.info("Loading Stage 2 model (DS vs Abnormal)...")
            (self._stage2_model,
             self._stage2_n_channels,
             self._stage2_n_times,
             self._stage2_l2c) = self._load_model(
                 model_name = self._cfg["stage2"]["model_name"],
                 model_path = self._cfg["stage2"]["model_path"],
                 classes    = self._cfg["stage2"]["classes"],
             )

        # Resolve window_sec from stage1 checkpoint if not set
        sig = self._cfg["signal"]
        if sig.get("window_sec") is None:
            sig["window_sec"] = self._stage1_n_times / sig["fs"]
            self._logger.info(
                f"window_sec inferred from Stage 1 checkpoint: "
                f"{self._stage1_n_times} / {sig['fs']} = {sig['window_sec']}s"
            )

    def _load_model(self, model_name: str, model_path: str,
                    classes: dict) -> tuple:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint not found: '{model_path}'")

        ckpt       = torch.load(model_path, map_location=self._device)
        n_channels = ckpt["n_channels"]
        n_times    = ckpt["n_times"]
        n_classes  = ckpt.get("n_classes", 2)
        l2c        = {int(k): v for k, v in classes.items()}

        model = self._build_model(model_name, n_channels, n_times, n_classes)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self._device)
        model.eval()

        self._logger.info(
            f"  Loaded '{model_name}'  n_channels={n_channels}  "
            f"n_times={n_times}  "
            f"best_val_acc={ckpt.get('best_val_acc', 'N/A'):.2f}%"
        )
        return model, n_channels, n_times, l2c

    @staticmethod
    def _build_model(model_name, n_channels, n_times, n_classes):
        try:
            from braindecode.models import ShallowFBCSPNet
        except ImportError:
            os.system("pip install braindecode")
            from braindecode.models import ShallowFBCSPNet

        def _build(cls, **kwargs):
            sig = inspect.signature(cls.__init__).parameters
            if "n_chans" in sig:
                kw = {"n_chans"   : kwargs["n_channels"],
                      "n_outputs" : kwargs["n_classes"],
                      "n_times"   : kwargs["n_times"]}
                if "final_conv_length" in kwargs and "final_conv_length" in sig:
                    kw["final_conv_length"] = kwargs["final_conv_length"]
            else:
                kw = {"in_chans"             : kwargs["n_channels"],
                      "n_classes"            : kwargs["n_classes"],
                      "input_window_samples" : kwargs["n_times"]}
                if "final_conv_length" in kwargs and "final_conv_length" in sig:
                    kw["final_conv_length"] = kwargs["final_conv_length"]
            return cls(**kw)

        common = dict(n_channels=n_channels, n_classes=n_classes, n_times=n_times)
        if model_name == "ShallowFBCSPNet":
            return _build(ShallowFBCSPNet, **common, final_conv_length="auto")
        else:
            raise ValueError(f"Unknown model: '{model_name}'")

    # ----------------------------------------------------------
    # INTERNAL — SIGNAL PROCESSING
    # ----------------------------------------------------------

    def _load_edf(self, edf_path: str) -> list:
        try:
            import mne
        except ImportError:
            raise ImportError("mne is required: pip install mne")

        sig = self._cfg["signal"]

        with contextlib.redirect_stdout(StringIO()):
            raw = mne.io.read_raw_edf(edf_path, preload=True)
            raw.pick_channels(sig["channels"])

        return self._window_array(
            raw.get_data(),
            fs         = sig["fs"],
            window_sec = sig["window_sec"],
            overlap    = sig["overlap"],
        )

    @staticmethod
    def _window_array(eeg: np.ndarray, fs: float,
                      window_sec: float, overlap: float) -> list:
        chunk_size = int(window_sec * fs)
        step       = max(int(chunk_size * (1 - overlap)), 1)
        n          = eeg.shape[1]
        return [
            eeg[:, st:st + chunk_size].astype(np.float32)
            for st in range(0, n - chunk_size + 1, step)
        ]

    # ----------------------------------------------------------
    # INTERNAL — UTILITIES
    # ----------------------------------------------------------

    @staticmethod
    def _load_config(path: str) -> dict:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config not found: '{path}'")
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def _resolve_device(s: str):
        if s == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(s)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import sys
    handler = CombinedEEGHandler(path)

    print("\n── Model Info ──────────────────────────────────")
    for k, v in handler.model_info().items():
        print(f"  {k:25s}: {v}")

    if len(sys.argv) > 1:
        edf_path = sys.argv[1]
        result   = handler.predict_file(edf_path)
        print(f"\n  Final label     : {result['final_label']}")
        print(f"  Stage 1         : {result['stage1_prediction']}  "
              f"(conf={result['stage1_confidence']})")
        print(f"  Stage 2         : {result['stage2_prediction']}  "
              f"(conf={result['stage2_confidence']})")
        print(f"  Windows         : {result['n_windows']}")