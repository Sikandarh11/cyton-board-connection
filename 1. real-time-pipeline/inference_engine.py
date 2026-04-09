"""
inference_engine.py
===================
Thin adapter between the real-time pipeline and CombinedEEGHandler.

Responsibilities
----------------
- Load the handler once at startup (models are heavy).
- Accept a (9, n_samples) numpy array.
- Return the standard result dict used by the rest of the pipeline.

Usage
-----
    from inference_engine import InferenceEngine
    engine = InferenceEngine(cfg["model"])
    result = engine.predict(window, subject_id="record_1")
    print(result["final_label"])
"""

import sys
import os
import numpy as np


class InferenceEngine:
    def __init__(self, model_cfg: dict):
        """
        Parameters
        ----------
        model_cfg : dict
            config_realtime.json["model"]
        """
        config_path = model_cfg["config_path"]

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"[InferenceEngine] CombinedEEGHandler config not found: '{config_path}'\n"
                "Check config_realtime.json → model.config_path"
            )

        # Lazy import — keeps startup fast when you're testing other modules
        from handler_combined import CombinedEEGHandler

        print(f"[InferenceEngine] Loading models from {config_path} ...")
        self._handler = CombinedEEGHandler(config_path)
        print("[InferenceEngine] Models ready.")

        info = self._handler.model_info()
        print(f"[InferenceEngine] Stage 1: {info['stage1_model']}")
        print(f"[InferenceEngine] Stage 2: {info['stage2_model']}")
        print(f"[InferenceEngine] Device : {info['device']}")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def predict(self, window: np.ndarray, subject_id: str = "realtime") -> dict:
        """
        Run two-stage inference on a single EEG window.

        Parameters
        ----------
        window : np.ndarray, shape (9, n_samples), float32
        subject_id : str
            Identifier written into the result dict (e.g. "record_3").

        Returns
        -------
        dict — same structure as CombinedEEGHandler.predict_array():
            {
              "subject_id"        : str,
              "final_label"       : str,   # "DS" / "Control (Non-DS)" / "Abnormal (Non-DS)"
              "n_windows"         : int,   # sub-windows inside the 10 s chunk
              "stage1_prediction" : str,
              "stage1_confidence" : float,
              "stage1_votes"      : dict,
              "stage1_mean_probs" : dict,
              "stage2_prediction" : str | None,
              "stage2_confidence" : float | None,
              "stage2_votes"      : dict | None,
              "stage2_mean_probs" : dict | None,
              "window_results"    : pd.DataFrame,
            }
        """
        return self._handler.predict_array(window, subject_id=subject_id)

    def model_info(self) -> dict:
        return self._handler.model_info()
