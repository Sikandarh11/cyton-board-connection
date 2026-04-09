"""
preprocessor.py
===============
Applies signal processing to a (9, n_samples) EEG window in-place
before passing it to the inference engine.

Processing chain
----------------
1. Linear detrend (removes DC drift and linear trends)
2. Notch filter at 50 Hz (or 60 Hz for US)   — removes power-line noise
3. Butterworth bandpass 0.5–45 Hz             — keeps EEG band

Uses BrainFlow DataFilter to stay consistent with the acquisition stack.

Usage
-----
    from preprocessor import Preprocessor
    pre = Preprocessor(cfg["signal"])
    clean = pre.process(window)   # (9, n_samples) float32 in, float32 out
"""

import numpy as np
from brainflow.data_filter import (
    DataFilter,
    FilterTypes,
    DetrendOperations,
    NoiseTypes,
)


class Preprocessor:
    def __init__(self, signal_cfg: dict):
        """
        Parameters
        ----------
        signal_cfg : dict
            config_realtime.json["signal"]
        """
        self._fs = signal_cfg["fs"]
        self._notch = signal_cfg.get("notch_freq", 50.0)
        self._bp_low = signal_cfg.get("bandpass_low", 0.5)
        self._bp_high = signal_cfg.get("bandpass_high", 45.0)
        self._detrend = signal_cfg.get("detrend", True)

        self._center = (self._bp_low + self._bp_high) / 2.0
        self._width = self._bp_high - self._bp_low

    def process(self, window: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        window : np.ndarray, shape (n_channels, n_samples), float32 or float64

        Returns
        -------
        np.ndarray, same shape, float32
        """
        n_ch = window.shape[0]
        out = window.astype(np.float64, copy=True)

        for i in range(n_ch):
            ch = out[i].copy()  # BrainFlow operates in-place on a copy

            if self._detrend:
                DataFilter.detrend(ch, DetrendOperations.LINEAR.value)

            try:
                noise_type = (
                    NoiseTypes.FIFTY.value
                    if abs(self._notch - 50.0) <= abs(self._notch - 60.0)
                    else NoiseTypes.SIXTY.value
                )
                DataFilter.remove_environmental_noise(ch, self._fs, noise_type)
            except Exception as exc:
                print(f"[Preprocessor] Notch filter failed ch{i}: {exc}")

            try:
                # BrainFlow expects start/stop frequencies, not center/bandwidth.
                DataFilter.perform_bandpass(
                    ch,
                    self._fs,
                    self._bp_low,
                    self._bp_high,
                    4,
                    FilterTypes.BUTTERWORTH.value,
                    0,
                )
            except Exception as exc:
                print(f"[Preprocessor] Bandpass failed ch{i}: {exc}")

            out[i] = ch

        return out.astype(np.float32)
