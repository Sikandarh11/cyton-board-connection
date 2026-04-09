"""
recorder.py
===========
Accumulates raw EEG samples into fixed-length windows and applies the
Cyton channel → electrode name mapping from config_realtime.json.

Key design decisions
--------------------
- The Cyton board returns 8 EEG channels (indices 0-7).
- The model expects 9 channels (F3, Fz, F4, C3, Cz, C4, P3, Pz, P4).
- When ``use_ch9_fallback`` is True in the config, channel 9 (P4-Cz)
  is synthesised as a copy of channel 8 (Pz-Cz).  This lets you run
  the model without a Daisy module.  Replace with real data if available.

Usage
-----
    from recorder import Recorder
    rec = Recorder(cfg["channel_map"], cfg["signal"], cfg["recording"])

    # inside your loop:
    rec.push(eeg_chunk)          # push (8, n) raw samples
    if rec.window_ready():
        window = rec.pop_window()  # (9, window_samples) float32
        ch_names = rec.channel_names  # list[str], length 9
"""

import numpy as np
from collections import deque


# Canonical channel order expected by the model
CANONICAL_ORDER = [
    "EEG F3-Cz",
    "EEG Fz-Cz",
    "EEG F4-Cz",
    "EEG C3-Cz",
    "EEG Cz-Cz",
    "EEG C4-Cz",
    "EEG P3-Cz",
    "EEG Pz-Cz",
    "EEG P4-Cz",
]


class Recorder:
    def __init__(self, channel_map_cfg: dict, signal_cfg: dict, recording_cfg: dict):
        """
        Parameters
        ----------
        channel_map_cfg : dict
            config_realtime.json["channel_map"]
        signal_cfg : dict
            config_realtime.json["signal"]
        recording_cfg : dict
            config_realtime.json["recording"]
        """
        self._fs = signal_cfg["fs"]
        self._window_sec = recording_cfg["window_sec"]
        self._window_samples = int(self._fs * self._window_sec)

        self._use_ch9_fallback = channel_map_cfg.get("use_ch9_fallback", True)

        # Build index map: Cyton channel index (0-based) → electrode name
        # Config keys are 1-based strings
        self._ch_to_name = {
            int(k) - 1: v
            for k, v in channel_map_cfg.items()
            if k.isdigit()
        }

        # Ring buffer: deque of (n_raw_channels,) vectors
        self._buffer: deque = deque()
        self._buffer_len = 0  # current number of samples in buffer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def channel_names(self) -> list:
        """9-element list in canonical model order."""
        return list(CANONICAL_ORDER)

    @property
    def n_channels(self) -> int:
        return len(CANONICAL_ORDER)

    @property
    def window_samples(self) -> int:
        return self._window_samples

    def push(self, eeg: np.ndarray) -> None:
        """
        Append raw board data to the internal buffer.

        Parameters
        ----------
        eeg : np.ndarray, shape (n_raw_channels, n_new_samples)
            Raw data from BoardInterface.read().
        """
        n_samples = eeg.shape[1]
        for i in range(n_samples):
            self._buffer.append(eeg[:, i])
        self._buffer_len += n_samples

    def window_ready(self) -> bool:
        """True when at least one full window is available."""
        return self._buffer_len >= self._window_samples

    def pop_window(self) -> np.ndarray:
        """
        Consume exactly *window_samples* samples from the front of the
        buffer and return them as a (9, window_samples) float32 array
        in canonical channel order.

        Raises
        ------
        RuntimeError if fewer than window_samples are buffered.
        """
        if not self.window_ready():
            raise RuntimeError(
                f"Not enough samples: have {self._buffer_len}, "
                f"need {self._window_samples}."
            )

        # Collect window_samples columns
        cols = [self._buffer.popleft() for _ in range(self._window_samples)]
        self._buffer_len -= self._window_samples

        raw = np.stack(cols, axis=1)  # (n_raw_channels, window_samples)
        return self._remap(raw)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _remap(self, raw: np.ndarray) -> np.ndarray:
        """
        Reorder/synthesise channels so output matches CANONICAL_ORDER.

        Parameters
        ----------
        raw : (n_raw_channels, n_samples)

        Returns
        -------
        (9, n_samples) float32
        """
        n_samples = raw.shape[1]
        out = np.zeros((9, n_samples), dtype=np.float32)

        # Invert the map: name → raw index
        name_to_raw = {v: k for k, v in self._ch_to_name.items()}

        for ch_out, name in enumerate(CANONICAL_ORDER):
            if name in name_to_raw:
                raw_idx = name_to_raw[name]
                if raw_idx < raw.shape[0]:
                    out[ch_out] = raw[raw_idx].astype(np.float32)
                else:
                    out[ch_out] = 0.0
            elif name == "EEG P4-Cz" and self._use_ch9_fallback:
                # Synthesise P4-Cz from Pz-Cz (channel 7, 0-based = 7)
                pz_idx = name_to_raw.get("EEG Pz-Cz")
                if pz_idx is not None and pz_idx < raw.shape[0]:
                    out[ch_out] = raw[pz_idx].astype(np.float32)
            # else: remains zero

        return out
