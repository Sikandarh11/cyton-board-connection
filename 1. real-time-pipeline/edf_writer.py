"""
edf_writer.py
=============
Saves a single (9, n_samples) EEG window to a properly-formed EDF file
with all required metadata (Fs, channel labels, physical units, etc.).

Files are named  record_1.edf, record_2.edf, …  under the configured
output directory.  The counter is persisted in a small sidecar JSON so
resuming a session continues the numbering correctly.

Dependencies
------------
- pyEDFlib  (pip install pyEDFlib)

Usage
-----
    from edf_writer import EDFWriter
    writer = EDFWriter(cfg["recording"], cfg["edf"], cfg["signal"], ch_names)

    path = writer.save(window, record_index=1)
    print("Saved:", path)
"""

import os
import json
import numpy as np


class EDFWriter:
    def __init__(
        self,
        recording_cfg: dict,
        edf_cfg: dict,
        signal_cfg: dict,
        channel_names: list,
    ):
        """
        Parameters
        ----------
        recording_cfg : dict
            config_realtime.json["recording"]
        edf_cfg : dict
            config_realtime.json["edf"]
        signal_cfg : dict
            config_realtime.json["signal"]
        channel_names : list[str]
            9-element list, e.g. ["EEG F3-Cz", …]
        """
        self._out_dir = recording_cfg["output_dir"]
        self._prefix = recording_cfg.get("filename_prefix", "record")
        self._fs = signal_cfg["fs"]
        self._ch_names = channel_names

        self._patient_id = edf_cfg.get("patient_id", "realtime")
        self._recording_id = edf_cfg.get("recording_id", "session")
        self._dig_min = edf_cfg.get("digital_min", -32768)
        self._dig_max = edf_cfg.get("digital_max", 32767)
        self._phys_min = edf_cfg.get("physical_min", -500.0)
        self._phys_max = edf_cfg.get("physical_max", 500.0)
        self._phys_dim = edf_cfg.get("physical_dimension", "uV")

        os.makedirs(self._out_dir, exist_ok=True)

    def save(self, window: np.ndarray, record_index: int) -> str:
        """
        Write one 10-second window to disk.

        Parameters
        ----------
        window : np.ndarray, shape (9, n_samples)
        record_index : int
            1-based counter used in the filename.

        Returns
        -------
        str — absolute path of the saved EDF file.
        """
        try:
            import pyedflib
        except ImportError:
            raise ImportError("pyEDFlib is required: pip install pyEDFlib")

        filename = f"{self._prefix}_{record_index}.edf"
        path = os.path.abspath(os.path.join(self._out_dir, filename))

        n_ch, n_samples = window.shape
        duration_sec = n_samples / self._fs

        f = pyedflib.EdfWriter(path, n_ch, file_type=pyedflib.FILETYPE_EDFPLUS)

        # Global header
        f.setPatientCode(self._patient_id)
        f.setRecordingAdditional(self._recording_id)

        # Per-channel headers
        headers = []
        for name in self._ch_names:
            headers.append({
                "label"           : name,
                "dimension"       : self._phys_dim,
                "sample_frequency": self._fs,
                "physical_min"    : self._phys_min,
                "physical_max"    : self._phys_max,
                "digital_min"     : self._dig_min,
                "digital_max"     : self._dig_max,
                "transducer"      : "Cyton EEG",
                "prefilter"       : "0.5-45 Hz Butterworth; notch",
            })
        f.setSignalHeaders(headers)

        # Write samples channel by channel
        for i in range(n_ch):
            f.writePhysicalSamples(window[i].astype(np.float64))

        f.close()
        return path

    # ------------------------------------------------------------------
    # Counter helpers (persist record index across restarts)
    # ------------------------------------------------------------------

    @staticmethod
    def load_counter(out_dir: str, prefix: str, resume: bool) -> int:
        """
        Return the next record index.
        If resume=True and a counter file exists, continue from where
        we left off.  Otherwise start at 1.
        """
        counter_path = os.path.join(out_dir, f".{prefix}_counter.json")
        if resume and os.path.exists(counter_path):
            with open(counter_path) as fh:
                return json.load(fh).get("next_index", 1)
        return 1

    @staticmethod
    def save_counter(out_dir: str, prefix: str, next_index: int) -> None:
        counter_path = os.path.join(out_dir, f".{prefix}_counter.json")
        with open(counter_path, "w") as fh:
            json.dump({"next_index": next_index}, fh)
