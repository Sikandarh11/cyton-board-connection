"""
board_interface.py
==================
Thin wrapper around BrainFlow for the Cyton board.

Responsibilities
----------------
- Connect / disconnect
- Start / stop streaming
- Return a numpy array of shape (n_channels, n_samples) for the
  latest N samples

Usage
-----
    from board_interface import BoardInterface
    bi = BoardInterface(cfg["board"])
    bi.connect()
    raw = bi.read(n_samples=2500)   # (n_eeg_channels, 2500)
    bi.disconnect()
"""

import time
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams


class BoardInterface:
    def __init__(self, board_cfg: dict):
        """
        Parameters
        ----------
        board_cfg : dict
            Sub-dict from config_realtime.json["board"].
        """
        self._cfg = board_cfg
        board_id = board_cfg["board_id"]

        params = BrainFlowInputParams()
        params.serial_port = board_cfg["serial_port"]
        # Timeout 0 matches the behavior of the known-good manual diagnose flow.
        params.timeout = board_cfg.get("timeout", 0)

        self._params = params
        self._board = None
        self._board_id = board_id
        self.sampling_rate = BoardShim.get_sampling_rate(board_id)
        self.eeg_channels = BoardShim.get_eeg_channels(board_id)

        self._streaming = False

    def _wake_board_serial(self) -> None:
        """Best-effort serial wake probe before BrainFlow handshake."""
        try:
            import serial
        except Exception:
            return

        port = self._cfg["serial_port"]
        baud = int(self._cfg.get("baud_rate", 115200))

        try:
            with serial.Serial(port=port, baudrate=baud, timeout=1.0) as ser:
                # Give USB-serial stack a moment after opening the port.
                time.sleep(1.0)
                ser.reset_input_buffer()
                ser.reset_output_buffer()

                # First request stop, then request version banner.
                ser.write(b"s")
                ser.flush()
                time.sleep(0.15)
                ser.write(b"v")
                ser.flush()

                deadline = time.time() + 4.0
                chunks = []
                while time.time() < deadline:
                    waiting = ser.in_waiting
                    if waiting > 0:
                        chunks.append(ser.read(waiting))
                    time.sleep(0.05)

                # Ensure board is left in stop state before BrainFlow takes over.
                ser.write(b"s")
                ser.flush()

                response = b"".join(chunks).decode(errors="ignore")
                tokens = ["OpenBCI", "ADS1299", "Firmware", "Cyton"]
                if any(t in response for t in tokens):
                    print("[BoardInterface] Serial wake probe: board response detected.")
                else:
                    print("[BoardInterface] Serial wake probe: no board response yet.")
        except Exception as exc:
            print(f"[BoardInterface] Serial wake probe skipped: {exc}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Prepare session and start data stream."""
        BoardShim.enable_dev_board_logger()
        print(f"[BoardInterface] Connecting on {self._cfg['serial_port']} ...")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                self._wake_board_serial()
                time.sleep(0.2)

                # Recreate BoardShim on each retry to avoid stale state after failed attempts.
                self._board = BoardShim(self._board_id, self._params)
                self._board.prepare_session()
                break
            except Exception as exc:
                # Best-effort cleanup before retrying.
                try:
                    if self._board is not None and self._board.is_prepared():
                        self._board.release_session()
                except Exception:
                    pass
                self._board = None

                if attempt < max_retries - 1:
                    print(f"[BoardInterface] Attempt {attempt + 1} failed - retrying in 2 s")
                    time.sleep(2)
                else:
                    raise RuntimeError(
                        f"Could not prepare BrainFlow session after {max_retries} attempts: {exc}"
                    ) from exc

        self._board.start_stream()
        self._streaming = True
        print(f"[BoardInterface] Streaming at {self.sampling_rate} Hz  "
              f"({len(self.eeg_channels)} EEG channels)")

    def disconnect(self) -> None:
        """Stop stream and release session."""
        if self._board is not None:
            try:
                if self._board.is_prepared():
                    if self._streaming:
                        self._board.stop_stream()
                        self._streaming = False
                    self._board.release_session()
            except Exception:
                pass
            finally:
                self._board = None
                self._streaming = False
        print("[BoardInterface] Disconnected.")

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def read(self, n_samples: int) -> np.ndarray:
        """
        Return the latest *n_samples* samples for all EEG channels.

        Returns
        -------
        np.ndarray of shape (n_eeg_channels, n_samples)
            Contains whatever data are in the ring buffer — may be zeros
            if fewer than n_samples have arrived yet.
        """
        if not self._streaming:
            raise RuntimeError("Board is not streaming. Call connect() first.")
        if self._board is None:
            raise RuntimeError("Board object is not initialised. Call connect() first.")

        raw = self._board.get_current_board_data(n_samples)
        # raw shape: (all_channels, actual_samples)
        eeg = raw[self.eeg_channels, :]

        # Pad with zeros on the left if not enough samples yet
        if eeg.shape[1] < n_samples:
            pad = np.zeros((eeg.shape[0], n_samples - eeg.shape[1]), dtype=np.float64)
            eeg = np.concatenate([pad, eeg], axis=1)

        return eeg.astype(np.float64)

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()
