from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque


class EEGVisualizationWithFFT:
    def __init__(self, board_id, params):
        # Board setup
        self.board_id = board_id
        self.params = params
        self.board = BoardShim(board_id, params)

        # Get board info
        self.sampling_rate = BoardShim.get_sampling_rate(board_id)
        self.eeg_channels = BoardShim.get_eeg_channels(board_id)
        self.num_channels = len(self.eeg_channels)

        # Data buffer settings
        self.window_size = 5  # seconds
        self.max_points = int(self.sampling_rate * self.window_size)

        # Initialize deques for each channel
        self.data_buffers = [deque(maxlen=self.max_points) for _ in range(self.num_channels)]
        self.time_buffer = deque(maxlen=self.max_points)

        # Initialize with zeros
        for _ in range(self.max_points):
            self.time_buffer.append(0)
            for buf in self.data_buffers:
                buf.append(0)

        # Adaptive scaling settings
        self.adaptive_scale = True
        self.target_range = 400  # Target ±400 µV (80% of plot range)
        self.scale_factors = [1.0] * self.num_channels
        self.scale_smooth = 0.95

        # Signal quality tracking
        self.signal_stats = [{'mean': 0, 'std': 1} for _ in range(self.num_channels)]

        # Filter settings
        self.use_notch_filter = True
        self.notch_freq = 50.0  # Change to 60.0 for US
        self.bandpass_low = 0.5  # Hz
        self.bandpass_high = 45.0  # Hz

        # FFT settings
        self.fft_window = 2  # seconds of data for FFT
        self.fft_points = int(self.sampling_rate * self.fft_window)

        # EEG frequency bands
        self.freq_bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 45)
        }

        # Setup GUI
        self.setup_gui()

    def setup_gui(self):
        """Create the matplotlib GUI with time-domain and frequency-domain plots"""
        plt.style.use('dark_background')

        # Create figure with 2 columns: time-domain (left) and FFT (right)
        self.fig = plt.figure(figsize=(20, 12))

        # Create grid spec for layout
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(self.num_channels, 2, figure=self.fig,
                               width_ratios=[2, 1], hspace=0.3, wspace=0.25)

        self.fig.canvas.manager.set_window_title('BrainFlow EEG - Time & Frequency Analysis')

        # Color palette
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A',
                  '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2']

        # Band colors for FFT
        band_colors = {
            'Delta': '#9B59B6',
            'Theta': '#3498DB',
            'Alpha': '#2ECC71',
            'Beta': '#F39C12',
            'Gamma': '#E74C3C'
        }

        # Create time-domain and FFT axes for each channel
        self.time_axes = []
        self.fft_axes = []
        self.time_lines = []
        self.fft_lines = []
        self.scale_texts = []
        self.power_texts = []

        for i in range(self.num_channels):
            # Time-domain plot (left column)
            ax_time = self.fig.add_subplot(gs[i, 0])
            self.time_axes.append(ax_time)

            line, = ax_time.plot([], [], color=colors[i % len(colors)],
                                 linewidth=1.2, alpha=0.9, antialiased=True)
            self.time_lines.append(line)

            # Add scale factor text
            scale_text = ax_time.text(0.02, 0.85, '', transform=ax_time.transAxes,
                                      fontsize=8, color='white', alpha=0.7,
                                      bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
            self.scale_texts.append(scale_text)

            # Style time-domain subplot
            pin_label = f'N{i + 1}P'
            ax_time.set_ylabel(f'CH{i + 1}\n{pin_label}\n(µV)',
                               fontsize=9, fontweight='bold', color=colors[i % len(colors)])
            ax_time.set_ylim(-500, 500)
            ax_time.grid(True, alpha=0.15, linestyle='--', linewidth=0.5)
            ax_time.set_facecolor('#0a0a0a')
            ax_time.axhline(y=0, color='white', linestyle='-', linewidth=0.8, alpha=0.4)
            ax_time.set_yticks([-500, -250, 0, 250, 500])

            if i < self.num_channels - 1:
                ax_time.set_xticklabels([])
            else:
                ax_time.set_xlabel('Time (s)', fontsize=10, fontweight='bold')

            # FFT plot (right column)
            ax_fft = self.fig.add_subplot(gs[i, 1])
            self.fft_axes.append(ax_fft)

            fft_line, = ax_fft.plot([], [], color=colors[i % len(colors)],
                                    linewidth=1.5, alpha=0.9)
            self.fft_lines.append(fft_line)

            # Add power text
            power_text = ax_fft.text(0.98, 0.85, '', transform=ax_fft.transAxes,
                                     fontsize=7, color='white', alpha=0.7,
                                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.5),
                                     ha='right', va='top')
            self.power_texts.append(power_text)

            # Style FFT subplot
            ax_fft.set_ylabel('Power (µV²/Hz)', fontsize=8)
            ax_fft.set_xlim(0, 50)  # Show 0-50 Hz
            ax_fft.set_ylim(0, 100)  # Will auto-adjust
            ax_fft.grid(True, alpha=0.15, linestyle='--', linewidth=0.5)
            ax_fft.set_facecolor('#0a0a0a')

            # Add colored bands for EEG frequencies
            for band_name, (low, high) in self.freq_bands.items():
                ax_fft.axvspan(low, high, alpha=0.1, color=band_colors[band_name], label=band_name)

            if i < self.num_channels - 1:
                ax_fft.set_xticklabels([])
            else:
                ax_fft.set_xlabel('Frequency (Hz)', fontsize=10, fontweight='bold')
                # Add legend only to bottom plot
                ax_fft.legend(loc='upper right', fontsize=6, ncol=5, framealpha=0.7)

        # Overall title
        self.fig.suptitle(f'EEG Real-time Analysis | {self.sampling_rate} Hz | Time & Frequency Domain',
                          fontsize=14, fontweight='bold', y=0.995)

    def process_signal(self, data_array, channel_idx):
        """Apply comprehensive signal processing"""
        if len(data_array) < 100:
            return data_array

        signal = np.array(data_array, dtype=np.float64)

        try:
            # Remove DC offset
            DataFilter.detrend(signal, DetrendOperations.LINEAR.value)

            # Apply notch filter
            if self.use_notch_filter:
                DataFilter.remove_environmental_noise(
                    signal,
                    self.sampling_rate,
                    self.notch_freq
                )

            # Apply bandpass filter
            DataFilter.perform_bandpass(
                signal,
                self.sampling_rate,
                center_freq=(self.bandpass_low + self.bandpass_high) / 2,
                band_width=self.bandpass_high - self.bandpass_low,
                order=4,
                filter_type=FilterTypes.BUTTERWORTH.value,
                ripple=0
            )

            # Update signal statistics
            self.signal_stats[channel_idx]['mean'] = np.mean(signal)
            self.signal_stats[channel_idx]['std'] = np.std(signal)

            # Adaptive scaling
            if self.adaptive_scale and self.signal_stats[channel_idx]['std'] > 0.1:
                current_max = np.percentile(np.abs(signal), 98)
                if current_max > 1:
                    target_scale = self.target_range / current_max
                    self.scale_factors[channel_idx] = (
                            self.scale_smooth * self.scale_factors[channel_idx] +
                            (1 - self.scale_smooth) * target_scale
                    )

            # Apply scaling
            signal = signal * self.scale_factors[channel_idx]
            signal = np.clip(signal, -500, 500)

        except Exception as e:
            print(f"⚠️  Processing error on channel {channel_idx}: {e}")
            signal = np.clip(signal - np.mean(signal), -500, 500)

        return signal

    def compute_fft(self, data_array):
        """Compute FFT and return frequency and power spectrum"""
        if len(data_array) < self.fft_points:
            return np.array([]), np.array([])

        # Get last N points for FFT
        signal = np.array(list(data_array)[-self.fft_points:], dtype=np.float64)

        # Remove mean
        signal = signal - np.mean(signal)

        # Apply window to reduce spectral leakage
        window = np.hanning(len(signal))
        signal = signal * window

        # Compute FFT
        fft_data = np.fft.rfft(signal)
        fft_freqs = np.fft.rfftfreq(len(signal), 1.0 / self.sampling_rate)

        # Compute power spectral density (PSD)
        psd = np.abs(fft_data) ** 2 / len(signal)

        # Convert to µV²/Hz
        psd = psd * 2.0  # Account for negative frequencies

        return fft_freqs, psd

    def analyze_frequency_bands(self, freqs, psd):
        """Analyze power in different EEG frequency bands"""
        band_powers = {}

        for band_name, (low, high) in self.freq_bands.items():
            # Find indices for this frequency band
            band_idx = np.where((freqs >= low) & (freqs <= high))[0]
            if len(band_idx) > 0:
                # Calculate average power in this band
                band_powers[band_name] = np.mean(psd[band_idx])
            else:
                band_powers[band_name] = 0

        return band_powers

    def update_plot(self, frame):
        """Animation update function with time and frequency analysis"""
        # Get current board data
        data = self.board.get_current_board_data(50)

        if data.size > 0 and data.shape[1] > 0:
            # Extract EEG channels
            eeg_data = data[self.eeg_channels, :]

            # Get timestamps
            num_new_samples = eeg_data.shape[1]
            current_time = self.time_buffer[-1] if len(self.time_buffer) > 0 else 0
            time_increment = 1.0 / self.sampling_rate

            # Update buffers
            for j in range(num_new_samples):
                current_time += time_increment
                self.time_buffer.append(current_time)

                for i in range(self.num_channels):
                    self.data_buffers[i].append(eeg_data[i, j])

            # Update time-domain plots
            time_array = np.array(self.time_buffer)
            time_array = time_array - time_array[-1]

            max_psd = 1.0  # For auto-scaling FFT plots

            for i in range(self.num_channels):
                # Process and plot time-domain signal
                data_array = list(self.data_buffers[i])
                processed_signal = self.process_signal(data_array, i)

                self.time_lines[i].set_data(time_array, processed_signal)

                scale_info = f"Scale: {1 / self.scale_factors[i]:.1f}× | σ: {self.signal_stats[i]['std']:.1f}µV"
                self.scale_texts[i].set_text(scale_info)

                # Compute and plot FFT
                raw_data = np.array(data_array)  # Use raw data for FFT
                freqs, psd = self.compute_fft(raw_data)

                if len(freqs) > 0:
                    # Only plot up to 50 Hz
                    freq_mask = freqs <= 50
                    plot_freqs = freqs[freq_mask]
                    plot_psd = psd[freq_mask]

                    self.fft_lines[i].set_data(plot_freqs, plot_psd)

                    # Update max for scaling
                    if len(plot_psd) > 0:
                        max_psd = max(max_psd, np.max(plot_psd))

                    # Analyze frequency bands
                    band_powers = self.analyze_frequency_bands(freqs, psd)

                    # Create power text
                    power_info = "Band Power:\n"
                    for band_name in ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']:
                        power = band_powers.get(band_name, 0)
                        power_info += f"{band_name}: {power:.1f}\n"
                    self.power_texts[i].set_text(power_info.strip())

            # Auto-scale FFT y-axes
            for ax_fft in self.fft_axes:
                ax_fft.set_ylim(0, max_psd * 1.1)

            # Update x-axis limits for time plots
            for ax_time in self.time_axes:
                ax_time.set_xlim(time_array[0], time_array[-1])

        return self.time_lines + self.fft_lines + self.scale_texts + self.power_texts

    def start(self):
        """Start the board and GUI"""
        try:
            BoardShim.enable_dev_board_logger()

            print("🔌 Connecting to board on", self.params.serial_port)
            print("⏳ Please wait... (this may take up to 10 seconds)")

            # Try to prepare session with retry
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.board.prepare_session()
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"⚠️  Attempt {attempt + 1} failed, retrying...")
                        import time
                        time.sleep(2)
                    else:
                        raise e

            self.board.start_stream()
            print(f"✅ Streaming started. Sampling rate: {self.sampling_rate} Hz")
            print(f"📺 Displaying {self.num_channels} EEG channels with FFT analysis")
            print(f"\n🎛️  SIGNAL PROCESSING:")
            print(f"   • Detrending: Linear")
            print(f"   • Notch filter: {self.notch_freq} Hz (power line noise)")
            print(f"   • Bandpass: {self.bandpass_low}-{self.bandpass_high} Hz")
            print(f"   • Adaptive scaling: {'Enabled' if self.adaptive_scale else 'Disabled'}")
            print(f"\n📊 FFT ANALYSIS:")
            print(f"   • FFT window: {self.fft_window} seconds")
            print(f"   • Frequency bands: Delta, Theta, Alpha, Beta, Gamma")
            print(f"   • Frequency range: 0-50 Hz")
            print(f"\n💡 INTERPRETING THE FFT:")
            print(f"   • NOISE: Relatively flat spectrum across all frequencies")
            print(f"   • REAL EEG: Clear peaks in specific bands (especially Alpha 8-13 Hz)")
            print(f"   • 50/60 Hz spike = Power line interference (should be filtered)")
            print(f"   • High power in all bands = Movement artifact or loose electrode")

            # Create animation
            self.anim = FuncAnimation(
                self.fig,
                self.update_plot,
                interval=100,  # 100ms = 10 FPS (slower for FFT computation)
                blit=True,
                cache_frame_data=False
            )

            plt.show()

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.stop()

    def stop(self):
        """Stop the board and cleanup"""
        print("\n🛑 Stopping stream...")
        if self.board.is_prepared():
            self.board.stop_stream()
            self.board.release_session()
        print("✅ Session ended")


if __name__ == "__main__":
    # Setup parameters
    params = BrainFlowInputParams()
    params.serial_port = "COM10"  # FTDI USB Cyton dongle
    board_id = BoardIds.CYTON_BOARD.value

    # Create and start visualization
    viz = EEGVisualizationWithFFT(board_id, params)
    viz.start()