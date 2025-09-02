import sys
import time
import warnings
import numpy as np
import h5py
from loguru import logger
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QLabel,
    QHBoxLayout,
)
from PyQt5.QtCore import QTimer, pyqtSignal, QObject
import pyqtgraph as pg
from conversion_utils import adc_to_mV, min_max_decimate_numba


class HDF5LivePlotter(QMainWindow):
    """
    Real-time oscilloscope-style plotter that reads from HDF5 files.
    Completely independent of acquisition system for zero-risk operation.
    """

    def __init__(self, hdf5_path="/tmp/data.hdf5", update_interval_ms=50):
        super().__init__()

        # Configuration
        self.hdf5_path = hdf5_path
        self.update_interval_ms = update_interval_ms
        self.display_window_samples = 30_000_000  # ~0.5 seconds at 62.5MS/s
        self.decimation_factor = 150  # 15M -> 100k display points

        # UI Heartbeat
        self.heartbeat_chars = ["|", "/", "-", "\\"]
        self.heartbeat_index = 0

        # Data storage
        self.display_data = np.array([])
        self.time_data = np.array([])
        self.data_start_sample = 0  # Track where our display window starts in the file

        # Metadata from HDF5
        self.sample_interval_ns = 16  # Default, will be read from file
        self.ch_range = None
        self.max_adc = None
        self.voltage_range_v = None

        # Debug counters
        self.update_count = 0
        self.file_read_count = 0
        self.display_update_count = 0

        # Performance monitoring
        self.last_file_size = 0
        self.last_rate_time = time.time()
        self.data_rate_mb_s = 0.0
        self.display_latency_ms = 0.0
        self.last_data_timestamp = None
        
        # Data freshness tracking
        self.last_displayed_size = 0
        self.data_change_count = 0
        self.stale_update_count = 0
        self.last_freshness_check = time.time()
        
        # Error tracking
        self.conversion_error_count = 0
        self.file_error_count = 0

        # Setup UI
        self.setup_ui()

        # Setup update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_from_file)
        self.timer.start(self.update_interval_ms)

        logger.info(
            f"HDF5LivePlotter initialized: path={hdf5_path}, interval={update_interval_ms}ms"
        )

        # Initial file check
        self.check_file_exists()

    def setup_ui(self):
        """Setup the oscilloscope-style UI"""
        self.setWindowTitle("PicoScope Live Plotter - HDF5 Reader")
        self.setGeometry(100, 100, 1200, 800)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Status bar
        status_layout = QHBoxLayout()
        self.heartbeat_label = QLabel("UI: -")
        self.status_label = QLabel("Status: Waiting for data...")
        self.samples_label = QLabel("Samples: 0")
        self.rate_label = QLabel("Rate: 0 MS/s")
        self.data_rate_label = QLabel("Data: 0 MB/s")
        self.latency_label = QLabel("Latency: 0 ms")
        self.error_label = QLabel("Errors: 0")
        self.acq_status_label = QLabel("Acquisition: Starting...")
        
        # Add separators between status items
        status_layout.addWidget(self.heartbeat_label)
        status_layout.addWidget(QLabel(" | "))
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(QLabel(" | "))
        status_layout.addWidget(self.samples_label)
        status_layout.addWidget(QLabel(" | "))
        status_layout.addWidget(self.rate_label)
        status_layout.addWidget(QLabel(" | "))
        status_layout.addWidget(self.data_rate_label)
        status_layout.addWidget(QLabel(" | "))
        status_layout.addWidget(self.latency_label)
        status_layout.addWidget(QLabel(" | "))
        status_layout.addWidget(self.error_label)
        status_layout.addWidget(QLabel(" | "))
        status_layout.addWidget(self.acq_status_label)
        status_layout.addStretch()
        layout.addLayout(status_layout)

        # Plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("left", "Voltage", "mV")
        self.plot_widget.setLabel("bottom", "Time", "s")
        self.plot_widget.setTitle("Channel A - Live Data")
        self.plot_widget.showGrid(x=True, y=True)

        # Plot curve
        self.curve = self.plot_widget.plot(pen="y", width=1)

        layout.addWidget(self.plot_widget)

        # Performance optimization
        self.plot_widget.setDownsampling(mode="peak")
        self.plot_widget.setClipToView(True)

    def check_file_exists(self):
        """Check if HDF5 file exists and is readable"""
        try:
            with h5py.File(self.hdf5_path, "r") as f:
                if "adc_counts" in f:
                    self.status_label.setText("Status: File found, reading...")
                    self.read_metadata(f)
                else:
                    self.status_label.setText("Status: File exists but no data yet")
        except (FileNotFoundError, OSError):
            self.status_label.setText(f"Status: Waiting for {self.hdf5_path}")

    def read_metadata(self, hdf5_file):
        """Read metadata from HDF5 file"""
        try:
            if "metadata" in hdf5_file:
                metadata = hdf5_file["metadata"]
                self.sample_interval_ns = metadata.attrs.get("timeIntervalns", 16)
                self.ch_range = metadata.attrs.get("chARange", None)
                self.max_adc = metadata.attrs.get("maxADC", None)
                self.voltage_range_v = metadata.attrs.get("chAVoltageRange", 20.0)
        except Exception as e:
            logger.warning(f"Could not read metadata: {e}")

    def update_from_file(self):
        """Periodically read the latest window of data from the HDF5 file."""
        self.update_count += 1

        # Update UI heartbeat to show the UI thread is alive
        self.heartbeat_index = (self.heartbeat_index + 1) % len(self.heartbeat_chars)
        self.heartbeat_label.setText(f"UI: {self.heartbeat_chars[self.heartbeat_index]}")

        try:
            with h5py.File(self.hdf5_path, "r") as f:
                if "adc_counts" not in f:
                    return

                dataset = f["adc_counts"]
                current_size = dataset.shape[0]

                if current_size == 0:
                    return

                # Read metadata if not already done
                if self.voltage_range_v is None:
                    self.read_metadata(f)

                # Calculate where to start reading to get the last window
                start_index = max(0, current_size - self.display_window_samples)
                self.data_start_sample = start_index

                # Calculate data rate
                current_time = time.time()
                current_file_size = current_size * 2  # 2 bytes per int16 sample
                time_delta = current_time - self.last_rate_time
                
                if time_delta >= 1.0:  # Update rate every second
                    size_delta = current_file_size - self.last_file_size
                    self.data_rate_mb_s = (size_delta / (1024 * 1024)) / time_delta
                    self.last_file_size = current_file_size
                    self.last_rate_time = current_time

                # Track data freshness and changes
                if current_size > self.last_displayed_size:
                    # NEW DATA - update plot
                    self.data_change_count += 1
                    self.last_displayed_size = current_size
                    self.last_data_timestamp = current_time
                    self.acq_status_label.setText("Acquisition: Active")

                    # Read only the most recent data window
                    data_window = dataset[start_index:current_size]
                    self.file_read_count += 1

                    logger.debug(
                        f"Update {self.update_count}: Reading window of {len(data_window):,} samples from index {start_index:,}"
                    )

                    # Update the display with this complete window (only when data changes)
                    self.update_display(data_window)

                else:
                    # NO NEW DATA - skip expensive plot update
                    self.stale_update_count += 1
                    self.acq_status_label.setText("Acquisition: Acquiring...")
                    # Log if we're frequently updating with no new data
                    if self.stale_update_count % 10 == 0:
                        logger.debug(f"File check #{self.update_count} with no new data (stale checks: {self.stale_update_count})")

                    # Check data staleness
                    if self.last_data_timestamp:
                        data_age_ms = (current_time - self.last_data_timestamp) * 1000
                        if data_age_ms > 500:  # Data older than 500ms
                            logger.warning(f"Displaying stale data: {data_age_ms:.0f}ms old")

                # Update status labels with abbreviations
                samples_text = self.format_sample_count(current_size)
                self.samples_label.setText(f"Samples: {samples_text}")
                
                elapsed_time = time.time() - getattr(self, "start_time", time.time())
                if not hasattr(self, "start_time"):
                    self.start_time = time.time()
                rate_ms = (
                    (current_size / elapsed_time / 1_000_000)
                    if elapsed_time > 0
                    else 0
                )
                self.rate_label.setText(f"Rate: {rate_ms:.1f} MS/s")
                self.data_rate_label.setText(f"Data: {self.data_rate_mb_s:.1f} MB/s")
                
                # Color-code latency: Green < 100ms, Yellow < 500ms, Red >= 500ms
                latency_color = "green" if self.display_latency_ms < 100 else "orange" if self.display_latency_ms < 500 else "red"
                self.latency_label.setText(f'<span style="color: {latency_color}">Latency: {self.display_latency_ms:.0f}ms</span>')
                
                # Error counter with color coding
                total_errors = self.conversion_error_count + self.file_error_count
                error_color = "green" if total_errors == 0 else "orange" if total_errors < 10 else "red"
                self.error_label.setText(f'<span style="color: {error_color}">Errors: {total_errors}</span>')
                
                self.status_label.setText(f"Status: Live (R:{self.file_read_count})")

        except (FileNotFoundError, OSError):
            self.status_label.setText(f"Status: Waiting for file")
        except Exception as e:
            self.file_error_count += 1
            logger.error(f"Update {self.update_count}: Error reading file - {e}")
            self.status_label.setText(f"Status: File error ({self.file_error_count})")

    def update_display(self, data_window):
        """Update the oscilloscope display with a full window of data."""
        if len(data_window) == 0:
            return

        self.display_data = data_window
        self.display_update_count += 1

        logger.debug(
            f"Display update {self.display_update_count}: "
            f"Displaying window of {len(self.display_data):,} samples, "
            f"starting at sample {self.data_start_sample:,}"
        )

        # Apply Numba-optimized decimation for display
        decimated_data = min_max_decimate_numba(
            self.display_data, self.decimation_factor
        )

        # Debug: Log ADC values and metadata
        logger.debug(f"ADC range: {decimated_data.min()} to {decimated_data.max()}")
        logger.debug(f"voltage_range_v: {self.voltage_range_v}, max_adc: {self.max_adc}")

        # Convert to voltage if we have calibration data
        if self.voltage_range_v is not None and self.max_adc is not None:
            try:
                voltage_data = adc_to_mV(decimated_data, self.voltage_range_v, self.max_adc)
                logger.debug(f"Voltage conversion successful, range: {voltage_data.min():.1f} to {voltage_data.max():.1f} mV")
            except Exception as e:
                self.conversion_error_count += 1
                logger.warning(f"Voltage conversion failed: {e}, using raw ADC values")
                voltage_data = decimated_data.astype(float)
        else:
            logger.warning("Missing calibration data (voltage_range_v or max_adc), using raw ADC values")
            voltage_data = decimated_data.astype(float)

        # Create time axis for the current window
        time_axis = self.create_time_axis(len(voltage_data))

        logger.debug(
            f"Display update {self.display_update_count}: "
            f"Decimated to {len(voltage_data):,} points, "
            f"time range: {time_axis[0]:.3f}s to {time_axis[-1]:.3f}s"
        )

        # Calculate display latency
        if self.last_data_timestamp:
            self.display_latency_ms = (time.time() - self.last_data_timestamp) * 1000

        # Update plot
        self.curve.setData(time_axis, voltage_data)

        # Update plot title with counters for feedback
        title = f"Channel A - File reads: {self.file_read_count}, Plot updates: {self.display_update_count}"
        self.plot_widget.setTitle(title)

        # Manually set the X-axis range to follow the data, creating a scroll effect.
        self.plot_widget.setXRange(time_axis[0], time_axis[-1], padding=0)

        # Auto-scale the Y-axis occasionally.
        if self.display_update_count % 10 == 1:
            self.plot_widget.enableAutoRange(axis='y')


    def format_sample_count(self, count):
        """Format large sample counts with appropriate units"""
        if count >= 1_000_000_000:
            return f"{count / 1_000_000_000:.1f}G"
        elif count >= 1_000_000:
            return f"{count / 1_000_000:.1f}M"
        elif count >= 1_000:
            return f"{count / 1_000:.1f}K"
        else:
            return str(count)

    def create_time_axis(self, n_samples):
        """
        Create a simplified, linearly spaced time axis for the display window.
        This is an approximation but is much simpler than calculating exact times
        for min-max decimated points.
        """
        time_per_sample = self.sample_interval_ns * 1e-9
        start_time = self.data_start_sample * time_per_sample
        # The end time is based on the original number of samples in the window
        end_time = (self.data_start_sample + len(self.display_data)) * time_per_sample
        return np.linspace(start_time, end_time, n_samples)

    def closeEvent(self, event):
        """Clean shutdown"""
        self.timer.stop()
        event.accept()


def main():
    """Standalone application entry point"""
    app = QApplication(sys.argv)

    # Command line argument for HDF5 file path
    hdf5_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/data.hdf5"

    plotter = HDF5LivePlotter(hdf5_path)
    plotter.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
