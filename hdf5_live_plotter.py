import sys
import time
import numpy as np
import h5py
import logging
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
from picosdk.functions import adc2mV
from picosdk.ps5000a import ps5000a as ps


class HDF5LivePlotter(QMainWindow):
    """
    Real-time oscilloscope-style plotter that reads from HDF5 files.
    Completely independent of acquisition system for zero-risk operation.
    """

    def __init__(self, hdf5_path="/tmp/data.hdf5", update_interval_ms=200, debug=False):
        super().__init__()

        # Configuration
        self.hdf5_path = hdf5_path
        self.update_interval_ms = update_interval_ms
        self.display_window_samples = 15_000_000  # 1 second at 15MS/s
        self.decimation_factor = 150  # 15M -> 100k display points
        self.debug = debug

        # Setup logging
        if self.debug:
            logging.basicConfig(
                level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
            )
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.WARNING)

        # Data storage
        self.display_data = np.array([])
        self.time_data = np.array([])
        self.data_start_sample = 0  # Track where our display window starts in the file

        # Metadata from HDF5
        self.sample_interval_ns = 16  # Default, will be read from file
        self.ch_range = None
        self.max_adc = None

        # Debug counters
        self.update_count = 0
        self.file_read_count = 0
        self.display_update_count = 0

        # Setup UI
        self.setup_ui()

        # Setup update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_from_file)
        self.timer.start(self.update_interval_ms)

        self.logger.info(
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
        self.status_label = QLabel("Status: Waiting for data...")
        self.samples_label = QLabel("Samples: 0")
        self.rate_label = QLabel("Rate: 0 MS/s")
        self.acq_status_label = QLabel("Acquisition: Starting...")
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.samples_label)
        status_layout.addWidget(self.rate_label)
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
        except Exception as e:
            print(f"Warning: Could not read metadata: {e}")

    def update_from_file(self):
        """Periodically read the latest window of data from the HDF5 file."""
        self.update_count += 1
        try:
            with h5py.File(self.hdf5_path, "r") as f:
                if "adc_counts" not in f:
                    return

                dataset = f["adc_counts"]
                current_size = dataset.shape[0]

                if current_size == 0:
                    return

                # Read metadata if not already done
                if self.ch_range is None:
                    self.read_metadata(f)

                # Calculate where to start reading to get the last window
                start_index = max(0, current_size - self.display_window_samples)
                self.data_start_sample = start_index

                # Read only the most recent data window
                data_window = dataset[start_index:current_size]
                self.file_read_count += 1

                self.logger.debug(
                    f"Update {self.update_count}: Reading window of {len(data_window):,} samples from index {start_index:,}"
                )

                # Update the display with this complete window
                self.update_display(data_window)

                # Update status labels
                self.samples_label.setText(f"Samples: {current_size:,}")
                elapsed_time = time.time() - getattr(self, "start_time", time.time())
                if not hasattr(self, "start_time"):
                    self.start_time = time.time()
                rate_ms = (
                    (current_size / elapsed_time / 1_000_000)
                    if elapsed_time > 0
                    else 0
                )
                self.rate_label.setText(f"Rate: {rate_ms:.1f} MS/s")
                self.status_label.setText(
                    f"Status: Live streaming (Updates: {self.file_read_count})"
                )
                self.acq_status_label.setText("Acquisition: Active")

        except (FileNotFoundError, OSError):
            self.status_label.setText(f"Status: Waiting for {self.hdf5_path}")
        except Exception as e:
            self.logger.error(f"Update {self.update_count}: Error reading file - {e}")
            self.status_label.setText(f"Status: Error reading file - {str(e)}")

    def update_display(self, data_window):
        """Update the oscilloscope display with a full window of data."""
        if len(data_window) == 0:
            return

        self.display_data = data_window
        self.display_update_count += 1

        self.logger.debug(
            f"Display update {self.display_update_count}: "
            f"Displaying window of {len(self.display_data):,} samples, "
            f"starting at sample {self.data_start_sample:,}"
        )

        # Apply decimation for display
        decimated_data = self.min_max_decimate(
            self.display_data, self.decimation_factor
        )

        # Convert to voltage if we have calibration data
        if self.ch_range is not None and self.max_adc is not None:
            try:
                voltage_data = adc2mV(decimated_data, self.ch_range, self.max_adc)
            except Exception:
                voltage_data = decimated_data.astype(float)  # Fallback
        else:
            voltage_data = decimated_data.astype(float)

        # Create time axis for the current window
        time_axis = self.create_time_axis(len(voltage_data))

        self.logger.debug(
            f"Display update {self.display_update_count}: "
            f"Decimated to {len(voltage_data):,} points, "
            f"time range: {time_axis[0]:.3f}s to {time_axis[-1]:.3f}s"
        )

        # Update plot
        self.curve.setData(time_axis, voltage_data)

        # Auto-scale occasionally
        if self.display_update_count % 10 == 1:
            self.plot_widget.autoRange()

    def min_max_decimate(self, data, factor):
        """
        Min-max decimation to preserve transients while reducing data points.
        For each group of 'factor' samples, keep both min and max values.
        """
        if len(data) < factor:
            return data

        # Reshape data into groups
        n_complete_groups = len(data) // factor
        if n_complete_groups == 0:
            return data

        # Take only complete groups
        grouped_data = data[: n_complete_groups * factor].reshape(-1, factor)

        # Get min and max for each group
        mins = np.min(grouped_data, axis=1)
        maxs = np.max(grouped_data, axis=1)

        # Interleave mins and maxs to preserve transients
        decimated = np.empty(n_complete_groups * 2, dtype=data.dtype)
        decimated[0::2] = mins
        decimated[1::2] = maxs

        # Add any remaining samples
        remainder = data[n_complete_groups * factor :]
        if len(remainder) > 0:
            decimated = np.concatenate([decimated, remainder])

        return decimated

    def create_time_axis(self, n_samples):
        """Create a scrolling time axis in seconds for the oscilloscope view."""
        time_per_sample = self.sample_interval_ns * 1e-9

        # Calculate the absolute start and end time of the current display window
        start_time = self.data_start_sample * time_per_sample
        end_time = (self.data_start_sample + len(self.display_data)) * time_per_sample

        # Create a time axis that spans this window for the decimated data points
        time_axis = np.linspace(start_time, end_time, n_samples)

        self.logger.debug(
            f"Time axis: {start_time:.3f}s to {end_time:.3f}s, "
            f"samples={n_samples}, "
            f"display_buffer_size={len(self.display_data):,}"
        )

        return time_axis

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
