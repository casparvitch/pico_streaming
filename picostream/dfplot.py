from __future__ import annotations

import sys
import time
from typing import List, Optional

import click
import h5py
import numpy as np
import pyqtgraph as pg
from loguru import logger
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QCloseEvent, QFont, QKeyEvent
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from .conversion_utils import adc_to_mV, min_max_decimate_numba


class HDF5LivePlotter(QMainWindow):
    """
    Real-time oscilloscope-style plotter that reads from HDF5 files.
    Completely independent of acquisition system for zero-risk operation.
    """

    def __init__(
        self,
        hdf5_path: str = "/tmp/data.hdf5",
        update_interval_ms: int = 50,
        display_window_seconds: float = 0.5,
        decimation_factor: int = 150,
    ) -> None:
        """Initializes the HDF5LivePlotter window.

        Args:
            hdf5_path: Path to the HDF5 file to monitor.
            update_interval_ms: How often to check the file for updates (in ms).
            display_window_seconds: The time duration of data to display.
            decimation_factor: The factor by which to decimate data for plotting.
        """
        super().__init__()

        # --- Configuration ---
        self.hdf5_path: str = hdf5_path
        self.update_interval_ms: int = update_interval_ms
        self.display_window_seconds: float = display_window_seconds
        self.decimation_factor: int = decimation_factor

        # --- UI State ---
        self.heartbeat_chars: List[str] = ["|", "/", "-", "\\"]
        self.heartbeat_index: int = 0

        # --- Data Buffers ---
        self.display_data: np.ndarray = np.array([])
        self.time_data: np.ndarray = np.array([])
        self.data_start_sample: int = 0

        # --- HDF5 Metadata ---
        self.sample_interval_ns: float = 16.0  # Default, will be read from file
        self.hardware_downsample_ratio: int = 1
        self.ch_range: Optional[int] = None
        self.max_adc: Optional[int] = None
        self.voltage_range_v: Optional[float] = None
        self.downsample_mode: Optional[str] = None

        # --- Debug Counters ---
        self.update_count: int = 0
        self.file_read_count: int = 0
        self.display_update_count: int = 0

        # --- Performance Monitoring ---
        self.display_latency_ms: float = 0.0
        self.last_data_timestamp: Optional[float] = None

        # --- Rate Checking ---
        self.rate_check_start_time: Optional[float] = None
        self.rate_check_start_samples: int = 0

        # --- Data Freshness Tracking ---
        self.last_displayed_size: int = 0
        self.data_change_count: int = 0
        self.stale_update_count: int = 0
        self.last_freshness_check: float = time.time()

        # --- Error Tracking ---
        self.conversion_error_count: int = 0
        self.file_error_count: int = 0

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

    def setup_ui(self) -> None:
        """Sets up the main window, widgets, and plot layout."""
        self.setWindowTitle("PicoScope Live Plotter - HDF5 Reader")
        self.setGeometry(100, 100, 1200, 800)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Status bar
        status_layout = QHBoxLayout()
        self.heartbeat_label = QLabel("UI: -")
        self.samples_label = QLabel("Samples: 0")
        self.rate_label = QLabel("Rate: -")
        self.plotter_latency_label = QLabel("Plotter Latency: 0 ms")
        self.error_label = QLabel("Errors: 0")
        self.acq_status_label = QLabel(
            '<span style="color: orange">Waiting for file...</span>'
        )
        font = QFont()
        font.setFamily("Monospace")
        font.setFixedPitch(True)
        for label in [
            self.heartbeat_label,
            self.samples_label,
            self.rate_label,
            self.plotter_latency_label,
            self.error_label,
            self.acq_status_label,
        ]:
            label.setFont(font)

        # Add separators between status items
        status_layout.addWidget(self.heartbeat_label)
        status_layout.addWidget(QLabel(" | "))
        status_layout.addWidget(self.error_label)
        status_layout.addWidget(QLabel(" | "))
        status_layout.addWidget(self.samples_label)
        status_layout.addWidget(QLabel(" | "))
        status_layout.addWidget(self.plotter_latency_label)
        status_layout.addWidget(QLabel(" | "))
        status_layout.addWidget(self.rate_label)
        status_layout.addWidget(QLabel(" | "))
        status_layout.addWidget(self.acq_status_label)
        status_layout.addStretch()
        layout.addLayout(status_layout)

        # Plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("left", "Voltage", "mV")
        self.plot_widget.setLabel("bottom", "Time", "s")
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setXRange(0, self.display_window_seconds, padding=0)

        # Plot curve
        self.curve = self.plot_widget.plot(pen="y", width=1)

        layout.addWidget(self.plot_widget)

        # Performance optimization
        self.plot_widget.setDownsampling(mode="peak")
        self.plot_widget.setClipToView(True)

    def check_file_exists(self) -> None:
        """Checks if the HDF5 file exists and attempts to read metadata."""
        try:
            with h5py.File(self.hdf5_path, "r") as f:
                if "adc_counts" in f:
                    self.acq_status_label.setText(
                        '<span style="color: orange">Reading metadata...</span>'
                    )
                    self.read_metadata(f)
                else:
                    self.acq_status_label.setText(
                        '<span style="color: orange">Waiting for data...</span>'
                    )
        except (FileNotFoundError, OSError):
            self.acq_status_label.setText(
                '<span style="color: orange">Waiting for file...</span>'
            )

    def read_metadata(self, hdf5_file: h5py.File) -> None:
        """Reads metadata attributes from the root of an open HDF5 file.

        Args:
            hdf5_file: An open h5py.File object.
        """
        try:
            # Metadata is stored as root-level attributes
            base_sample_interval_ns = hdf5_file.attrs["sample_interval_ns"]
            self.hardware_downsample_ratio = hdf5_file.attrs.get(
                "hardware_downsample_ratio", 1
            )
            self.sample_interval_ns = (
                base_sample_interval_ns * self.hardware_downsample_ratio
            )

            self.max_adc = hdf5_file.attrs["max_adc"]
            self.voltage_range_v = hdf5_file.attrs["voltage_range_v"]
            self.downsample_mode = hdf5_file.attrs.get("downsample_mode", "average")

            # Update rate label with configured sample rate
            configured_rate_sps = 1e9 / self.sample_interval_ns
            self.rate_label.setText(
                f"Rate: ... : {self._format_rate_sps(configured_rate_sps)}"
            )
        except KeyError:
            logger.debug("Metadata not fully available yet. Will retry.")

    def _update_heartbeat(self) -> None:
        """Update UI heartbeat to show the UI thread is alive."""
        self.heartbeat_index = (self.heartbeat_index + 1) % len(self.heartbeat_chars)
        self.heartbeat_label.setText(
            f"UI: {self.heartbeat_chars[self.heartbeat_index]}"
        )

    def _handle_new_data(
        self, dataset: h5py.Dataset, start_index: int, current_size: int
    ) -> None:
        """Process a new window of data."""
        self.data_change_count += 1
        self.last_displayed_size = current_size
        self.last_data_timestamp = time.time()
        self.acq_status_label.setText('<span style="color: green">Active</span>')

        # Read only the most recent data window
        data_window = dataset[start_index:current_size]
        self.file_read_count += 1

        logger.debug(
            f"Update {self.update_count}: Reading window of {len(data_window):,} samples from index {start_index:,}"
        )

        # Update the display with this complete window (only when data changes)
        self.update_display(data_window)

    def _handle_stale_data(self) -> None:
        """Handle a file check where no new data is found."""
        self.stale_update_count += 1
        self.acq_status_label.setText(
            '<span style="color: orange">Acquiring... </span>'
        )
        # Log if we're frequently updating with no new data
        if self.stale_update_count % 10 == 0:
            logger.debug(
                f"File check #{self.update_count} with no new data (stale checks: {self.stale_update_count})"
            )

    def _update_status_labels(self, current_size: int) -> None:
        """Update the various status labels in the UI."""
        # Update samples label
        samples_text = self.format_sample_count(current_size)
        self.samples_label.setText(f"Samples: {samples_text}")

        # Color-code latency
        latency_color = (
            "green"
            if self.display_latency_ms < 100
            else "orange"
            if self.display_latency_ms < 500
            else "red"
        )
        self.plotter_latency_label.setText(
            f'<span style="color: {latency_color}">Plotter Latency: {self.display_latency_ms:.0f}ms</span>'
        )

        # Error counter
        total_errors = self.conversion_error_count + self.file_error_count
        error_color = (
            "green" if total_errors == 0 else "orange" if total_errors < 10 else "red"
        )
        self.error_label.setText(
            f'<span style="color: {error_color}">Errors: {total_errors}</span>'
        )

    def _update_rate_label(self, current_size: int) -> None:
        """Check and update acquisition rate status."""
        if not self.rate_check_start_time:
            return

        elapsed_time = time.perf_counter() - self.rate_check_start_time
        if elapsed_time > 1.0:  # Check only after 1s for stability
            points_per_timestep = 2 if self.downsample_mode == "aggregate" else 1
            samples_acquired = current_size - self.rate_check_start_samples
            timesteps_acquired = samples_acquired / points_per_timestep
            actual_rate_sps = timesteps_acquired / elapsed_time

            configured_rate_sps = 1e9 / self.sample_interval_ns
            rate_ratio = actual_rate_sps / configured_rate_sps

            configured_rate_str = self._format_rate_sps(configured_rate_sps)
            actual_rate_str = self._format_rate_sps(actual_rate_sps)

            rate_text = f"Rate: {actual_rate_str} : {configured_rate_str}"
            if rate_ratio < 0.95:
                self.rate_label.setText(f'<span style="color: red">{rate_text}</span>')
            else:
                self.rate_label.setText(rate_text)

    def _process_data_from_file(self, f: h5py.File) -> None:
        """Read and process data from an open HDF5 file."""
        if "adc_counts" not in f:
            return

        dataset = f["adc_counts"]
        current_size = dataset.shape[0]

        if current_size == 0:
            return

        # Start the rate check timer on the first data point
        if self.rate_check_start_time is None:
            self.rate_check_start_time = time.perf_counter()
            self.rate_check_start_samples = current_size

        # Read metadata if not already done
        if self.voltage_range_v is None:
            self.read_metadata(f)

        # Dynamically calculate the number of timesteps for the display window
        display_window_timesteps = int(
            self.display_window_seconds / (self.sample_interval_ns * 1e-9)
        )

        # In aggregate mode, each timestep has two points (min/max)
        display_window_points = display_window_timesteps
        if self.downsample_mode == "aggregate":
            display_window_points *= 2

        # Calculate where to start reading to get the last window of points
        start_index = max(0, current_size - display_window_points)
        self.data_start_sample = start_index

        # Track data freshness and changes
        if current_size > self.last_displayed_size:
            self._handle_new_data(dataset, start_index, current_size)
        else:
            self._handle_stale_data()

        self._update_status_labels(current_size)
        self._update_rate_label(current_size)

    def update_from_file(self) -> None:
        """Timer-driven function to read data from the HDF5 file and update the plot."""
        self.update_count += 1
        self._update_heartbeat()

        try:
            with h5py.File(self.hdf5_path, "r") as f:
                self._process_data_from_file(f)
        except (FileNotFoundError, OSError):
            self.acq_status_label.setText(
                '<span style="color: orange">Waiting for file...</span>'
            )
        except Exception as e:
            self.file_error_count += 1
            logger.error(f"Update {self.update_count}: Error reading file - {e}")
            self.acq_status_label.setText('<span style="color: red">File error!</span>')

    def update_display(self, data_window: np.ndarray) -> None:
        """Processes and displays a new window of data.

        This involves decimation, voltage conversion, and updating the plot curve.

        Args:
            data_window: A NumPy array containing the raw ADC counts for display.
        """
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
        if self.decimation_factor > 1:
            decimated_data = min_max_decimate_numba(
                self.display_data, self.decimation_factor
            )
        else:
            decimated_data = self.display_data

        # Debug: Log ADC values and metadata
        logger.debug(f"ADC range: {decimated_data.min()} to {decimated_data.max()}")
        logger.debug(
            f"voltage_range_v: {self.voltage_range_v}, max_adc: {self.max_adc}"
        )

        # Convert to voltage if we have calibration data
        if self.voltage_range_v is not None and self.max_adc is not None:
            try:
                voltage_data = adc_to_mV(
                    decimated_data, self.voltage_range_v, self.max_adc
                )
                logger.debug(
                    f"Voltage conversion successful, range: {voltage_data.min():.1f} to {voltage_data.max():.1f} mV"
                )
            except Exception as e:
                self.conversion_error_count += 1
                logger.warning(f"Voltage conversion failed: {e}, using raw ADC values")
                voltage_data = decimated_data.astype(float)
        else:
            logger.warning(
                "Missing calibration data (voltage_range_v or max_adc), using raw ADC values"
            )
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

        # Update the X-axis range to match the new time axis, creating a "snapshot" effect.
        self.plot_widget.setXRange(time_axis[0], time_axis[-1], padding=0)

        # Auto-scale the Y-axis occasionally.
        if self.display_update_count % 10 == 1:
            self.plot_widget.enableAutoRange(axis="y")

    def _format_rate_sps(self, rate_sps: float) -> str:
        """Formats a sample rate in Samples/sec into a human-readable string."""
        if rate_sps >= 1e9:
            return f"{rate_sps / 1e9:.2f} GS/s"
        if rate_sps >= 1e6:
            return f"{rate_sps / 1e6:.2f} MS/s"
        if rate_sps >= 1e3:
            return f"{rate_sps / 1e3:.2f} kS/s"
        return f"{rate_sps:.2f} S/s"

    def format_sample_count(self, count: int) -> str:
        """Formats a large integer count into a human-readable string with units.

        Args:
            count: The integer number to format.

        Returns:
            A formatted string (e.g., "1.23M", "2.34G").
        """
        if count >= 1_000_000_000:
            return f"{count / 1_000_000_000:.2f}G"
        if count >= 1_000_000:
            return f"{count / 1_000_000:.2f}M"
        if count >= 1_000:
            return f"{count / 1_000:.2f}K"
        else:
            return str(count)

    def create_time_axis(self, n_samples: int) -> np.ndarray:
        """Creates a time axis for the displayed data window.

        The time axis is absolute, based on the data window's start position in
        the overall acquisition. It accounts for the `aggregate` downsample mode,
        where the data stream consists of interleaved min/max pairs.

        For min-max decimated data, it generates pairs of time coordinates to
        draw vertical lines for each min-max pair. For non-decimated data, it
        generates a linearly spaced time axis.

        Args:
            n_samples: The number of points for the time axis. This should be
                the number of points *after* decimation.

        Returns:
            A NumPy array representing the time axis in seconds.
        """
        if n_samples == 0:
            return np.array([])

        time_per_timestep = self.sample_interval_ns * 1e-9
        points_per_timestep = 2 if self.downsample_mode == "aggregate" else 1
        time_per_point = time_per_timestep / points_per_timestep

        start_time = (self.data_start_sample / points_per_timestep) * time_per_timestep

        if self.decimation_factor > 1:
            # For min-max, create pairs of time points for vertical lines
            num_pairs = n_samples // 2
            time_step_between_groups = self.decimation_factor * time_per_point
            group_times = start_time + np.arange(num_pairs) * time_step_between_groups
            return np.repeat(group_times, 2)
        else:
            # For non-decimated data
            if self.downsample_mode == "aggregate":
                # Data is already min/max pairs from hardware. Create vertical lines.
                num_pairs = n_samples // 2
                time_step_between_pairs = time_per_timestep
                pair_times = start_time + np.arange(num_pairs) * time_step_between_pairs
                return np.repeat(pair_times, 2)
            else:
                # For linear (non-aggregate) data, create a simple time axis
                duration = (
                    (n_samples - 1) * time_per_point if n_samples > 1 else 0
                )
                end_time = start_time + duration
                return np.linspace(start_time, end_time, n_samples)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Handles the window close event for a clean shutdown."""
        logger.info("Close event received. Stopping timer.")
        self.timer.stop()
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handles key presses for application control (e.g., 'Q' to quit)."""
        if event.key() == Qt.Key_Q:
            logger.info("'Q' key pressed. Closing application.")
            self.close()
        else:
            super().keyPressEvent(event)


@click.command()
@click.argument("hdf5_path", type=click.Path(dir_okay=False))
@click.option(
    "--window",
    type=float,
    default=0.5,
    help="Display window in seconds. [default: 0.5]",
)
@click.option(
    "--decimation",
    type=int,
    default=150,
    help="Decimation factor for plotting. [default: 150]",
)
def main(hdf5_path: str, window: float, decimation: int) -> None:
    """Standalone HDF5 live plotter."""
    logger.info("Plotter process starting")
    app = QApplication([])
    plotter = HDF5LivePlotter(
        hdf5_path=hdf5_path,
        display_window_seconds=window,
        decimation_factor=decimation,
    )
    plotter.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
