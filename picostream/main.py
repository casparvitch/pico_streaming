from __future__ import annotations

import queue
import signal
import sys
import threading
import time
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from PyQt5.QtWidgets import QApplication

import click
import h5py
import numpy as np
from loguru import logger

from .consumer import Consumer
from .pico import PicoDevice


class Streamer:
    """Orchestrates the Picoscope data acquisition process.

    This class initializes the Picoscope device (producer), the HDF5 writer
    (consumer), and the live plotter. It manages the threads, queues, and
    graceful shutdown of the entire application.
    """

    def __init__(
        self,
        sample_rate_msps: float = 62.5,
        resolution_bits: int = 12,
        channel_range_str: str = "PS5000A_20V",
        enable_live_plot: bool = False,
        output_file: str = "./output.hdf5",
        debug: bool = False,
        plot_window_s: float = 0.5,
        plot_points: int = 4000,
        hardware_downsample: int = 1,
        downsample_mode: str = "average",
    ) -> None:
        # --- Configuration ---
        self.output_file = output_file
        self.debug = debug
        self.enable_live_plot = enable_live_plot
        self.plot_window_s = plot_window_s

        (
            sample_rate_msps,
            pico_downsample_ratio,
            pico_ratio_mode,
        ) = self._validate_config(
            resolution_bits,
            sample_rate_msps,
            channel_range_str,
            hardware_downsample,
            downsample_mode,
        )
        # Dynamically size buffers to hold a specific duration of data. This makes
        # memory usage proportional to the data rate, providing a consistent
        # time-based buffer to handle processing latencies.
        effective_rate_sps = (sample_rate_msps * 1e6) / pico_downsample_ratio

        # Consumer buffers (for writing to HDF5) are sized to hold 1 second of data.
        # This is a good balance, as larger buffers lead to more efficient disk writes
        # but use more RAM.
        consumer_buffer_duration_s = 1.0
        self.consumer_buffer_size = int(
            effective_rate_sps * consumer_buffer_duration_s
        )
        if downsample_mode == "aggregate":
            self.consumer_buffer_size *= 2
        self.consumer_num_buffers = 5  # A pool of 5 buffers

        # The Picoscope driver buffer is sized to hold 0.5 seconds of data. This
        # buffer receives data directly from the hardware. A smaller size ensures
        # that the application receives data in timely chunks, reducing latency.
        driver_buffer_duration_s = 0.5
        self.pico_driver_buffer_size = int(
            effective_rate_sps * driver_buffer_duration_s
        )
        self.pico_driver_num_buffers = (
            1  # A single large buffer is efficient for the driver
        )

        logger.info(
            f"Consumer buffer sized to {self.consumer_buffer_size:,} samples "
            f"({consumer_buffer_duration_s}s at effective rate)"
        )
        logger.info(
            f"Pico driver buffer sized to {self.pico_driver_buffer_size:,} samples "
            f"({driver_buffer_duration_s}s at effective rate)"
        )

        # --- Plotting Decimation ---
        # Calculate the decimation factor needed to achieve the target number of plot points.
        points_per_timestep = 2 if downsample_mode == "aggregate" else 1
        samples_in_window = effective_rate_sps * plot_window_s * points_per_timestep
        self.decimation_factor = max(1, int(samples_in_window / plot_points))
        logger.info(
            f"Plotting with target of {plot_points} points. "
            f"Calculated decimation factor: {self.decimation_factor}"
        )

        # Picoscope hardware settings
        self.pico_resolution = f"PS5000A_DR_{resolution_bits}BIT"
        self.pico_channel_range = channel_range_str
        self.pico_sample_interval_ns = int(1000 / sample_rate_msps)
        self.pico_sample_unit = "PS5000A_NS"

        # Streaming settings
        self.pico_auto_stop = 0  # Don't auto stop
        self.pico_auto_stop_stream = False
        # --- End Configuration ---

        # --- System Components ---
        self.shutdown_event: threading.Event = threading.Event()
        data_queue: queue.Queue[int] = queue.Queue()
        empty_queue: queue.Queue[int] = queue.Queue()
        data_buffers: List[np.ndarray] = []

        # Pre-allocate a pool of numpy arrays for data transfer and populate the
        # empty_queue with their indices.
        for idx in range(self.consumer_num_buffers):
            data_buffers.append(np.empty((self.consumer_buffer_size,), dtype="int16"))
            empty_queue.put(idx)

        # --- Producer ---
        self.pico_device: PicoDevice = PicoDevice(
            0,  # handle
            self.pico_resolution,
            self.pico_driver_buffer_size,
            self.pico_driver_num_buffers,
            self.consumer_buffer_size,
            data_queue,
            empty_queue,
            data_buffers,
            self.shutdown_event,
            downsample_mode=downsample_mode,
        )

        self.pico_device.set_channel(
            "PS5000A_CHANNEL_A", 1, "PS5000A_DC", self.pico_channel_range, 0.0
        )
        self.pico_device.set_channel(
            "PS5000A_CHANNEL_B", 0, "PS5000A_DC", self.pico_channel_range, 0.0
        )
        self.pico_device.set_data_buffer("PS5000A_CHANNEL_A", 0, pico_ratio_mode)
        self.pico_device.configure_streaming_var(
            self.pico_sample_interval_ns,
            self.pico_sample_unit,
            0,  # pre-trigger samples
            pico_downsample_ratio,
            pico_ratio_mode,
            self.pico_auto_stop,
            self.pico_auto_stop_stream,
        )

        # Run streaming once to get the actual sample interval from the driver
        self.pico_device.run_streaming()

        # --- Consumer ---
        # Get metadata from configured device and pass to consumer
        metadata = self.pico_device.get_metadata()
        self.consumer: Consumer = Consumer(
            self.consumer_buffer_size,
            data_queue,
            empty_queue,
            data_buffers,
            output_file,
            self.shutdown_event,
            metadata=metadata,
        )

        # --- Threads ---
        self.consumer_thread: threading.Thread = threading.Thread(
            target=self.consumer.consume
        )
        self.pico_thread: threading.Thread = threading.Thread(
            target=self.pico_device.run_capture
        )

        # --- Signal Handling ---
        # Only set the signal handler if not in GUI mode.
        # In GUI mode, the main function will handle signals to quit the Qt app.
        if not self.enable_live_plot:
            signal.signal(signal.SIGINT, self.signal_handler)

        # --- Live Plotting (optional) ---
        self.start_time: Optional[float] = None

    def _validate_config(
        self,
        resolution_bits: int,
        sample_rate_msps: float,
        channel_range_str: str,
        hardware_downsample: int,
        downsample_mode: str,
    ) -> tuple[float, int, str]:
        """Validates user-provided settings and returns derived configuration."""
        if resolution_bits == 8:
            max_rate_msps = 125.0
        elif resolution_bits in [12, 14, 15, 16]:
            max_rate_msps = 62.5
        else:
            raise ValueError(
                f"Unsupported resolution: {resolution_bits} bits. Must be one of 8, 12, 14, 15, 16."
            )

        if sample_rate_msps <= 0:
            sample_rate_msps = max_rate_msps
            logger.info(f"Max sample rate requested. Setting to {max_rate_msps} MS/s.")

        if sample_rate_msps > max_rate_msps:
            raise ValueError(
                f"Sample rate {sample_rate_msps} MS/s exceeds maximum of {max_rate_msps} MS/s for {resolution_bits}-bit resolution."
            )

        # Check if sample rate is excessive for the analog bandwidth.
        # Bandwidth is dependent on both resolution and voltage range.
        # (Based on PicoScope 5000A/B Series datasheet)
        inv_voltage_map = {v: k for k, v in VOLTAGE_RANGE_MAP.items()}
        voltage_v = inv_voltage_map.get(channel_range_str, 0)

        if resolution_bits == 16:
            bandwidth_mhz = 20  # 20 MHz for all ranges
        elif resolution_bits == 15:
            # Bandwidth is 70MHz for < ±5V, 60MHz for >= ±5V
            bandwidth_mhz = 70 if voltage_v < 5.0 else 60
        else:  # 8-14 bits
            # Bandwidth is 100MHz for < ±5V, 60MHz for >= ±5V
            bandwidth_mhz = 100 if voltage_v < 5.0 else 60

        # Nyquist rate is 2x bandwidth. A common rule of thumb is 3-5x.
        # Warn if sampling faster than 5x the analog bandwidth.
        if sample_rate_msps > 5 * bandwidth_mhz:
            logger.warning(
                f"Sample rate ({sample_rate_msps} MS/s) may be unnecessarily high "
                f"for the selected voltage range ({channel_range_str}), which has an "
                f"analog bandwidth of {bandwidth_mhz} MHz."
            )

        if downsample_mode == "aggregate" and hardware_downsample <= 1:
            raise ValueError(
                "Hardware downsample ratio must be > 1 for 'aggregate' mode."
            )

        if hardware_downsample > 1:
            pico_downsample_ratio = hardware_downsample
            pico_ratio_mode = f"PS5000A_RATIO_MODE_{downsample_mode.upper()}"
            logger.info(
                f"Hardware down-sampling ({downsample_mode}) enabled "
                + f"with ratio {pico_downsample_ratio}."
            )
        else:
            pico_downsample_ratio = 1
            pico_ratio_mode = "PS5000A_RATIO_MODE_NONE"

        return sample_rate_msps, pico_downsample_ratio, pico_ratio_mode

    def signal_handler(self, _sig: int, frame: Optional[object]) -> None:
        """Handles Ctrl+C interrupts to initiate a graceful shutdown."""
        logger.warning("Ctrl+C detected. Shutting down.")
        self.shutdown()

    def shutdown(self) -> None:
        """Performs a graceful shutdown of all components.

        This method calculates final statistics, stops all threads, closes the
        plotter, and ensures the Picoscope device is properly closed.
        """
        if self.shutdown_event.is_set():
            return

        self._log_acquisition_summary()
        self.shutdown_event.set()

        logger.info("Stopping data acquisition and saving...")

        self._join_threads()
        self.pico_device.close_device()

        logger.success("Shutdown complete.")

    def _log_acquisition_summary(self) -> None:
        """Calculates and logs final acquisition statistics."""
        if not self.start_time:
            return

        end_time = time.time()
        duration = end_time - self.start_time
        total_samples = self.consumer.values_written
        effective_rate_msps = (total_samples / duration) / 1e6 if duration > 0 else 0
        configured_rate_msps = 1e3 / self.pico_device.sample_int.value

        logger.info("--- Acquisition Summary ---")
        logger.info(f"Total acquisition time: {duration:.2f} s")
        logger.info(
            "Total samples written: "
            + f"{self.consumer.format_sample_count(total_samples)}"
        )
        logger.info(f"Configured sample rate: {configured_rate_msps:.2f} MS/s")
        logger.info(f"Effective average rate: {effective_rate_msps:.2f} MS/s")

        rate_ratio = (
            effective_rate_msps / configured_rate_msps
            if configured_rate_msps > 0
            else 0
        )
        if rate_ratio < 0.95:
            logger.warning(
                f"Effective rate was only {rate_ratio:.1%} " + "of the configured rate."
            )
        else:
            logger.success("Effective rate matches configured rate.")
        logger.info("--------------------------")

    def _join_threads(self) -> None:
        """Waits for the producer and consumer threads to terminate."""
        for thread_name in ["pico_thread", "consumer_thread"]:
            thread = getattr(self, thread_name, None)
            if thread and thread.is_alive():
                logger.info(f"Waiting for {thread_name} to terminate...")
                thread.join(timeout=2.0)
                if thread.is_alive():
                    logger.critical(f"{thread_name} failed to terminate.")

    def run(self, app: Optional[QApplication] = None) -> None:
        """Starts the acquisition threads and optionally the Qt event loop."""
        # Start acquisition threads
        self.start_time = time.time()
        self.consumer_thread.start()
        self.pico_thread.start()

        # Handle Qt event loop if plotting is enabled
        if self.enable_live_plot and app:
            from .dfplot import HDF5LivePlotter

            plotter = HDF5LivePlotter(
                hdf5_path=self.output_file,
                display_window_seconds=self.plot_window_s,
                decimation_factor=self.decimation_factor,
                shutdown_event=self.shutdown_event,
            )
            plotter.show()

            # Run the Qt event loop. This will block until the plot window is closed.
            app.exec_()

            # Once the window is closed, the shutdown event should have been set.
            # We call shutdown() to ensure threads are joined and cleanup happens.
            self.shutdown()
        else:
            # Original non-GUI behavior
            self.consumer_thread.join()
            self.pico_thread.join()
            logger.success("Acquisition complete!")


# --- Argument Parsing ---
VOLTAGE_RANGE_MAP = {
    0.01: "PS5000A_10MV",
    0.02: "PS5000A_20MV",
    0.05: "PS5000A_50MV",
    0.1: "PS5000A_100MV",
    0.2: "PS5000A_200MV",
    0.5: "PS5000A_500MV",
    1.0: "PS5000A_1V",
    2.0: "PS5000A_2V",
    5.0: "PS5000A_5V",
    10.0: "PS5000A_10V",
    20.0: "PS5000A_20V",
}


@click.command()
@click.option(
    "--sample-rate",
    "-s",
    type=float,
    default=20,
    help="Sample rate in MS/s (e.g., 62.5). Use 0 for max rate. [default: 20]",
)
@click.option(
    "--resolution",
    "-b",
    type=click.Choice(["8", "12", "16"]),
    default="12",
    help="Resolution in bits. [default: 12]",
)
@click.option(
    "--rangev",
    "-r",
    type=click.Choice([str(k) for k in sorted(VOLTAGE_RANGE_MAP.keys())]),
    default="20.0",
    help=f"Voltage range in Volts. [default: 20.0]",
)
@click.option(
    "--plot/--no-plot",
    "-p",
    is_flag=True,
    default=True,
    help="Enable/disable live plotting. [default: --plot]",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, writable=True),
    help="Output HDF5 file (default: auto-timestamped).",
)
@click.option(
    "--plot-window",
    "-w",
    type=float,
    default=0.5,
    help="Live plot display window duration in seconds. [default: 0.5]",
)
@click.option(
    "--verbose", "-v", is_flag=True, default=False, help="Enable debug logging."
)
@click.option(
    "--plot-npts",
    "-n",
    type=int,
    default=4000,
    help="Target number of points for the plot window. [default: 4000]",
)
@click.option(
    "--hardware-downsample",
    "-h",
    type=int,
    default=1,
    help="Hardware down-sampling ratio (power of 2 for 'average' mode). [default: 1]",
)
@click.option(
    "--downsample-mode",
    "-m",
    type=click.Choice(["average", "aggregate"]),
    default="voltage_range_v",
    help="Hardware down-sampling mode. [default: average]",
)
def main(
    sample_rate: float,
    resolution: str,
    rangev: str,
    plot: bool,
    output: Optional[str],
    plot_window: float,
    verbose: bool,
    plot_npts: int,
    hardware_downsample: int,
    downsample_mode: str,
) -> None:
    """High-speed data acquisition tool for Picoscope 5000a series."""
    # --- Argument Validation and Processing ---
    channel_range_str = VOLTAGE_RANGE_MAP[float(rangev)]
    resolution_bits = int(resolution)

    app: Optional[QApplication] = None
    if plot:
        from PyQt5.QtWidgets import QApplication

        app = QApplication(sys.argv)

        # When plotting, SIGINT should gracefully close the Qt application.
        # The main loop will then handle the shutdown.
        def sigint_handler(_sig: int, _frame: Optional[object]) -> None:
            logger.warning("Ctrl+C detected. Closing application.")
            QApplication.quit()

        signal.signal(signal.SIGINT, sigint_handler)

    # Configure logging
    logger.remove()
    log_level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=log_level)
    logger.info(f"Logging configured at level: {log_level}")

    # Auto-generate filename if not specified
    if not output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = f"./output_{timestamp}.hdf5"

    logger.info(f"Output file: {output}")
    logger.info(f"Selected voltage range: {rangev}V -> {channel_range_str}")

    try:
        # Create and run the streamer
        streamer = Streamer(
            sample_rate_msps=sample_rate,
            resolution_bits=resolution_bits,
            channel_range_str=channel_range_str,
            enable_live_plot=plot,
            output_file=output,
            debug=verbose,
            plot_window_s=plot_window,
            plot_points=plot_npts,
            hardware_downsample=hardware_downsample,
            downsample_mode=downsample_mode,
        )
        streamer.run(app)
    except RuntimeError as e:
        if "PICO_NOT_FOUND" in str(e):
            logger.critical(
                "Picoscope device not found. Please check connection and ensure no other software is using it."
            )
        else:
            logger.critical(f"Failed to initialize Picoscope: {e}")
        sys.exit(1)

    # --- Verification Step ---
    logger.info(f"Verifying output file: {output}")
    try:
        expected_samples = streamer.consumer.values_written
        if expected_samples == 0:
            logger.warning("Consumer processed no samples. Nothing to verify.")
        else:
            with h5py.File(output, "r") as f:
                if "adc_counts" not in f:
                    raise ValueError("Dataset 'adc_counts' not found in HDF5 file.")

                actual_samples = len(f["adc_counts"])
                if actual_samples == expected_samples:
                    logger.success(
                        f"Verification PASSED: File contains {actual_samples} samples, as expected."
                    )
                else:
                    logger.error(
                        f"Verification FAILED: Expected {expected_samples} samples, but file has {actual_samples}."
                    )
    except Exception as e:
        logger.error(f"HDF5 file verification failed: {e}")


if __name__ == "__main__":
    main()
