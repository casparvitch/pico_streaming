from __future__ import annotations

import queue
import signal
import threading
import time
import numpy as np
import sys
from loguru import logger
import h5py
from typing import List, Optional

from consumer import Consumer
from pico import PicoDevice


class StreamExample:
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
        plot_resolution: int = 4000,
        hardware_downsample: int = 1,
    ) -> None:
        # --- Configuration ---
        self.output_file = output_file
        self.debug = debug
        self.enable_live_plot = enable_live_plot


        # --- Validate configuration ---
        max_rate_msps = 0
        if resolution_bits == 8:
            max_rate_msps = 125.0
        elif resolution_bits in [12, 14, 15, 16]:
            max_rate_msps = 62.5
        else:
            # This should be caught by argparse choices, but as a safeguard:
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

        # Check if sample rate is excessive for the analog bandwidth of the selected range
        ANALOG_BANDWIDTH_MHZ = {
            "PS5000A_10MV": 200, "PS5000A_20MV": 200, "PS5000A_50MV": 200,
            "PS5000A_100MV": 200, "PS5000A_200MV": 200, "PS5000A_500MV": 200,
            "PS5000A_1V": 200, "PS5000A_2V": 150, "PS5000A_5V": 100,
            "PS5000A_10V": 50, "PS5000A_20V": 25, "PS5000A_50V": 25,
            "PS5000A_100V": 25, "PS5000A_200V": 25,
        }
        if channel_range_str in ANALOG_BANDWIDTH_MHZ:
            bandwidth_mhz = ANALOG_BANDWIDTH_MHZ[channel_range_str]
            # Nyquist rate is 2x bandwidth. A common rule of thumb is 3-5x.
            # Warn if sampling faster than 5x the analog bandwidth.
            if sample_rate_msps > 5 * bandwidth_mhz:
                logger.warning(
                    f"Sample rate ({sample_rate_msps} MS/s) may be unnecessarily high "
                    f"for the selected voltage range ({channel_range_str}), which has an "
                    f"analog bandwidth of {bandwidth_mhz} MHz."
                )

        # --- Buffer Sizing ---
        # Dynamically size buffers to hold a specific duration of data. This makes
        # memory usage proportional to the data rate, providing a consistent
        # time-based buffer to handle processing latencies.

        # Consumer buffers (for writing to HDF5) are sized to hold 1 second of data.
        # This is a good balance, as larger buffers lead to more efficient disk writes
        # but use more RAM.
        consumer_buffer_duration_s = 1.0
        self.consumer_buffer_size = int(sample_rate_msps * 1e6 * consumer_buffer_duration_s)
        self.consumer_num_buffers = 5  # A pool of 5 buffers

        # The Picoscope driver buffer is sized to hold 0.5 seconds of data. This
        # buffer receives data directly from the hardware. A smaller size ensures
        # that the application receives data in timely chunks, reducing latency.
        driver_buffer_duration_s = 0.5
        self.pico_driver_buffer_size = int(sample_rate_msps * 1e6 * driver_buffer_duration_s)
        self.pico_driver_num_buffers = 1  # A single large buffer is efficient for the driver

        logger.info(
            f"Consumer buffer sized to {self.consumer_buffer_size:,} samples "
            f"({consumer_buffer_duration_s}s)"
        )
        logger.info(
            f"Pico driver buffer sized to {self.pico_driver_buffer_size:,} samples "
            f"({driver_buffer_duration_s}s)"
        )

        # --- Plotting Decimation ---
        # Calculate the decimation factor needed to achieve the target plot resolution.
        samples_in_window = sample_rate_msps * 1e6 * plot_window_s
        decimation_factor = max(1, int(samples_in_window / plot_resolution))
        logger.info(
            f"Plotting with target resolution of {plot_resolution} points. "
            f"Calculated decimation factor: {decimation_factor}"
        )

        # --- Hardware Down-sampling ---
        if hardware_downsample > 1:
            pico_downsample_ratio = hardware_downsample
            pico_ratio_mode = "PS5000A_RATIO_MODE_AVERAGE"
            logger.info(
                f"Hardware down-sampling (averaging) enabled with ratio {pico_downsample_ratio}."
            )
        else:
            pico_downsample_ratio = 1
            pico_ratio_mode = "PS5000A_RATIO_MODE_NONE"

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
        )

        self.pico_device.set_channel(
            "PS5000A_CHANNEL_A", 1, "PS5000A_DC", self.pico_channel_range, 0.0
        )
        self.pico_device.set_channel(
            "PS5000A_CHANNEL_B", 0, "PS5000A_DC", self.pico_channel_range, 0.0
        )
        self.pico_device.set_data_buffer(
            "PS5000A_CHANNEL_A", 0, "PS5000A_RATIO_MODE_NONE"
        )
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
        signal.signal(signal.SIGINT, self.signal_handler)

        # --- Live Plotting (optional) ---
        self.live_plotter: Optional["HDF5LivePlotter"] = None
        self.qt_app: Optional["QApplication"] = None
        self.start_time: Optional[float] = None

        if self.enable_live_plot:
            # Import Qt components only when needed
            from PyQt5.QtWidgets import QApplication
            from dfplot import HDF5LivePlotter

            # Create Qt application if it doesn't exist
            if not QApplication.instance():
                self.qt_app = QApplication(sys.argv)

            # Create the live plotter
            self.live_plotter = HDF5LivePlotter(
                output_file,
                display_window_seconds=plot_window_s,
                decimation_factor=decimation_factor,
            )

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

        # Calculate effective duration and rate
        if self.start_time:
            end_time = time.time()
            duration = end_time - self.start_time
            total_samples = self.consumer.values_written
            effective_rate_msps = (
                (total_samples / duration) / 1e6 if duration > 0 else 0
            )
            configured_rate_msps = 1e3 / self.pico_device.sample_int.value

            logger.info("--- Acquisition Summary ---")
            logger.info(f"Total acquisition time: {duration:.2f} s")
            logger.info(
                f"Total samples written: {self.consumer.format_sample_count(total_samples)}"
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
                    f"Effective rate was only {rate_ratio:.1%} of the configured rate."
                )
            else:
                logger.success("Effective rate matches configured rate.")
            logger.info("--------------------------")

        self.shutdown_event.set()

        logger.info("Stopping data acquisition and saving...")

        # 2. Stop plotter timer and close window
        if self.live_plotter:
            self.live_plotter.timer.stop()
            self.live_plotter.close()

        # 3. Wait for threads to finish
        logger.info("Waiting for Picoscope thread to terminate...")
        if hasattr(self, "pico_thread") and self.pico_thread.is_alive():
            self.pico_thread.join(timeout=2.0)
            if self.pico_thread.is_alive():
                logger.critical("Pico thread failed to terminate.")

        logger.info("Waiting for Consumer thread to terminate...")
        if hasattr(self, "consumer_thread") and self.consumer_thread.is_alive():
            self.consumer_thread.join(timeout=2.0)
            if self.consumer_thread.is_alive():
                logger.critical("Consumer thread failed to terminate.")

        # 4. Now safe to close device
        self.pico_device.close_device()

        # 5. Quit Qt
        if self.qt_app:
            self.qt_app.quit()

        logger.success("Shutdown complete.")

    def run(self) -> None:
        """Starts the acquisition threads and, if enabled, the Qt event loop."""
        # Start acquisition threads
        self.start_time = time.time()
        self.consumer_thread.start()
        self.pico_thread.start()

        # Handle Qt event loop if plotting is enabled
        if self.enable_live_plot and self.qt_app:
            # Show plotter window and run Qt event loop (blocking)
            self.live_plotter.show()
            self.qt_app.exec_()

            # Once the plot window is closed, initiate shutdown
            self.shutdown()
        else:
            # Original behavior for non-plotting mode
            self.consumer_thread.join()
            self.pico_thread.join()
            logger.success("Acquisition complete!")


if __name__ == "__main__":
    # This block runs when the script is executed directly.
    # It handles command-line argument parsing, logging setup, and
    # instantiates and runs the main StreamExample class.
    import argparse
    from datetime import datetime

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="PicoScope Data Acquisition")
    parser.add_argument(
        "--sample-rate",
        "-s",
        type=float,
        default=62.5,
        help="Sample rate in MS/s (e.g., 62.5 for 62.5MS/s). Use 0 for max rate.",
    )
    parser.add_argument(
        "--resolution",
        "-b",
        type=int,
        default=12,
        choices=[8, 12, 14, 15, 16],
        help="Resolution in bits (default: 12).",
    )
    voltage_ranges = [
        "PS5000A_10MV", "PS5000A_20MV", "PS5000A_50MV", "PS5000A_100MV",
        "PS5000A_200MV", "PS5000A_500MV", "PS5000A_1V", "PS5000A_2V",
        "PS5000A_5V", "PS5000A_10V", "PS5000A_20V", "PS5000A_50V",
        "PS5000A_100V", "PS5000A_200V"
    ]
    parser.add_argument(
        "--range",
        choices=voltage_ranges,
        default="PS5000A_20V",
        help="Voltage range for Channel A (default: PS5000A_20V).",
    )
    parser.add_argument(
        "--plot",
        "-p",
        action="store_true",
        default=True,
        help="Enable live plotting (requires PyQt5 and pyqtgraph, default: true).",
    )
    parser.add_argument(
        "--output", "-o", help="Output HDF5 file (default: auto-timestamped as ./output_{}.hdf5)."
    )
    parser.add_argument(
        "--plot-window",
        "-w",
        type=float,
        default=0.5,
        help="Set the live plot display window duration in seconds (default: 0.5s).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "--plot-resolution",
        type=int,
        default=4000,
        help="Target number of points for the plot window (default: 4000).",
    )
    parser.add_argument(
        "--hardware-downsample",
        type=int,
        default=1,
        help="Hardware down-sampling (averaging) ratio. 1 for none (default: 1).",
    )
    args = parser.parse_args()

    # Configure logging
    logger.remove()
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.add(sys.stderr, level=log_level)
    logger.info(f"Logging configured at level: {log_level}")

    # Auto-generate filename if not specified
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"./output_{timestamp}.hdf5"

    logger.info(f"Output file: {args.output}")

    try:
        # Create and run the streamer
        streamer = StreamExample(
            sample_rate_msps=args.sample_rate,
            resolution_bits=args.resolution,
            channel_range_str=args.range,
            enable_live_plot=args.plot,
            output_file=args.output,
            debug=args.verbose,
            plot_window_s=args.plot_window,
            plot_resolution=args.plot_resolution,
            hardware_downsample=args.hardware_downsample,
        )
        streamer.run()
    except RuntimeError as e:
        if "PICO_NOT_FOUND" in str(e):
            logger.critical(
                "Picoscope device not found. Please check connection and ensure no other software is using it."
            )
        else:
            logger.critical(f"Failed to initialize Picoscope: {e}")
        sys.exit(1)

    # --- Verification Step ---
    logger.info(f"Verifying output file: {args.output}")
    try:
        expected_samples = streamer.consumer.values_written
        if expected_samples == 0:
            logger.warning("Consumer processed no samples. Nothing to verify.")
        else:
            with h5py.File(args.output, "r") as f:
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
