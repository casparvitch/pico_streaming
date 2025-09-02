import queue
import signal
import threading
import time
import numpy as np
import sys
from loguru import logger

from consumer import Consumer
from pico import PicoDevice


class StreamExample:

    def __init__(
        self,
        enable_live_plot=False,
        output_file="/tmp/data.hdf5",
        debug=False,
        plot_window_s=0.5,
        decimation_factor=150,
    ):
        # --- Configuration ---
        self.output_file = output_file
        self.debug = debug
        self.enable_live_plot = enable_live_plot

        # Consumer buffer settings (for writing to HDF5)
        self.consumer_buffer_size = 6_400_000  # Samples per buffer
        self.consumer_num_buffers = 5  # Number of buffers

        # Picoscope hardware settings
        self.pico_resolution = "PS5000A_DR_12BIT"
        self.pico_channel_range = "PS5000A_20V"
        self.pico_sample_interval_ns = 16
        self.pico_sample_unit = "PS5000A_NS"

        # Picoscope driver buffer settings (internal to the driver)
        self.pico_driver_buffer_size = 6_400_000  # Samples
        self.pico_driver_num_buffers = 1

        # Streaming settings
        self.pico_auto_stop = 0  # Don't auto stop
        self.pico_auto_stop_stream = False
        # --- End Configuration ---

        self.shutdown_event = threading.Event()
        data_queue = queue.Queue()
        empty_queue = queue.Queue()
        data_buffers = []

        # Creates an empty_queue that stores the indexes of empty buffers
        for idx in range(self.consumer_num_buffers):
            data_buffers.append(np.empty((self.consumer_buffer_size,), dtype="int16"))
            empty_queue.put(idx)

        self.pico_device = PicoDevice(
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
            1,  # down-sample ratio
            "PS5000A_RATIO_MODE_NONE",
            self.pico_auto_stop,
            self.pico_auto_stop_stream,
        )

        # Run streaming once to get the actual sample interval from the driver
        self.pico_device.run_streaming()

        # Get metadata from configured device and pass to consumer
        metadata = self.pico_device.get_metadata()
        self.consumer = Consumer(
            self.consumer_buffer_size,
            data_queue,
            empty_queue,
            data_buffers,
            output_file,
            self.shutdown_event,
            **metadata,
        )

        self.consumer_thread = threading.Thread(target=self.consumer.consume)
        self.pico_thread = threading.Thread(target=self.pico_device.run_capture)

        signal.signal(signal.SIGINT, self.signal_handler)

        # Optional live plotting
        self.enable_live_plot = enable_live_plot
        self.live_plotter = None
        self.qt_app = None
        self.start_time = None

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

    def signal_handler(self, sig, frame):
        logger.warning("Ctrl+C detected. Shutting down.")
        self.shutdown()

    def shutdown(self):
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

    def run(self):
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
    import argparse
    from datetime import datetime

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="PicoScope Data Acquisition")
    parser.add_argument(
        "--plot",
        "-p",
        action="store_true",
        help="Enable live plotting (requires PyQt5 and pyqtgraph)",
    )
    parser.add_argument(
        "--output", "-o", help="Output HDF5 file (default: auto-timestamped in /tmp/)"
    )
    parser.add_argument(
        "--plot-window",
        "-w",
        type=float,
        default=0.5,
        help="Set the live plot display window duration in seconds.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "--dec-fac", "-d", type=int, default=150, help="Decimation factor for plotting"
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
        args.output = f"/tmp/data_{timestamp}.hdf5"

    logger.info(f"Output file: {args.output}")

    # Create and run the streamer
    streamer = StreamExample(
        enable_live_plot=args.plot,
        output_file=args.output,
        debug=args.verbose,
        plot_window_s=args.plot_window,
        decimation_factor=args.dec_fac,
    )
    streamer.run()
