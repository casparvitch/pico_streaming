import queue
import signal
import threading
import numpy as np
import sys

from consumer import Consumer
from pico import PicoDevice


class StreamExample:

    def __init__(
        self, enable_live_plot=False, output_file="/tmp/data.hdf5", debug=False
    ):

        data_queue = queue.Queue()
        empty_queue = queue.Queue()
        buffer_size = 51200000
        num_buffers = 5
        data_buffers = []

        auto_stop = 0
        auto_stop_stream = False

        file_name = "data.npy"  # not in use

        # Creates an empty_queue that stores the indexes of empty buffers
        # The buffers are created as empty buffers and all added to the empty
        # queue ready to be filled with data by the producer
        for idx in range(num_buffers):
            data_buffers.append(np.empty((buffer_size,), dtype="int16"))
            empty_queue.put(idx)

        self.consumer = Consumer(
            buffer_size, data_queue, empty_queue, data_buffers, output_file
        )
        self.pico_device = PicoDevice(
            0,
            "PS5000A_DR_12BIT",
            640000,
            1,
            buffer_size,
            data_queue,
            empty_queue,
            data_buffers,
        )

        self.pico_device.set_channel(
            "setChA", "PS5000A_CHANNEL_A", 1, "PS5000A_DC", "PS5000A_20V", 0.0
        )
        self.pico_device.set_channel(
            "setChB", "PS5000A_CHANNEL_B", 0, "PS5000A_DC", "PS5000A_20V", 0.0
        )
        self.pico_device.set_data_buffer(
            "setDataBufferA", "PS5000A_CHANNEL_A", 0, "PS5000A_RATIO_MODE_NONE"
        )
        self.pico_device.configure_streaming_var(
            16,
            "PS5000A_NS",
            0,
            1,
            "PS5000A_RATIO_MODE_NONE",
            auto_stop,
            auto_stop_stream,
        )

        self.consumer_thread = threading.Thread(target=self.consumer.consume)
        self.pico_thread = threading.Thread(target=self.pico_device.run_capture)

        signal.signal(signal.SIGINT, self.signal_handler)

        # Optional live plotting
        self.enable_live_plot = enable_live_plot
        self.live_plotter = None
        self.qt_app = None

        if self.enable_live_plot:
            # Import Qt components only when needed
            from PyQt5.QtWidgets import QApplication
            from hdf5_live_plotter import HDF5LivePlotter

            # Create Qt application if it doesn't exist
            if not QApplication.instance():
                self.qt_app = QApplication(sys.argv)

            # Create the live plotter
            self.live_plotter = HDF5LivePlotter(output_file, debug=debug)

    def signal_handler(self, sig, frame):
        print("Stopping data acquisition/saving")

        # 1. Signal everything to stop
        self.consumer.stop()
        self.pico_device.running = False

        # 2. Stop plotter timer and close window
        if self.live_plotter:
            self.live_plotter.timer.stop()  # Stop the update timer
            self.live_plotter.close()  # Close the window

        # 3. Wait for threads to actually finish
        if hasattr(self, "consumer_thread"):
            self.consumer_thread.join(timeout=2.0)
        if hasattr(self, "pico_thread"):
            self.pico_thread.join(timeout=2.0)

        # 4. Now safe to close device
        self.pico_device.close_device()

        # 5. Quit Qt
        if self.qt_app:
            self.qt_app.quit()

    def run(self):
        # Start acquisition threads
        self.consumer_thread.start()
        self.pico_thread.start()

        # Handle Qt event loop if plotting is enabled
        if self.enable_live_plot and self.qt_app:
            # Show plotter window
            self.live_plotter.show()

            # Run acquisition monitoring in background thread
            import threading

            def acquisition_monitor():
                # Wait for acquisition to complete in background
                self.consumer_thread.join()
                self.pico_thread.join()
                # Notify user but keep plot open for examination
                print("\n🎯 Acquisition complete!")
                print("📊 Plot window shows captured data")
                print("💡 Close the plot window or press Ctrl+C to exit")

            monitor_thread = threading.Thread(target=acquisition_monitor, daemon=True)
            monitor_thread.start()

            # Run Qt event loop in main thread (blocking until quit)
            self.qt_app.exec_()
        else:
            # Original behavior for non-plotting mode
            self.consumer_thread.join()
            self.pico_thread.join()

        # Notify user of completion
        print("🎯 Acquisition complete!")
        if self.enable_live_plot:
            print("📊 Plot window remains open for data examination")
            print("   Close window or press Ctrl+C to exit")


if __name__ == "__main__":
    import argparse
    from datetime import datetime

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="PicoScope Data Acquisition")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Enable live plotting (requires PyQt5 and pyqtgraph)",
    )
    parser.add_argument(
        "--output", "-o", help="Output HDF5 file (default: auto-timestamped in /tmp/)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Auto-generate filename if not specified
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"/tmp/data_{timestamp}.hdf5"

    print(f"Output file: {args.output}")

    # Create and run the streamer
    streamer = StreamExample(
        enable_live_plot=args.plot, output_file=args.output, debug=args.debug
    )
    streamer.run()
