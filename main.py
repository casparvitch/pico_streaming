import queue
import signal
import threading
import numpy as np
import sys

from consumer import Consumer
from pico import PicoDevice


class StreamExample():

    def __init__(self, enable_live_plot=False):


class StreamExample():

    def __init__(self, enable_live_plot=False):

        data_queue = queue.Queue()
        empty_queue = queue.Queue()
        buffer_size = 51200000  
        num_buffers = 5
        data_buffers = []

        auto_stop = 0
        auto_stop_stream = False

        file_name = 'data.npy' #not in use

        # Creates an empty_queue that stores the indexes of empty buffers
        # The buffers are created as empty buffers and all added to the empty 
        # queue ready to be filled with data by the producer
        for idx in range(num_buffers):
            data_buffers.append(np.empty((buffer_size,), dtype='int16'))
            empty_queue.put(idx)
        
        self.consumer = Consumer(buffer_size,data_queue,empty_queue,data_buffers,file_name)
        self.pico_device = PicoDevice(0,"PS5000A_DR_12BIT",640000,1,buffer_size,data_queue,empty_queue,data_buffers)
        
        self.pico_device.set_channel('setChA','PS5000A_CHANNEL_A',1,'PS5000A_DC','PS5000A_20V',0.0)
        self.pico_device.set_channel('setChB','PS5000A_CHANNEL_B',0,'PS5000A_DC','PS5000A_20V',0.0)
        self.pico_device.set_channel('setChC','PS5000A_CHANNEL_C',0,'PS5000A_DC','PS5000A_20V',0.0)
        self.pico_device.set_channel('setChD','PS5000A_CHANNEL_D',0,'PS5000A_DC','PS5000A_20V',0.0)
        self.pico_device.set_data_buffer('setDataBufferA','PS5000A_CHANNEL_A',0,'PS5000A_RATIO_MODE_NONE')
        self.pico_device.configure_streaming_var(16,'PS5000A_NS',0,1,'PS5000A_RATIO_MODE_NONE',auto_stop, auto_stop_stream)
        
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
            self.live_plotter = HDF5LivePlotter('/tmp/data.hdf5')

    def signal_handler(self, sig, frame):
        print("Stopping data acquisition/saving")
        self.consumer.stop()
        self.pico_device.running = False
        self.pico_device.close_device()
        
        # Close plotter if running
        if self.live_plotter:
            self.live_plotter.close()
        
        # Quit Qt application if we created it
        if self.qt_app:
            self.qt_app.quit()

    def run(self):
        # Start acquisition threads
        self.consumer_thread.start()
        self.pico_thread.start()
        
        # Show plotter if enabled
        if self.live_plotter:
            self.live_plotter.show()
        
        # Handle Qt event loop if plotting is enabled
        if self.enable_live_plot and self.qt_app:
            # Run Qt event loop in a separate thread to avoid blocking
            import threading
            from PyQt5.QtCore import QTimer
            
            def qt_event_loop():
                # Process Qt events periodically
                timer = QTimer()
                timer.timeout.connect(lambda: None)  # Keep event loop alive
                timer.start(100)  # 100ms intervals
                self.qt_app.exec_()
            
            qt_thread = threading.Thread(target=qt_event_loop, daemon=True)
            qt_thread.start()
        
        # Wait for acquisition threads to complete
        self.consumer_thread.join()        
        self.pico_thread.join()
        
        # Clean shutdown of Qt if we created it
        if self.qt_app:
            self.qt_app.quit()


if __name__ == '__main__':
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='PicoScope Data Acquisition')
    parser.add_argument('--plot', action='store_true', 
                       help='Enable live plotting (requires PyQt5 and pyqtgraph)')
    args = parser.parse_args()
    
    # Create and run the streamer
    streamer = StreamExample(enable_live_plot=args.plot)
    streamer.run()
