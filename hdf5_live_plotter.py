import sys
import time
import numpy as np
import h5py
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QHBoxLayout
from PyQt5.QtCore import QTimer, pyqtSignal, QObject
import pyqtgraph as pg
from picosdk.functions import adc2mV
from picosdk.ps5000a import ps5000a as ps


class HDF5LivePlotter(QMainWindow):
    """
    Real-time oscilloscope-style plotter that reads from HDF5 files.
    Completely independent of acquisition system for zero-risk operation.
    """
    
    def __init__(self, hdf5_path='/tmp/data.hdf5', update_interval_ms=200):
        super().__init__()
        
        # Configuration
        self.hdf5_path = hdf5_path
        self.update_interval_ms = update_interval_ms
        self.last_read_position = 0
        self.display_window_samples = 15_000_000  # 1 second at 15MS/s
        self.decimation_factor = 150  # 15M -> 100k display points
        
        # Data storage
        self.display_data = np.array([])
        self.time_data = np.array([])
        
        # Metadata from HDF5
        self.sample_interval_ns = 16  # Default, will be read from file
        self.ch_range = None
        self.max_adc = None
        
        # Setup UI
        self.setup_ui()
        
        # Setup update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_from_file)
        self.timer.start(self.update_interval_ms)
        
        # Initial file check
        self.check_file_exists()
    
    def setup_ui(self):
        """Setup the oscilloscope-style UI"""
        self.setWindowTitle('PicoScope Live Plotter - HDF5 Reader')
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Status bar
        status_layout = QHBoxLayout()
        self.status_label = QLabel('Status: Waiting for data...')
        self.samples_label = QLabel('Samples: 0')
        self.rate_label = QLabel('Rate: 0 MS/s')
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.samples_label)
        status_layout.addWidget(self.rate_label)
        status_layout.addStretch()
        layout.addLayout(status_layout)
        
        # Plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Voltage', 'mV')
        self.plot_widget.setLabel('bottom', 'Time', 's')
        self.plot_widget.setTitle('Channel A - Live Data')
        self.plot_widget.showGrid(x=True, y=True)
        
        # Plot curve
        self.curve = self.plot_widget.plot(pen='y', width=1)
        
        layout.addWidget(self.plot_widget)
        
        # Performance optimization
        self.plot_widget.setDownsampling(mode='peak')
        self.plot_widget.setClipToView(True)
    
    def check_file_exists(self):
        """Check if HDF5 file exists and is readable"""
        try:
            with h5py.File(self.hdf5_path, 'r') as f:
                if 'adc_counts' in f:
                    self.status_label.setText('Status: File found, reading...')
                    self.read_metadata(f)
                else:
                    self.status_label.setText('Status: File exists but no data yet')
        except (FileNotFoundError, OSError):
            self.status_label.setText(f'Status: Waiting for {self.hdf5_path}')
    
    def read_metadata(self, hdf5_file):
        """Read metadata from HDF5 file"""
        try:
            if 'metadata' in hdf5_file:
                metadata = hdf5_file['metadata']
                self.sample_interval_ns = metadata.attrs.get('timeIntervalns', 16)
                self.ch_range = metadata.attrs.get('chARange', None)
                self.max_adc = metadata.attrs.get('maxADC', None)
        except Exception as e:
            print(f"Warning: Could not read metadata: {e}")
    
    def update_from_file(self):
        """Periodically read new data from HDF5 file"""
        try:
            with h5py.File(self.hdf5_path, 'r') as f:
                if 'adc_counts' not in f:
                    return
                
                dataset = f['adc_counts']
                current_size = dataset.shape[0]
                
                # Check if new data is available
                if current_size <= self.last_read_position:
                    return
                
                # Read metadata if not already done
                if self.ch_range is None:
                    self.read_metadata(f)
                
                # Read new data
                new_data = dataset[self.last_read_position:current_size]
                self.last_read_position = current_size
                
                # Update display
                self.update_display(new_data)
                
                # Update status
                self.samples_label.setText(f'Samples: {current_size:,}')
                elapsed_time = time.time() - getattr(self, 'start_time', time.time())
                if not hasattr(self, 'start_time'):
                    self.start_time = time.time()
                rate_ms = (current_size / elapsed_time / 1_000_000) if elapsed_time > 0 and current_size > 0 else 0
                self.rate_label.setText(f'Rate: {rate_ms:.1f} MS/s')
                self.status_label.setText('Status: Live streaming')
                
        except (FileNotFoundError, OSError):
            self.status_label.setText(f'Status: File not found - {self.hdf5_path}')
        except Exception as e:
            self.status_label.setText(f'Status: Error reading file - {str(e)}')
    
    def update_display(self, new_data):
        """Update the oscilloscope display with new data"""
        if len(new_data) == 0:
            return
        
        # Append new data to display buffer
        self.display_data = np.concatenate([self.display_data, new_data])
        
        # Maintain sliding window
        if len(self.display_data) > self.display_window_samples:
            excess = len(self.display_data) - self.display_window_samples
            self.display_data = self.display_data[excess:]
        
        # Apply decimation for display
        decimated_data = self.min_max_decimate(self.display_data, self.decimation_factor)
        
        # Convert to voltage if we have calibration data
        if self.ch_range is not None and self.max_adc is not None:
            try:
                voltage_data = adc2mV(decimated_data, self.ch_range, self.max_adc)
            except:
                voltage_data = decimated_data.astype(float)  # Fallback to raw ADC
        else:
            voltage_data = decimated_data.astype(float)
        
        # Create time axis
        time_axis = self.create_time_axis(len(voltage_data))
        
        # Update plot
        self.curve.setData(time_axis, voltage_data)
        
        # Auto-scale occasionally
        if len(self.display_data) % 1000 == 0:
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
        grouped_data = data[:n_complete_groups * factor].reshape(-1, factor)
        
        # Get min and max for each group
        mins = np.min(grouped_data, axis=1)
        maxs = np.max(grouped_data, axis=1)
        
        # Interleave mins and maxs to preserve transients
        decimated = np.empty(n_complete_groups * 2, dtype=data.dtype)
        decimated[0::2] = mins
        decimated[1::2] = maxs
        
        # Add any remaining samples
        remainder = data[n_complete_groups * factor:]
        if len(remainder) > 0:
            decimated = np.concatenate([decimated, remainder])
        
        return decimated
    
    def create_time_axis(self, n_samples):
        """Create time axis in seconds"""
        # Time per sample in seconds
        time_per_sample = self.sample_interval_ns * 1e-9
        
        # For decimated data, we need to account for the decimation
        # Each pair of decimated points represents 'decimation_factor' original samples
        effective_time_per_point = time_per_sample * (self.decimation_factor / 2)
        
        return np.arange(n_samples) * effective_time_per_point
    
    def closeEvent(self, event):
        """Clean shutdown"""
        self.timer.stop()
        event.accept()


def main():
    """Standalone application entry point"""
    app = QApplication(sys.argv)
    
    # Command line argument for HDF5 file path
    hdf5_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/data.hdf5'
    
    plotter = HDF5LivePlotter(hdf5_path)
    plotter.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
