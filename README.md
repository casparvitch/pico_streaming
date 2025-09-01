# PicoScope High-Speed Data Acquisition with Live Plotting

Real-time 15MS/s data acquisition from PicoScope with optional oscilloscope-style visualization using HDF5-based plotting for zero-risk operation.

## Features

- **High-Speed Acquisition**: 15MS/s streaming from PicoScope to HDF5
- **Zero-Risk Architecture**: Plotting completely independent of acquisition system
- **Real-Time Visualization**: Oscilloscope-style display with min-max decimation
- **Robust Buffer Management**: 5 buffers of 51.2M samples for smooth data flow
- **Optional Plotting**: Run with or without visualization as needed

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

**Acquisition Only** (proven 15MS/s performance):
```bash
python main.py
```

**Acquisition + Live Plotting**:
```bash
python main.py --plot
```

**View Existing Data**:
```bash
python hdf5_live_plotter.py /path/to/data.hdf5
```

## System Architecture

```
PicoScope → pico.py → buffers → consumer.py → /tmp/data.hdf5
                                                    ↓
                                            hdf5_live_plotter.py → Real-time Display
```

### Core Components

- **`pico.py`**: PicoScope hardware interface and data acquisition
- **`consumer.py`**: High-speed buffer-to-HDF5 writer
- **`main.py`**: Orchestrates acquisition threads with optional plotting
- **`hdf5_live_plotter.py`**: Independent HDF5-based oscilloscope visualization

## Performance Specifications

### Acquisition System
- **Sample Rate**: 15MS/s (16ns intervals)
- **Buffer Size**: 51.2M samples per buffer
- **Number of Buffers**: 5 (256M samples total)
- **Data Latency**: 3.4 seconds (buffer size ÷ sample rate)
- **Output Format**: HDF5 with metadata

### Visualization System
- **Update Rate**: 5Hz (200ms intervals)
- **Display Points**: ~100k after min-max decimation
- **Decimation Ratio**: 150:1 (preserves transients)
- **Display Window**: 1 second of recent data
- **CPU Overhead**: <5% additional load

## Configuration

### Hardware Setup
The system is configured for:
- **Channel A**: Enabled, 20V range, DC coupling
- **Channels B,C,D**: Disabled
- **Resolution**: 12-bit
- **Trigger**: Free-running (no trigger)

### File Output
- **Default Path**: `/tmp/data.hdf5`
- **Format**: HDF5 with datasets:
  - `adc_counts`: Raw ADC values (int16)
  - `metadata`: Acquisition parameters

### Plotting Parameters
- **Update Interval**: 200ms (configurable)
- **Decimation Factor**: 150 (configurable)
- **Display Window**: 15M samples (1 second at 15MS/s)

## Advanced Usage

### Standalone Plotting
View any existing HDF5 data file:
```bash
python hdf5_live_plotter.py /path/to/your/data.hdf5
```

### Custom Configuration
Modify parameters in the source files:

**Acquisition (main.py)**:
```python
buffer_size = 51200000      # Samples per buffer
num_buffers = 5             # Number of buffers
```

**Plotting (hdf5_live_plotter.py)**:
```python
update_interval_ms = 200    # Refresh rate
decimation_factor = 150     # Data reduction ratio
display_window_samples = 15_000_000  # Visible timespan
```

## Data Analysis

### HDF5 File Structure
```python
import h5py
with h5py.File('/tmp/data.hdf5', 'r') as f:
    data = f['adc_counts'][:]           # Raw ADC counts
    metadata = f['metadata'].attrs      # Acquisition parameters
    
    # Convert to voltage
    from picosdk.functions import adc2mV
    voltage = adc2mV(data, metadata['chARange'], metadata['maxADC'])
```

### Min-Max Decimation Algorithm
The plotter uses min-max decimation to preserve transient information:
- Groups samples into blocks of `decimation_factor` size
- Keeps both minimum and maximum from each block
- Interleaves min/max values to preserve peak information
- Reduces 15M samples to ~100k display points

## Troubleshooting

### Common Issues

**"File not found" errors**:
- Ensure acquisition is running and creating `/tmp/data.hdf5`
- Check file permissions in `/tmp/` directory

**Plotting window not appearing**:
- Verify PyQt5 and pyqtgraph installation: `pip install PyQt5 pyqtgraph`
- Check X11 forwarding if using SSH: `ssh -X username@host`

**Performance issues**:
- Disable plotting for maximum acquisition performance: `python main.py`
- Monitor CPU usage and adjust decimation factor if needed

### Performance Monitoring
The system reports key metrics:
- Buffer queue status (producer/consumer balance)
- Maximum samples per callback
- File write performance
- Plotting update rates

## Development

### Adding New Features
The modular architecture makes it easy to add:
- **Event Detection**: Analyze HDF5 data for triggers/events
- **Multiple Channels**: Extend plotting for channels B,C,D
- **Data Export**: Add CSV/MATLAB export functionality
- **Remote Monitoring**: Network-based plotting clients

### Testing
Test individual components:
```bash
# Test acquisition only
python main.py

# Test plotting with existing data
python hdf5_live_plotter.py /tmp/data.hdf5

# Test full system
python main.py --plot
```

## Dependencies

- `numpy`: Numerical computing
- `h5py`: HDF5 file I/O
- `picosdk`: PicoScope hardware interface
- `pyqtgraph`: High-performance plotting
- `PyQt5`: GUI framework
- `matplotlib`: Additional plotting support

## License

This project interfaces with PicoScope hardware using the official PicoSDK. Ensure you have appropriate licenses for PicoScope software and hardware.

## Support

For issues related to:
- **PicoScope Hardware**: Contact Pico Technology support
- **Data Acquisition**: Check buffer sizes and sample rates
- **Visualization**: Verify Qt/graphics system compatibility
