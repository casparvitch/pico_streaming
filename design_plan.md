# Picoscope Live Plotting Integration - Design & Implementation Plan

## Overview
Add real-time oscilloscope-style visualization to an existing Picoscope streaming system using an HDF5-based plotting approach for safe, robust monitoring at high data rates.

## Architecture Design

### Core Principles
- **Zero Impact on Acquisition**: Plotting completely decoupled from proven acquisition system
- **HDF5-Based Reading**: Read from existing HDF5 file stream for visualization
- **Min-Max Decimation**: Preserve transient information while reducing display points
- **Non-Blocking**: Plotting has zero impact on 15MS/s acquisition performance
- **Future-Ready**: Easy to add event detection and analysis later

### Component Architecture
```
┌─────────────────┐    buffers    ┌──────────────┐    HDF5    ┌─────────────────┐
│   PicoDevice    │ ────────────> │   Consumer   │ ─────────> │     dfplot      │
│ (Producer)      │               │ (Consumer)   │            │ (Plotter)       │
│ - Acquisition   │               │ - HDF5 Write │            │ - Read HDF5     │
│ - SDK Interface │               │              │            │ - Min-Max Decim │
└─────────────────┘               └──────────────┘            │ - PyQtGraph     │
                                                              └─────────────────┘
```

## Implementation Plan

### Phase 1: Core Infrastructure

#### 1.1 No Changes to Existing System
**PicoDevice (pico.py)**: No modifications required
**Consumer (consumer.py)**: No modifications required
**Main (main.py)**: Minimal changes for optional plotter instantiation

#### 1.2 Create HDF5 Live Plotter (dfplot.py)
**New File:** Complete HDF5-based oscilloscope plotter

**Key Features:**
- Periodic HDF5 file reading (every 100-200ms)
- Min-max decimation for efficient display
- Sliding window display of recent data
- PyQtGraph for high-performance rendering
- Independent operation from acquisition

**Core Structure:**
```python
class HDF5LivePlotter:
    def __init__(self, hdf5_path, update_interval_ms, display_window_seconds, decimation_factor):
        self.hdf5_path = hdf5_path
        self.data_start_sample = 0
        self.display_window_seconds = display_window_seconds
        
        # Setup PyQtGraph
        self.setup_plot()
        
        # Timer for periodic file reading
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_from_file)
        self.timer.start(update_interval_ms)
    
    def update_from_file(self):
        # Read new data from HDF5
        # Apply min-max decimation
        # Update sliding window display
    
    def min_max_decimate(self, data, factor=150):
        # Preserve peaks while reducing data points
        # 15M samples -> 100k display points
```

#### 1.3 Update Main Application (main.py)
**Minimal Changes:**
- Add optional HDF5 plotter instantiation
- Handle Qt event loop if plotting enabled

**Integration:**
```python
class StreamExample:
    def __init__(self, enable_live_plot=False):
        # ... existing setup unchanged ...
        
        # Optionally create independent plotter
        if enable_live_plot:
            self.live_plotter = HDF5LivePlotter(
                output_file,
                display_window_seconds=plot_window_s,
                decimation_factor=decimation_factor,
            )
    
    def run(self):
        # Existing acquisition threads unchanged
        # Handle Qt event loop if plotter enabled
```

### Phase 2: Testing & Optimization

#### 2.1 Performance Validation
- Verify zero impact on 15MS/s acquisition
- Test HDF5 concurrent read/write performance
- Measure plotting CPU/memory usage
- Optimize decimation algorithms

#### 2.2 Integration Testing
- Test with existing data files
- Verify file locking behavior
- Test startup/shutdown sequences

### Phase 3: Future Enhancements (Optional)

#### 3.1 Advanced Decimation Strategies
- Multi-resolution display (recent=high-res, older=low-res)
- Adaptive decimation based on data characteristics
- Peak detection preservation

#### 3.2 Analysis Features
- Event detection from HDF5 data
- Statistical analysis overlay
- Trigger-based data capture

#### 3.3 High-Performance Consumer
- For sample rates approaching the hardware limit, the Python-based HDF5 consumer may become a bottleneck.
- Future work could involve rewriting the consumer logic in a compiled language (e.g., C++ with native HDF5 libraries) to maximize I/O throughput and minimize the risk of buffer overflows.

## Technical Specifications

### Performance Targets
- **Update Rate**: Configurable, typically 5-10Hz.
- **Display Resolution**: Configurable via decimation factor.
- **Latency**: Dependent on buffer sizes and sample rate.
- **CPU Overhead**: Low, due to efficient plotting and decimation.

### Numerical Limitations
- **Data Latency**: A function of `consumer_buffer_size` / `sample_rate`.
- **File I/O**: Dependent on storage speed and data volume per update.
- **Decimation Ratio**: Configurable to balance visual fidelity and performance.
- **Display Window**: Configurable time window of recent data.

### Dependencies
```bash
pip install pyqtgraph PyQt5 h5py numpy numba loguru
```

### Configuration Parameters
All key parameters are configurable via command-line arguments in `main.py`.

## File Structure
```
project/
├── main.py                 # Orchestrator, handles CLI arguments
├── pico.py                 # Producer: Interfaces with hardware
├── consumer.py             # Consumer: Writes data to HDF5
├── dfplot.py               # Plotter: Visualizes data from HDF5
├── conversion_utils.py     # Numba-accelerated helper functions
└── requirements.txt        # Project dependencies
```

## Benefits of HDF5 Approach

### Advantages
1. **Zero Risk**: No changes to proven 15MS/s acquisition system
2. **Robust**: File I/O is well-tested, no threading complexity
3. **Testable**: Can develop and test independently with existing data
4. **Resumable**: Can restart plotting without losing data
5. **Flexible**: Easy to add sophisticated analysis features

### Trade-offs Accepted
- **Latency**: Some latency is inherent due to the buffered, file-based approach, but this is acceptable for monitoring.
- **I/O Overhead**: Periodic disk reads are required for plotting, but this is mitigated by OS file caching.

## Success Criteria
1. **Zero acquisition impact**: The acquisition rate is not negatively affected by enabling the plotter.
2. **Smooth visualization**: The plot updates at a consistent rate (e.g., 5-10Hz) without freezing.
3. **Transient preservation**: The min-max decimation algorithm effectively preserves signal peaks.
4. **Stable operation**: The application can run for extended periods without crashing or leaking memory.
5. **Independent operation**: The plotter can be started or stopped without affecting the core data acquisition process.

## Implementation Timeline
- **Day 1**: Create standalone HDF5 plotter with test data
- **Day 2**: Implement min-max decimation algorithms
- **Day 3**: Integration with main application as optional feature
- **Day 4**: Performance testing and optimization
- **Day 5**: Documentation and cleanup

This design provides effective "experiment is working" visualization while maintaining a robust, high-speed acquisition system.
