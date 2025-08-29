# Picoscope Live Plotting Integration - Design & Implementation Plan

## Overview
Add real-time oscilloscope-style visualization to existing picoscope streaming system using HDF5-based plotting approach for safe, robust monitoring at 15MS/s data rates.

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
│   PicoDevice    │ ────────────> │   Consumer   │ ─────────> │  HDF5LivePlotter│
│ (no changes)    │               │ (no changes) │            │ - Read HDF5     │
│ - Acquisition   │               │ - HDF5 Write │            │ - Min-Max Decim │
│ - No UI deps    │               │ - Unchanged  │            │ - PyQtGraph     │
└─────────────────┘               └──────────────┘            └─────────────────┘
                                                                       │
                                                                       ▼
                                                              ┌─────────────────┐
                                                              │ Future Modules  │
                                                              │ - Event Detect  │
                                                              │ - Statistics    │
                                                              │ - Triggers      │
                                                              └─────────────────┘
```

## Implementation Plan

### Phase 1: Core Infrastructure

#### 1.1 No Changes to Existing System
**PicoDevice (pico.py)**: No modifications required
**Consumer (consumer.py)**: No modifications required
**Main (main.py)**: Minimal changes for optional plotter instantiation

#### 1.2 Create HDF5 Live Plotter (hdf5_live_plotter.py)
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
    def __init__(self, hdf5_path='/tmp/data.hdf5', update_interval_ms=200):
        self.hdf5_path = hdf5_path
        self.last_read_position = 0
        self.display_window_samples = 15_000_000  # 1 second at 15MS/s
        
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
            self.live_plotter = HDF5LivePlotter('/tmp/data.hdf5')
    
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

## Technical Specifications

### Performance Targets
- **Update Rate**: 5-10Hz (every 100-200ms)
- **Display Resolution**: 100k-1M points after decimation
- **Latency**: 3.4 seconds behind real-time (acceptable for monitoring)
- **CPU Overhead**: <5% additional load
- **Memory Usage**: ~20MB for display buffers

### Numerical Limitations
- **Data Latency**: 3.4 seconds (51.2M buffer ÷ 15MS/s)
- **File I/O**: 100ms per HDF5 read operation (102MB)
- **Decimation Ratio**: 15:1 to 150:1 (15M samples → 100k-1M points)
- **Display Window**: 0.5-2 seconds of data visible

### Dependencies
```bash
pip install pyqtgraph PyQt5 h5py numpy
```

### Configuration Parameters
```python
# Recommended settings
UPDATE_INTERVAL_MS = 200    # 5Hz file reading
DISPLAY_WINDOW_SEC = 1.0    # 1 second visible window
DECIMATION_FACTOR = 150     # 15M -> 100k points
HDF5_PATH = '/tmp/data.hdf5'
```

## File Structure
```
project/
├── main.py                 # Minimal changes: Optional plotter
├── pico.py                 # Unchanged
├── consumer.py             # Unchanged
├── hdf5_live_plotter.py    # New: HDF5-based plotter
└── requirements.txt        # Updated: Add PyQtGraph dependencies
```

## Benefits of HDF5 Approach

### Advantages
1. **Zero Risk**: No changes to proven 15MS/s acquisition system
2. **Robust**: File I/O is well-tested, no threading complexity
3. **Testable**: Can develop and test independently with existing data
4. **Resumable**: Can restart plotting without losing data
5. **Flexible**: Easy to add sophisticated analysis features

### Trade-offs Accepted
- **Latency**: 3.4 seconds behind real-time (acceptable for monitoring)
- **I/O Overhead**: Periodic disk reads (minimal impact on modern SSDs)
- **Buffer Size Constraint**: Cannot reduce latency without affecting acquisition

## Success Criteria
1. **Zero acquisition impact**: 15MS/s rate maintained unchanged
2. **Smooth visualization**: 5-10Hz refresh rate with fluid scrolling
3. **Transient preservation**: Min-max decimation preserves peaks
4. **Stable operation**: 20-minute runs without issues
5. **Independent operation**: Plotter can start/stop without affecting acquisition

## Implementation Timeline
- **Day 1**: Create standalone HDF5 plotter with test data
- **Day 2**: Implement min-max decimation algorithms
- **Day 3**: Integration with main application as optional feature
- **Day 4**: Performance testing and optimization
- **Day 5**: Documentation and cleanup

This design provides effective "experiment is working" visualization while maintaining the proven high-speed acquisition system completely unchanged.
