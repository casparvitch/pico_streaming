# Picoscope Live Plotting Integration - Design & Implementation Plan

## Overview
Add real-time oscilloscope-style visualization to existing picoscope streaming system using PyQtGraph for high-performance plotting at 15MS/s data rates.

## Architecture Design

### Core Principles
- **Separation of Concerns**: Device handles acquisition, plotter handles visualization
- **Observer Pattern**: Device notifies registered callbacks of new data
- **Frame-Based Display**: Show complete time windows (50-100ms) at 20-50Hz refresh
- **Non-Blocking**: Plotting must not impact 15MS/s acquisition performance
- **Future-Ready**: Easy to add event detection and analysis later

### Component Architecture
```
┌─────────────────┐    data_callback    ┌──────────────────────┐
│   PicoDevice    │ ──────────────────> │ PicoOscilloscopePlotter │
│                 │                     │                      │
│ - Acquisition   │                     │ - PyQtGraph Display  │
│ - Callbacks[]   │                     │ - Frame Buffering    │
│ - No UI deps    │                     │ - ADC Conversion     │
└─────────────────┘                     └──────────────────────┘
        │
        │ data_callback
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

#### 1.1 Modify PicoDevice (pico.py)
**Changes:**
- Add callback registration system
- Modify streaming_callback to notify observers
- Remove any plotting dependencies

**Code Changes:**
```python
class PicoDevice:
    def __init__(self, ..., data_callbacks=None):
        self.data_callbacks = data_callbacks or []
    
    def add_data_callback(self, callback_func):
        self.data_callbacks.append(callback_func)
    
    def streaming_callback(self, handle, noOfSamples, startIndex, overflow, ...):
        # ... existing buffer management ...
        
        # Notify observers
        if self.running and noOfSamples > 0:
            new_data = self.bufferA[startIndex:startIndex + noOfSamples].copy()
            timestamp = time.time()
            
            for callback in self.data_callbacks:
                try:
                    callback(new_data, timestamp)
                except Exception as e:
                    logger.error(f"Data callback failed: {e}")
```

#### 1.2 Create PyQtGraph Plotter (pico_qt_plotter.py)
**New File:** Complete oscilloscope-style plotter

**Key Features:**
- Frame-based data accumulation
- Numba-accelerated ADC-to-voltage conversion
- Thread-safe queue for data handling
- 20-50Hz refresh rate
- Auto-scaling Y-axis

**Core Structure:**
```python
class PicoOscilloscopeQtPlotter:
    def __init__(self, frame_duration_ms=50, update_rate_hz=20):
        # PyQtGraph setup
        # Frame buffering
        # Qt timer for updates
    
    def data_callback(self, data_array, timestamp):
        # Receive data from device
        # Queue for processing
    
    def add_data_chunk(self, data_array, timestamp):
        # Accumulate frame data
        # Convert ADC to voltage
    
    def update_plot(self):
        # Update PyQtGraph display
        # Auto-scale axes
```

#### 1.3 Update Main Application (main.py)
**Changes:**
- Create plotter independently
- Register as observer
- Handle Qt event loop

**Integration:**
```python
class StreamExample:
    def __init__(self, enable_live_plot=True):
        # ... existing setup ...
        
        # Create device (no plotting knowledge)
        self.pico_device = PicoDevice(...)
        
        # Create plotter independently
        if enable_live_plot:
            self.live_plotter = PicoOscilloscopeQtPlotter()
            self.pico_device.add_data_callback(self.live_plotter.data_callback)
    
    def run(self):
        # Start acquisition threads
        # Handle Qt event loop
        # Graceful shutdown
```

### Phase 2: Testing & Optimization

#### 2.1 Performance Validation
- Verify no impact on 15MS/s acquisition
- Test with concurrent video feed
- Measure CPU/memory usage
- Optimize frame duration and refresh rate

#### 2.2 Integration Testing
- Test startup/shutdown sequences
- Verify thread safety
- Test error handling and recovery

### Phase 3: Future Enhancements (Optional)

#### 3.1 Event Detection Framework
```python
class EventDetector:
    def data_callback(self, data_array, timestamp):
        # Simple threshold-based detection
        # Log events for HDF5 analysis
```

#### 3.2 Trigger System
```python
class TriggerSystem:
    def data_callback(self, data_array, timestamp):
        # Capture snapshots around events
        # Save event contexts
```

## Technical Specifications

### Performance Targets
- **Refresh Rate**: 20-50 Hz
- **Frame Duration**: 50-100ms (750k-1.5M samples)
- **CPU Overhead**: <5% additional load
- **Memory Usage**: ~50MB for plotting buffers
- **Latency**: <100ms from acquisition to display

### Dependencies
```bash
pip install pyqtgraph PyQt5 numba
```

### Configuration Parameters
```python
# Recommended settings
FRAME_DURATION_MS = 50      # 50ms frames
UPDATE_RATE_HZ = 20         # 20Hz refresh
QUEUE_SIZE = 5              # Small queue to prevent lag
MAX_VOLTAGE_RANGE = 20000   # 20V in mV
```

## File Structure
```
project/
├── main.py                 # Modified: Observer registration
├── pico.py                 # Modified: Callback system
├── consumer.py             # Unchanged
├── pico_qt_plotter.py      # New: PyQtGraph plotter
└── requirements.txt        # Updated: Add PyQt dependencies
```

## Risk Mitigation

### Performance Risks
- **Queue overflow**: Use bounded queue with drop-on-full
- **Qt event loop blocking**: Process events in small chunks
- **Memory pressure**: Monitor frame buffer sizes

### Integration Risks
- **Thread safety**: Use proper locking for shared data
- **Startup timing**: Ensure device starts before plotter
- **Shutdown sequence**: Stop plotter before device

## Success Criteria
1. **No acquisition impact**: 15MS/s rate maintained
2. **Smooth visualization**: 20+ Hz refresh rate
3. **Transient preservation**: Full resolution in frames
4. **Stable operation**: 20-minute runs without issues
5. **Clean architecture**: Easy to add future features

## Implementation Timeline
- **Day 1**: Modify PicoDevice callback system
- **Day 2**: Create PyQtGraph plotter
- **Day 3**: Integration and testing
- **Day 4**: Performance optimization
- **Day 5**: Documentation and cleanup

This design provides immediate "experiment is working" visualization while maintaining high-speed acquisition performance and enabling future analysis capabilities.
