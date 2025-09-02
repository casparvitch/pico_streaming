# PicoScope High-Speed Data Acquisition with Live Plotting

A high-performance, multi-threaded application for streaming data from a PicoScope to an HDF5 file, with an optional, decoupled live visualization.

## Features

- **Robust Producer-Consumer Architecture**: Separates data acquisition from disk I/O using a large, shared memory buffer pool to prevent data loss.
- **Zero-Risk Live Plotting**: The plotter reads from the HDF5 file, not the live data stream. This ensures that a slow or crashing GUI cannot interfere with data acquisition.
- **Efficient Visualization**: Uses `pyqtgraph` and a Numba-accelerated min-max decimation algorithm to display large datasets with minimal CPU impact.

## System Architecture

```
PicoScope → pico.py (Producer) → Data Buffers → consumer.py (Consumer) → /data.hdf5
                                                                              ↓
                                                                      dfplot.py (Plotter) → Real-time Display
```

### Core Components

- **`main.py`**: Orchestrates the producer, consumer, and plotter threads. Handles command-line arguments and shutdown.
- **`pico.py`**: The "producer" thread. Interfaces with the PicoScope hardware via the C SDK and places data into shared buffers.
- **`consumer.py`**: The "consumer" thread. Retrieves full data buffers and writes them efficiently to an HDF5 file.
- **`dfplot.py`**: A `PyQt5` application for live visualization. Can be run as part of the main application or as a standalone viewer for existing HDF5 files.
- **`conversion_utils.py`**: Numba-accelerated helper functions for data conversion and decimation.

## Installation

1.  **Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **PicoSDK Setup**:
    You must install the official PicoSDK from Pico Technology. On Linux, you may need to perform additional steps. The following command was required on one system to fix a library issue:
    ```bash
    # This command may be needed if you encounter library loading errors.
    # It modifies the executable stack permissions of a PicoSDK library.
    sudo execstack -c /opt/picoscope/lib/libpicocv.so
    ```
    For more information, see these resources:
    - [PicoTech Forum Thread on Linux issues](https://www.picotech.com/support/viewtopic.php?t=43125&hilit=linux)
    - [Arch Linux PicoScope Package](https://aur.archlinux.org/packages/picoscope7)

## Usage

The application is controlled via command-line arguments to `main.py`.

### Basic Commands

**Acquisition without live plotting:**
```bash
python main.py -o my_data.hdf5
```

**Acquisition with live plotting:**
```bash
python main.py --plot -o my_data.hdf5
```

**View an existing data file:**
```bash
python dfplot.py /path/to/your/data.hdf5
```

### Command-Line Arguments

-   `--rate` / `-r`: Sample rate in MS/s. Default: `62.5`. Use `0` for the maximum possible rate.
-   `--plot` / `-p`: Enable the live plot window.
-   `--output` / `-o`: Path to the output HDF5 file. Default: `/tmp/data_YYYYMMDD_HHMMSS.hdf5`.
-   `--plot-window` / `-w`: The time duration (in seconds) to display in the plot window. Default: `0.5`.
-   `--dec-fac` / `-d`: The decimation factor for plotting, which controls how many points are grouped for min/max calculation. Higher values reduce plot density. Default: `150`.
-   `--verbose` / `-v`: Enable detailed `DEBUG` level logging.

## Data Analysis

The output HDF5 file contains the raw ADC counts and metadata attributes needed for conversion.

### HDF5 File Structure

-   **Dataset**: `adc_counts` (1D array of `int16`) - The raw sample values from the ADC.
-   **Attributes**: Attached to the root of the file. Key attributes include:
    -   `sample_interval_ns`: The time between samples in nanoseconds.
    -   `voltage_range_v`: The configured single-sided voltage range (e.g., `20.0` for ±20V).
    -   `max_adc`: The maximum integer value of the ADC (e.g., `32767`).

### Example Python Analysis

```python
import h5py
import numpy as np
from conversion_utils import adc_to_mV

# Open the HDF5 file
with h5py.File('my_data.hdf5', 'r') as f:
    # Load data and metadata
    adc_counts = f['adc_counts'][:]
    voltage_range = f.attrs['voltage_range_v']
    max_adc_val = f.attrs['max_adc']

    # Convert raw ADC counts to millivolts
    voltage_mv = adc_to_mV(adc_counts, voltage_range, max_adc_val)
    
    print(f"Successfully loaded {len(voltage_mv)} samples.")
    print(f"Voltage range: {voltage_mv.min():.2f} mV to {voltage_mv.max():.2f} mV")
```

## Troubleshooting

-   **"File not found" errors (plotter)**: Ensure the acquisition script (`main.py`) is running and has created the HDF5 file before the plotter tries to read it.
-   **Plotting window not appearing**: Verify `PyQt5` and `pyqtgraph` are installed. If using SSH, ensure X11 forwarding is enabled (`ssh -X user@host`).
-   **Performance Issues**: If the system is struggling, run acquisition without the `--plot` flag. You can also increase the plotting decimation factor (`--dec-fac`) to reduce the GUI workload.

## Dependencies

-   `numpy`: Numerical computing
-   `h5py`: HDF5 file I/O
-   `picosdk`: Official PicoScope Python SDK
-   `pyqtgraph`: High-performance plotting
-   `PyQt5`: GUI framework
-   `loguru`: Clean and simple logging
-   `numba`: JIT compiler for performance-critical functions

## License

This project interfaces with PicoScope hardware using the official PicoSDK. Ensure you have appropriate licenses for PicoScope software and hardware.
