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

1.  **Install the Package**:
    This project is packaged with `pyproject.toml`. Install it in editable mode, which will also install all required dependencies:
    ```bash
    pip install -e .
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

After installation, the `picostream` command will be available in your environment.

### Basic Commands

**Acquisition with live plotting (default):**
```bash
picostream -o my_data.hdf5 --sample-rate 62.5
```

**Acquisition without live plotting:**
```bash
picostream --no-plot -o my_data.hdf5
```

**View an existing data file:**
The plotter can be run as a standalone tool to view any compatible HDF5 file.
```bash
python -m dfplot /path/to/your/data.hdf5
```

### Command-Line Arguments (`picostream`)

Run `picostream --help` for a full list of options.

-   `--sample-rate, -s`: Sample rate in MS/s (e.g., 62.5). Use 0 for max rate. [default: 20]
-   `--resolution, -b`: Resolution in bits. [default: 12, choices: 8, 12, 16]
-   `--range`: Voltage range in Volts. [default: 20.0]
-   `--plot / --no-plot, -p`: Enable/disable live plotting. [default: --plot]
-   `--output, -o`: Output HDF5 file (default: auto-timestamped).
-   `--plot-window, -w`: Live plot display window duration in seconds. [default: 0.5]
-   `--plot-pts`: Target number of points for the plot window. [default: 4000]
-   `--hardware-downsample`: Hardware down-sampling ratio. [default: 1]
-   `--downsample-mode`: Hardware down-sampling mode. [default: average, choices: average, aggregate]
-   `--verbose, -v`: Enable debug logging.

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

### Processing Large Files

For very large files that do not fit into memory, you can process the data in chunks. The `h5py` library makes this easy by allowing you to slice the dataset on disk without loading the entire file.

This example shows how to iterate through the data, converting it to millivolts and calculating the global minimum and maximum without ever holding the full dataset in RAM.

```python
import h5py
import numpy as np
from conversion_utils import adc_to_mV

# Define a chunk size suitable for your system's RAM
chunk_size = 10_000_000  # Process 10 million samples at a time

with h5py.File('my_data.hdf5', 'r') as f:
    dset = f['adc_counts']
    voltage_range = f.attrs['voltage_range_v']
    max_adc_val = f.attrs['max_adc']

    num_samples = dset.shape[0]

    # Initialize variables for analysis
    global_min_mv = float('inf')
    global_max_mv = float('-inf')

    print(f"Processing {num_samples:,} samples in chunks of {chunk_size:,}...")

    for i in range(0, num_samples, chunk_size):
        start_idx = i
        end_idx = min(i + chunk_size, num_samples)
        adc_chunk = dset[start_idx:end_idx]
        voltage_chunk_mv = adc_to_mV(adc_chunk, voltage_range, max_adc_val)

        global_min_mv = min(global_min_mv, voltage_chunk_mv.min())
        global_max_mv = max(global_max_mv, voltage_chunk_mv.max())
        print(f"  Processed chunk {start_idx:,} to {end_idx:,}")

    print("\nFinished processing.")
    print(f"Global voltage range: {global_min_mv:.2f} mV to {global_max_mv:.2f} mV")
```

## Troubleshooting

-   **"File not found" errors (plotter)**: Ensure the acquisition script (`picostream`) is running and has created the HDF5 file before the plotter tries to read it.
-   **Plotting window not appearing**: Verify `PyQt5` and `pyqtgraph` are installed. If using SSH, ensure X11 forwarding is enabled (`ssh -X user@host`).
-   **Performance Issues**: If the system is struggling, run acquisition with the `--no-plot` flag. You can also reduce the GUI workload by decreasing the number of plotted points with `--plot-pts`.


# Pico status

See https://www.picotech.com/helpfiles/pl1000-api/pico_statusvalues.html for table
