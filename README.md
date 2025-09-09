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

The output HDF5 file contains raw ADC counts and metadata. To simplify reading this data, the `PicoStreamReader` helper class is provided in `picostream/reader.py`.

### Using `PicoStreamReader`

The `PicoStreamReader` class is the recommended way to work with data files produced by `picostream`. It handles:
- Opening the HDF5 file and reading all metadata.
- Converting raw ADC values to millivolts, including applying the analog offset.
- Generating the correct time axis in seconds.
- Providing a simple iterator to process large files in memory-efficient chunks.

Here is an example of how to use it to process a large file:

```python
import numpy as np
from picostream.reader import PicoStreamReader

# Define a chunk size suitable for your system's RAM
chunk_size = 10_000_000  # Process 10 million samples at a time

# Use the reader as a context manager
with PicoStreamReader('my_data.hdf5') as reader:
    # Metadata is available as attributes after opening
    sample_rate_sps = 1e9 / reader.sample_interval_ns
    print(f"File contains {reader.num_samples:,} samples.")
    print(f"Sample rate: {sample_rate_sps / 1e6:.2f} MS/s")
    print(f"Voltage range: ±{reader.voltage_range_v} V")
    print(f"Analog offset: {reader.analog_offset_v} V")

    # Initialize variables for analysis
    global_min_mv = float('inf')
    global_max_mv = float('-inf')

    print(f"\nProcessing data...")

    # Iterate through the file in chunks.
    # The reader yields (time_array, voltage_array_mv) tuples.
    for times, voltages_mv in reader.get_block_iter(chunk_size=chunk_size):
        if voltages_mv.size > 0:
            # Process the data chunk here (e.g., find min/max)
            global_min_mv = min(global_min_mv, voltages_mv.min())
            global_max_mv = max(global_max_mv, voltages_mv.max())
        
    print("Finished processing.")
    if np.isinf(global_min_mv):
        print("No data was found in the file.")
    else:
        print(f"Global voltage range: {global_min_mv:.2f} mV to {global_max_mv:.2f} mV")
```

## Troubleshooting

-   **"File not found" errors (plotter)**: Ensure the acquisition script (`picostream`) is running and has created the HDF5 file before the plotter tries to read it.
-   **Plotting window not appearing**: Verify `PyQt5` and `pyqtgraph` are installed. If using SSH, ensure X11 forwarding is enabled (`ssh -X user@host`).
-   **Performance Issues**: If the system is struggling, run acquisition with the `--no-plot` flag. You can also reduce the GUI workload by decreasing the number of plotted points with `--plot-pts`.


# Pico status

See https://www.picotech.com/helpfiles/pl1000-api/pico_statusvalues.html for table
