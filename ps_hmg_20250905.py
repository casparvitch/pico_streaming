import numpy as np
import matplotlib.pyplot as plt

from picostream.reader import PicoStreamReader

dp = "../data/20250905-haem-picostream/60nM.hdf5"


# Define a chunk size suitable for your system's RAM
chunk_size = 50_000_000


with PicoStreamReader(dp) as reader:
    print(f"File contains {reader.num_samples:,} samples.")
    print(f"Sample Interval: {reader.sample_interval_ns} ns")
    print(f"Voltage Range: ±{reader.voltage_range_v} V")
    print(f"Downsample Mode: {reader.downsample_mode}")

    # Initialize variables for analysis
    global_min_mv = float('inf')
    global_max_mv = float('-inf')

    print(f"\nProcessing {reader.num_samples:,} samples in chunks of {chunk_size:,}...")

    # The reader's iterator handles chunking, time axis generation, and voltage conversion.
    for i, (times, voltage_chunk_mv) in enumerate(reader.get_block_iter(chunk_size=chunk_size)):
        if voltage_chunk_mv.size > 0:
            global_min_mv = min(global_min_mv, voltage_chunk_mv.min())
            global_max_mv = max(global_max_mv, voltage_chunk_mv.max())

            # NOTE: This will create and show a new plot for every chunk.
            # The script will pause here until you close each plot window.
            print(f"Plotting chunk {i+1}...")
            fig, ax = plt.subplots()
            ax.plot(times, voltage_chunk_mv, lw=1)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Voltage (mV)")
            ax.set_title(f"Chunk {i+1} of {dp}")
            ax.grid(True)
            plt.show()

    print("\nFinished processing.")
    if np.isinf(global_min_mv):
        print("No data was processed.")
    else:
        print(f"Global voltage range: {global_min_mv:.2f} mV to {global_max_mv:.2f} mV")
