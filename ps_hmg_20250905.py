import h5py
import numpy as np
import matplotlib.pyplot as plt

from picostream.conversion_utils import adc_to_mV

dp = "../data/20250905-haem-picostream/60nM.hdf5"


# Define a chunk size suitable for your system's RAM
chunk_size = 50_000_000 


with h5py.File(dp, 'r') as f:
    dset = f['adc_counts']
    print({i: f.attrs[i] for i in f.attrs})
    voltage_range = f.attrs['voltage_range_v']
    res_str = f.attrs['resolution']
    res_int = int(res_str[11:-3])
    max_adc_val = (2**(res_int-1)) - 1 

    num_samples = dset.shape[0]

    # Initialize variables for analysis
    global_min_mv = float('inf')
    global_max_mv = float('-inf')

    print(f"Processing {num_samples:,} samples in chunks of {chunk_size:,}...")

    sample_interval_ns = f.attrs['sample_interval_ns']
    hardware_downsample_ratio = f.attrs['hardware_downsample_ratio']
    
    effective_interval_ns = sample_interval_ns * hardware_downsample_ratio

    
    for i in range(0, num_samples, chunk_size):
        start_idx = i
        end_idx = min(i + chunk_size, num_samples)
        adc_chunk = dset[start_idx:end_idx]
        voltage_chunk_mv = adc_to_mV(adc_chunk, voltage_range, max_adc_val)

        global_min_mv = min(global_min_mv, voltage_chunk_mv.min())
        global_max_mv = max(global_max_mv, voltage_chunk_mv.max())
        # print(f"  Processed chunk {start_idx:,} to {end_idx:,}")

        times = np.linspace(start_idx * effective_interval_ns, end_idx * effective_interval_ns, num=len(adc_chunk))

        fig, ax = plt.subplots()
        ax.plot(times, voltage_chunk_mv, lw=1)
        plt.show()
        

    print("\nFinished processing.")
    print(f"Global voltage range: {global_min_mv:.2f} mV to {global_max_mv:.2f} mV")
