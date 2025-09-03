from __future__ import annotations

import numba as nb
import numpy as np


@nb.jit(nopython=True, fastmath=True)
def adc_to_mV(
    adc_data: np.ndarray, voltage_range_v: float, max_adc_value: int
) -> np.ndarray:
    """Convert ADC counts to voltage in millivolts (mV).

    This function is JIT-compiled with Numba for high performance.

    Args:
        adc_data: A NumPy array of raw ADC integer values.
        voltage_range_v: The single-sided voltage range in Volts. For a device
            configured to ±1V, this value should be 1.0.
        max_adc_value: The maximum possible integer value from the ADC. For a
            16-bit ADC, this is typically 32767.

    Returns:
        A NumPy array of the same shape as `adc_data`, with values converted
        to millivolts.
    """
    voltage_range_mv = voltage_range_v * 1000.0  # Convert Volts to milliVolts
    # Scale ADC counts to millivolts. Cast to float64 for precision.
    return (adc_data.astype(np.float64) * voltage_range_mv) / max_adc_value


@nb.jit(nopython=True, fastmath=True)
def min_max_decimate_numba(data: np.ndarray, factor: int) -> np.ndarray:
    """Downsample data using a min-max technique to preserve signal envelope.

    This Numba-optimized function decimates an array by a given factor. For
    each block of `factor` samples, it finds the minimum and maximum values.
    The output array contains these min/max pairs, effectively preserving
    transients and the signal envelope while reducing the number of data points.

    Note: Any remaining data points that do not form a complete block of size
    `factor` at the end of the array are discarded.

    Args:
        data: The 1D NumPy array of numerical data to be decimated.
        factor: The decimation factor (an integer).

    Returns:
        A new NumPy array containing the decimated data, with a length of
        `2 * (len(data) // factor)`. The format is [min1, max1, min2, max2, ...].
        If the input data has fewer points than `factor`, the original data
        array is returned unmodified.
    """
    # If the data is too short to be decimated, return it as is.
    if len(data) < factor:
        return data

    # Calculate how many full blocks of size `factor` we can process.
    n_complete_groups = len(data) // factor
    if n_complete_groups == 0:
        return data

    # Pre-allocate the output array. It will hold 2 values (min and max)
    # for each group.
    decimated = np.empty(n_complete_groups * 2, dtype=data.dtype)

    # Iterate over each block of data.
    for i in range(n_complete_groups):
        start_idx = i * factor
        end_idx = start_idx + factor

        # Find the minimum and maximum value in the current block.
        # Initialize with the first value in the group.
        group_min = data[start_idx]
        group_max = data[start_idx]

        for j in range(start_idx + 1, end_idx):
            if data[j] < group_min:
                group_min = data[j]
            if data[j] > group_max:
                group_max = data[j]

        # Store the min and max pair in the output array.
        decimated[i * 2] = group_min
        decimated[i * 2 + 1] = group_max

    return decimated
