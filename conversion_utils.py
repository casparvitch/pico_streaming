import numpy as np
import numba as nb


@nb.jit(nopython=True, fastmath=True)
def adc_to_mV(adc_data, voltage_range_v, max_adc_value):
    """
    Convert ADC counts to voltage in mV using Numba JIT compilation.
    
    Args:
        adc_data: numpy array of int16 ADC values
        voltage_range_v: float, full scale voltage range (e.g., 20.0 for ±20V)
        max_adc_value: int, maximum ADC value (typically 32767 for 16-bit)
    
    Returns:
        numpy array of voltages in mV
    """
    voltage_range_mv = voltage_range_v * 1000.0  # Convert to mV
    return (adc_data.astype(np.float64) * voltage_range_mv) / max_adc_value


@nb.jit(nopython=True, fastmath=True)
def min_max_decimate_numba(data, factor):
    """
    Numba-optimized min-max decimation to preserve transients while reducing data points.
    For each group of 'factor' samples, keep both min and max values.
    """
    if len(data) < factor:
        return data

    # Calculate number of complete groups
    n_complete_groups = len(data) // factor
    if n_complete_groups == 0:
        return data

    # Pre-allocate output array
    decimated = np.empty(n_complete_groups * 2, dtype=data.dtype)
    
    # Process each group
    for i in range(n_complete_groups):
        start_idx = i * factor
        end_idx = start_idx + factor
        
        # Find min and max in this group
        group_min = data[start_idx]
        group_max = data[start_idx]
        
        for j in range(start_idx + 1, end_idx):
            if data[j] < group_min:
                group_min = data[j]
            if data[j] > group_max:
                group_max = data[j]
        
        # Store min and max
        decimated[i * 2] = group_min
        decimated[i * 2 + 1] = group_max
    
    return decimated
