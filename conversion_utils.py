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
