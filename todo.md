
- [ ] max/min decimation?

- [ ] reload hdf5 on exit, e.g. to plot, but also to ensure the data is saved correctly

- [x] user options, e.g.: analog bandwidth (Hz), sampling rate (S/s), data window (s). Where sampling rate can be slower than analog bandwidth (i.e. it's a digital bandwidth), but uses 'high res mode' (~min/max decimation?) like our analysis code? -> just set plot window size...

- [ ] x-axis values incorrect!?


----


fix this:
2025-09-02 17:39:00.772 | WARNING  | dfplot:update_display:357 - Missing calibration data (voltage_range_v or max_adc), using raw ADC values
2025-09-02 17:39:00.918 | WARNING  | dfplot:update_display:357 - Missing calibration data (voltage_range_v or max_adc), using raw ADC values
2025-09-02 17:39:01.184 | WARNING  | dfplot:update_display:357 - Missing calibration data (voltage_range_v or max_adc), using raw ADC values
2025-09-02 17:39:01.776 | WARNING  | dfplot:update_display:357 - Missing calibration data (voltage_range_v or max_adc), using raw ADC values
2025-09-02 17:39:01.910 | WARNING  | dfplot:update_display:357 - Missing calibration data (voltage_range_v or max_adc), using raw ADC values
2025-09-02 17:39:02.040 | WARNING  | dfplot:update_display:357 - Missing calibration data (voltage_range_v or max_adc), using raw ADC values
2025-09-02 17:39:02.955 | WARNING  | dfplot:update_display:357 - Missing calibration data (voltage_range_v or max_adc), using raw ADC values

---


Here is a summary of our present goals and challenges.

Our project is a multi-threaded Python application designed for high-speed data acquisition from a Picoscope, with a
focus on robustness, reliability, stability, and simplicity. The architecture uses a producer-consumer pattern where a
producer thread (pico.py) acquires data from the hardware, and a consumer thread (consumer.py) writes it to an HDF5
file. A separate, decoupled plotter (dfplot.py) provides a live view of the data. The system is orchestrated by
main.py.

The primary goal is to achieve the maximum possible data acquisition rate, specifically targeting 62.5 MS/s (a 16 ns
sample interval), without losing data or compromising system stability. The main challenge we face is a significant
performance bottleneck that prevents us from reaching this target. Our post-acquisition analysis shows that the system
is only achieving an effective rate of around 45 MS/s, which is approximately 70% of the configured rate.

Through a process of elimination and diagnostics, we have determined the root cause of this performance loss. We first
ruled out disk I/O speed and hardware buffer overflows. The key insight came from adding detailed performance timing to
the streaming_callback function in pico.py. The logs revealed that this callback, which is the bridge between the
C-level driver and our Python code, is taking a significant amount of time to execute (e.g., up to 42 ms). Because this
callback is blocking, it stalls the Picoscope driver, which in turn causes the hardware to pause acquisition while it
waits. The "missing" 30% of performance is the accumulated time the hardware spends stalled. The problem is not that
the data copying is slow, but that the overhead of frequently calling the Python callback is too high.

Our immediate next step is to implement one of two simple fixes to reduce the frequency of these callbacks. The first
option is to add a small delay (e.g., 1 ms) to the polling loop in pico.py, allowing the driver's internal buffer to
accumulate more data between polls. The second option is to increase the size of the data buffers in main.py, which
achieves a similar effect by requiring fewer callbacks to transfer the same amount of data.

--> I'm not sure those fixes stated above are the *best* options. We should reconsider at a high level.

main.py pico.py
