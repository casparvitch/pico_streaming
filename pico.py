import ctypes
import time
import matplotlib.pyplot as plot
import numpy as np
import queue
from loguru import logger

from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import adc2mV, PICO_STATUS


def check_status(status, function_name):
    """Checks the status returned by the PicoSDK and raises an exception if it's not PICO_OK."""
    if status != PICO_STATUS["PICO_OK"]:
        error_name = next(
            (k for k, v in PICO_STATUS.items() if v == status), "PICO_UNKNOWN_ERROR"
        )
        raise Exception(f"{function_name} failed with status {status} ({error_name})")


class PicoDevice:
    def __init__(
        self,
        handle,
        resolution,
        pico_buffer_size,
        pico_num_buffers,
        comp_buffer_size,
        data_queue,
        empty_queue,
        data_buffers,
        shutdown_event,
    ):
        ####### Setup variables #######
        self.handle = ctypes.c_int16(handle)
        self.resolution = resolution
        res = ps.PS5000A_DEVICE_RESOLUTION[resolution]

        ####### Local picoscope buffer variables #######
        self.pico_buffer_size = pico_buffer_size
        self.pico_num_buffers = pico_num_buffers
        self.total_samples = self.pico_buffer_size * self.pico_num_buffers

        ####### File writing buffer variables #######
        self.comp_buffer_size = comp_buffer_size

        ####### Temporary writing buffer variables #######
        self.bufferA = np.zeros(shape=self.pico_buffer_size, dtype=np.int16)

        ####### Misc variables #######
        self.callbackFuncPtr = ps.StreamingReadyType(self.streaming_callback)
        self.channel_range = None
        self.max_adc = ctypes.c_int16()
        self.shutdown_event = shutdown_event

        self.channel_a_coupling = None
        self.channel_a_range_str = None
        ####### Callback Function Variables #######

        self.nextSample = 0
        self.autoStopStream = False

        self.data_queue = data_queue
        self.empty_queue = empty_queue
        self.data_buffers = data_buffers

        self.buf_idx = self.empty_queue.get()
        self.buf_used = 0
        self.buf_free = self.comp_buffer_size

        ####### Streaming Variables #######
        self.streaming_configured = False
        self.sample_int = None
        self.sample_unit = None
        self.ratio = None
        self.pre_trig_samples = None
        self.down_sample_ratio = None
        self.auto_stop = None
        self.auto_stop_stream = None

        ####### Status information #######
        self.status = {}
        self.captured_samples = 0
        self.max_sample = 0
        self.max_sample_point = 0
        self.max_sample_count = 0
        self.empty_pro_queue_count = 0
        self.overflow_count = 0
        self.callback_durations = []

        ####### Open device conneciton #######
        status = ps.ps5000aOpenUnit(ctypes.byref(self.handle), None, res)
        check_status(status, "ps5000aOpenUnit")

        status = ps.ps5000aMaximumValue(self.handle, ctypes.byref(self.max_adc))
        check_status(status, "ps5000aMaximumValue")

    def set_channel(self, chan, en, coup, range, offset):
        channel_range = ps.PS5000A_RANGE[range]
        self.channel_range = channel_range

        # Store the actual voltage range for conversion
        range_to_voltage = {
            "PS5000A_10MV": 0.01,
            "PS5000A_20MV": 0.02,
            "PS5000A_50MV": 0.05,
            "PS5000A_100MV": 0.1,
            "PS5000A_200MV": 0.2,
            "PS5000A_500MV": 0.5,
            "PS5000A_1V": 1.0,
            "PS5000A_2V": 2.0,
            "PS5000A_5V": 5.0,
            "PS5000A_10V": 10.0,
            "PS5000A_20V": 20.0,
            "PS5000A_50V": 50.0,
            "PS5000A_100V": 100.0,
            "PS5000A_200V": 200.0,
        }
        self.voltage_range_v = range_to_voltage.get(range, 20.0)  # Default to 20V

        if chan == "PS5000A_CHANNEL_A" and en:
            self.channel_a_coupling = coup
            self.channel_a_range_str = range

        channel = ps.PS5000A_CHANNEL[chan]
        coupling = ps.PS5000A_COUPLING[coup]
        status = ps.ps5000aSetChannel(
            self.handle, channel, en, coupling, channel_range, offset
        )
        check_status(status, f"ps5000aSetChannel ({chan})")
        logger.debug(f"Set channel {chan}: status {status}")
        logger.debug(
            f"Range '{range}' maps to enum value: {ps.PS5000A_RANGE[range]}, voltage range: {self.voltage_range_v}V"
        )

    def set_data_buffer(self, chan, segment, rat):
        channel = ps.PS5000A_CHANNEL[chan]
        ratio = ps.PS5000A_RATIO_MODE[rat]
        status = ps.ps5000aSetDataBuffers(
            self.handle,
            channel,
            self.bufferA.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            None,
            self.pico_buffer_size,
            segment,
            ratio,
        )
        check_status(status, f"ps5000aSetDataBuffers ({chan})")
        logger.debug(f"Set data buffer for {chan}: status {status}")

    def configure_streaming_var(
        self,
        samp_int,
        samp_unit,
        pre_trig_samp,
        down_samp_rat,
        rat,
        auto_stop,
        auto_stop_stream,
    ):
        self.sample_int = ctypes.c_int32(samp_int)
        self.sample_unit = ps.PS5000A_TIME_UNITS[samp_unit]
        self.ratio = ps.PS5000A_RATIO_MODE[rat]
        self.pre_trig_samples = pre_trig_samp
        self.down_sample_ratio = down_samp_rat
        self.auto_stop = auto_stop
        self.auto_stop_stream = auto_stop_stream

    def get_metadata(self):
        """Returns a dictionary of acquisition parameters for the HDF5 file."""
        return {
            "resolution": self.resolution,
            "sample_interval_ns": self.sample_int.value,
            "voltage_range_v": self.voltage_range_v,
            "max_adc": self.max_adc.value,
            "channel_a_coupling": self.channel_a_coupling,
            "channel_a_range": self.channel_a_range_str,
        }

    def run_streaming(self):
        status = ps.ps5000aRunStreaming(
            self.handle,
            ctypes.byref(self.sample_int),
            self.sample_unit,
            self.pre_trig_samples,
            self.total_samples,
            self.auto_stop,
            self.down_sample_ratio,
            self.ratio,
            self.pico_buffer_size,
        )
        check_status(status, "ps5000aRunStreaming")
        self.streaming_configured = True
        logger.info(
            f"Streaming configured. Actual sample interval: {self.sample_int.value} ns"
        )

    # this function is called each time data is avaible from the picoscope, from here the data in the buffer should be accessed
    def streaming_callback(
        self,
        _handle,
        noOfSamples,
        startIndex,
        _overflow,
        _triggerAt,
        _triggered,
        _autoStop,
        _param,
    ):
        if _overflow:
            self.overflow_count += 1
            logger.warning(
                "Picoscope hardware buffer overflow detected. Data has been lost."
            )

        if not self.shutdown_event.is_set():
            callback_start_time = time.perf_counter()
            if noOfSamples > 0:
                self.captured_samples += noOfSamples
                len_data = noOfSamples
                src_idx = startIndex

                while len_data > 0:
                    self.buf_free = self.comp_buffer_size - self.buf_used
                    copy_size = min(len_data, self.buf_free)

                    self.data_buffers[self.buf_idx][
                        self.buf_used : self.buf_used + copy_size
                    ] = self.bufferA[src_idx : src_idx + copy_size]

                    self.buf_used += copy_size
                    len_data -= copy_size
                    src_idx += copy_size

                    if self.buf_used == self.comp_buffer_size:
                        self.data_queue.put(self.buf_idx)
                        try:
                            self.buf_idx = self.empty_queue.get_nowait()
                            self.buf_used = 0
                        except queue.Empty:
                            self.empty_pro_queue_count += 1
                            logger.critical(
                                "Producer queue is empty. Consumer cannot keep up. Shutting down to prevent data loss."
                            )
                            self.shutdown_event.set()
                            # Break the inner loop; we can't process more data without a buffer.
                            break
                duration_ms = (time.perf_counter() - callback_start_time) * 1000
                self.callback_durations.append(duration_ms)

    def run_capture(self):
        if not self.streaming_configured:
            self.run_streaming()
        while not self.shutdown_event.is_set():
            ps.ps5000aGetStreamingLatestValues(self.handle, self.callbackFuncPtr, None)
            # Yield the CPU to other threads without a long pause
            time.sleep(0.0)

        logger.info(
            f"Producer couldn't obtain an empty queue {self.empty_pro_queue_count} times."
        )
        logger.info(f"Picoscope hardware overflowed {self.overflow_count} times.")
        if self.callback_durations:
            logger.info("--- Callback Performance ---")
            logger.info(f"Total callbacks: {len(self.callback_durations)}")
            logger.info(f"Min duration: {min(self.callback_durations):.2f} ms")
            logger.info(f"Max duration: {max(self.callback_durations):.2f} ms")
            logger.info(f"Avg duration: {np.mean(self.callback_durations):.2f} ms")
            logger.info("--------------------------")
        self.close_device()

    def close_device(self):
        # Check if handle is valid before trying to close
        if self.handle.value > 0:
            status_stop = ps.ps5000aStop(self.handle)
            logger.debug(f"Device stop status: {status_stop}")
            status_close = ps.ps5000aCloseUnit(self.handle)
            logger.debug(f"Device close status: {status_close}")
            # Invalidate handle
            self.handle.value = 0
