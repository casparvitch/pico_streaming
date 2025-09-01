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
    ):
        ####### Setup variables #######
        self.handle = ctypes.c_int16(handle)
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
        self.running = True
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

        ####### Open device conneciton #######
        status = ps.ps5000aOpenUnit(ctypes.byref(self.handle), None, res)
        check_status(status, "ps5000aOpenUnit")

        status = ps.ps5000aMaximumValue(self.handle, ctypes.byref(self.max_adc))
        check_status(status, "ps5000aMaximumValue")

    def stop(self):
        self.running = False

    def set_channel(self, chan, en, coup, range, offset):
        channel_range = ps.PS5000A_RANGE[range]
        self.channel_range = channel_range
        channel = ps.PS5000A_CHANNEL[chan]
        coupling = ps.PS5000A_COUPLING[coup]
        status = ps.ps5000aSetChannel(
            self.handle, channel, en, coupling, channel_range, offset
        )
        check_status(status, f"ps5000aSetChannel ({chan})")
        logger.debug(f"Set channel {chan}: status {status}")

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
            "cmaxSamples": self.total_samples,
            "timeIntervalns": self.sample_int.value,
            "chARange": self.channel_range,
            "maxADC": self.max_adc.value,
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
        logger.debug(f"Run streaming: status {status}")

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

        if self.running:
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
                            logger.error(
                                "Producer queue is empty. Data will be dropped until a buffer is available."
                            )
                            # Break the inner loop; we can't process more data without a buffer.
                            break

    def run_capture(self):
        self.run_streaming()
        while self.running:
            ps.ps5000aGetStreamingLatestValues(self.handle, self.callbackFuncPtr, None)
            # Give the CPU a break, crucial for preventing a busy-wait loop
            time.sleep(0.01)

        logger.info(
            f"Producer couldn't obtain an empty queue {self.empty_pro_queue_count} times."
        )
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
