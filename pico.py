from __future__ import annotations

import ctypes
import queue
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger
from picosdk.functions import PICO_STATUS
from picosdk.ps5000a import ps5000a as ps


def check_status(status: int, function_name: str) -> None:
    """Check the status returned by a Picoscope SDK call and raise on error.

    Args:
        status: The status code returned by the SDK function.
        function_name: The name of the function that was called.

    Raises:
        Exception: If the status is not PICO_OK.
    """
    if status != PICO_STATUS["PICO_OK"]:
        # Find the string name of the error code
        error_name = next(
            (k for k, v in PICO_STATUS.items() if v == status), "PICO_UNKNOWN_ERROR"
        )
        raise RuntimeError(
            f"{function_name} failed with status {status} ({error_name})"
        )


class PicoDevice:
    """A class to manage a Picoscope 5000a series device for data streaming.

    This class handles device configuration, buffer management, and the data
    capture loop. It acts as the "producer" in a producer-consumer pattern.

    Data Formats:
        - average mode: Single stream of averaged ADC values
        - aggregate mode: Interleaved stream [min1, max1, min2, max2, ...]
                         where each pair represents one downsampled timestep

    Buffer Management:
        - SDK buffers: Direct hardware interface (bufferA, bufferB for aggregate)
        - Application buffers: Larger buffers for efficient file writing
        - Interleaved buffer: Temporary buffer for aggregate mode processing
    """

    def __init__(
        self,
        handle: int,
        resolution: str,
        pico_buffer_size: int,
        pico_num_buffers: int,
        comp_buffer_size: int,
        data_queue: queue.Queue[int],
        empty_queue: queue.Queue[int],
        data_buffers: List[np.ndarray],
        shutdown_event: threading.Event,
        downsample_mode: str = "average",
    ) -> None:
        """Initializes the PicoDevice and opens a connection to the hardware.

        Args:
            handle: The device handle provided by the SDK.
            resolution: The desired resolution, e.g., "PS5000A_DR_16BIT".
            pico_buffer_size: The size of each buffer allocated within the SDK.
            pico_num_buffers: The number of buffers for the SDK to use.
            comp_buffer_size: The size of the application-side buffers for writing to disk.
            data_queue: Queue to send indices of full buffers to the consumer.
            empty_queue: Queue to receive indices of empty buffers from the consumer.
            data_buffers: A list of pre-allocated numpy arrays for data transfer.
            shutdown_event: A threading.Event to signal shutdown.
        """
        # --- Device and Resolution ---
        self.handle: ctypes.c_int16 = ctypes.c_int16(handle)
        self.resolution: str = resolution
        res_enum = ps.PS5000A_DEVICE_RESOLUTION[resolution]

        # --- Picoscope SDK Buffer Configuration ---
        self.pico_buffer_size: int = pico_buffer_size
        self.pico_num_buffers: int = pico_num_buffers
        self.total_samples: int = self.pico_buffer_size * self.pico_num_buffers

        # --- Application Buffer (for file writing) ---
        self.comp_buffer_size: int = comp_buffer_size

        # --- Internal Data Buffer (receives data from SDK) ---
        self.downsample_mode = downsample_mode
        self.bufferA: np.ndarray = np.zeros(shape=self.pico_buffer_size, dtype=np.int16)
        self.bufferB: Optional[np.ndarray] = None
        self.interleaved_buffer: Optional[np.ndarray] = None
        if self.downsample_mode == "aggregate":
            self.bufferB = np.zeros(shape=self.pico_buffer_size, dtype=np.int16)
            self.interleaved_buffer = np.zeros(
                shape=self.pico_buffer_size * 2, dtype=np.int16
            )

        # --- Ctypes and Callback ---
        self.callbackFuncPtr = ps.StreamingReadyType(self.streaming_callback)
        self.max_adc: ctypes.c_int16 = ctypes.c_int16()

        # --- Channel Configuration ---
        self.channel_range: Optional[int] = None
        self.voltage_range_v: Optional[float] = None
        self.channel_a_coupling: Optional[str] = None
        self.channel_a_range_str: Optional[str] = None

        # --- Threading and Queues ---
        self.shutdown_event: threading.Event = shutdown_event
        self.data_queue: queue.Queue[int] = data_queue
        self.empty_queue: queue.Queue[int] = empty_queue
        self.data_buffers: List[np.ndarray] = data_buffers

        # --- Buffer Management State ---
        self.buf_idx: int = self.empty_queue.get()
        self.buf_used: int = 0
        self.buf_free: int = self.comp_buffer_size

        # --- Streaming Configuration ---
        self.streaming_configured: bool = False
        self.sample_int: Optional[ctypes.c_int32] = None
        self.sample_unit: Optional[int] = None
        self.ratio: Optional[int] = None
        self.pre_trig_samples: Optional[int] = None
        self.down_sample_ratio: Optional[int] = None
        self.auto_stop: Optional[int] = None
        self.auto_stop_stream: Optional[int] = None

        # --- Status and Performance Metrics ---
        self.captured_samples: int = 0
        self.empty_pro_queue_count: int = 0
        self.overflow_count: int = 0
        self.callback_durations: List[float] = []

        # --- Aggregate Mode Performance Tracking ---
        self.interleave_durations: List[float] = []

        # --- Open device connection ---
        status = ps.ps5000aOpenUnit(ctypes.byref(self.handle), None, res_enum)
        check_status(status, "ps5000aOpenUnit")

        status = ps.ps5000aMaximumValue(self.handle, ctypes.byref(self.max_adc))
        check_status(status, "ps5000aMaximumValue")

    def set_channel(
        self, chan: str, en: int, coup: str, voltage_range_str: str, offset: float
    ) -> None:
        """Configure a channel on the Picoscope.

        Args:
            chan: The channel identifier string, e.g., "PS5000A_CHANNEL_A".
            en: Whether the channel is enabled (1) or disabled (0).
            coup: The coupling type string, e.g., "PS5000A_DC".
            voltage_range_str: The voltage range string, e.g., "PS5000A_20V".
            offset: The analog voltage offset in Volts.
        """
        channel_range_enum = ps.PS5000A_RANGE[voltage_range_str]
        self.channel_range = channel_range_enum

        # Store the actual voltage range for metadata and conversion
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
        }
        self.voltage_range_v = range_to_voltage.get(voltage_range_str)

        if chan == "PS5000A_CHANNEL_A" and en:
            self.channel_a_coupling = coup
            self.channel_a_range_str = voltage_range_str

        channel_enum = ps.PS5000A_CHANNEL[chan]
        coupling_enum = ps.PS5000A_COUPLING[coup]
        status = ps.ps5000aSetChannel(
            self.handle, channel_enum, en, coupling_enum, channel_range_enum, offset
        )
        check_status(status, f"ps5000aSetChannel ({chan})")
        logger.debug(
            f"Range '{voltage_range_str}' maps to enum value: {channel_range_enum}, voltage range: {self.voltage_range_v}V"
        )

    def set_data_buffer(self, chan: str, segment: int, rat: str) -> None:
        """Set up the data buffer for a specific channel for streaming.

        Args:
            chan: The channel identifier string, e.g., "PS5000A_CHANNEL_A".
            segment: The memory segment to use (0 for streaming).
            rat: The ratio mode string, e.g., "PS5000A_RATIO_MODE_NONE".
        """
        channel_enum = ps.PS5000A_CHANNEL[chan]
        ratio_enum = ps.PS5000A_RATIO_MODE[rat]

        buffer_min_ptr = None
        if self.bufferB is not None:
            buffer_min_ptr = self.bufferB.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))

        status = ps.ps5000aSetDataBuffers(
            self.handle,
            channel_enum,
            self.bufferA.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            buffer_min_ptr,
            self.pico_buffer_size,
            segment,
            ratio_enum,
        )
        check_status(status, f"ps5000aSetDataBuffers ({chan})")

    def configure_streaming_var(
        self,
        samp_int: int,
        samp_unit: str,
        pre_trig_samp: int,
        down_samp_rat: int,
        rat: str,
        auto_stop: int,
        auto_stop_stream: int,
    ) -> None:
        """Store streaming parameters before starting the capture.

        Args:
            samp_int: The desired sample interval in `samp_unit` units.
            samp_unit: The time unit string, e.g., "PS5000A_NS".
            pre_trig_samp: The number of pre-trigger samples.
            down_samp_rat: The downsampling ratio.
            rat: The ratio mode string, e.g., "PS5000A_RATIO_MODE_NONE".
            auto_stop: Whether to stop the capture automatically (1) or not (0).
            auto_stop_stream: Deprecated, not used.
        """
        self.sample_int = ctypes.c_int32(samp_int)
        self.sample_unit = ps.PS5000A_TIME_UNITS[samp_unit]
        self.ratio = ps.PS5000A_RATIO_MODE[rat]
        self.pre_trig_samples = pre_trig_samp
        self.down_sample_ratio = down_samp_rat
        self.auto_stop = auto_stop
        self.auto_stop_stream = auto_stop_stream

    def get_metadata(self) -> Dict[str, Any]:
        """Return comprehensive acquisition metadata."""
        metadata = {
            "resolution": self.resolution,
            "sample_interval_ns": self.sample_int.value if self.sample_int else None,
            "voltage_range_v": self.voltage_range_v,
            "max_adc": self.max_adc.value,
            "channel_a_coupling": self.channel_a_coupling,
            "channel_a_range": self.channel_a_range_str,
            "downsample_mode": self.downsample_mode,
            "hardware_downsample_ratio": self.down_sample_ratio,
            "data_format_version": "1.0",
            "interleaved_format": self.downsample_mode == "aggregate",
        }

        if self.downsample_mode == "aggregate":
            metadata.update(
                {
                    "aggregate_format": "interleaved_min_max",
                    "aggregate_description": "Data format: [min1, max1, min2, max2, ...]",
                }
            )

        return metadata

    def run_streaming(self) -> None:
        """Starts the Picoscope streaming capture."""
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

    def streaming_callback(
        self,
        _handle: int,
        noOfSamples: int,
        startIndex: int,
        _overflow: int,
        _triggerAt: int,
        _triggered: int,
        _autoStop: int,
        _param: int,
    ) -> None:
        """Callback function executed by the SDK when new streaming data is available.

        This function is the heart of the producer. It copies data from the
        Picoscope's internal buffer into the application's shared buffer pool.
        When an application buffer is full, its index is placed on the data_queue
        for the consumer.

        Note: This function is called from a thread created by the Picoscope SDK.
        It must be fast and thread-safe.
        """
        if _overflow:
            self.overflow_count += 1
            logger.warning(
                "Picoscope hardware buffer overflow detected. Data has been lost."
            )

        # Stop processing if a shutdown is requested.
        if self.shutdown_event.is_set():
            return

        callback_start_time = time.perf_counter()
        if noOfSamples > 0:
            self.captured_samples += noOfSamples

            source_buffer = self.bufferA
            samples_to_process = noOfSamples

            # In aggregate mode, interleave the min/max buffers into one
            if (
                self.downsample_mode == "aggregate"
                and self.bufferB is not None
                and self.interleaved_buffer is not None
            ):
                interleave_start = time.perf_counter()

                # The SDK provides min/max data in separate buffers (B/A).
                # We interleave them into a single [min, max, min, max, ...]
                # stream for the consumer.
                total_interleaved_samples = noOfSamples * 2

                # Use numpy's more efficient interleaving
                np.stack(
                    [
                        self.bufferB[startIndex : startIndex + noOfSamples],
                        self.bufferA[startIndex : startIndex + noOfSamples],
                    ],
                    axis=1,
                    out=self.interleaved_buffer[:total_interleaved_samples].reshape(
                        -1, 2
                    ),
                )

                source_buffer = self.interleaved_buffer
                samples_to_process = total_interleaved_samples
                # After interleaving, the source index is always 0
                source_index = 0

                # Track interleaving performance
                self.interleave_durations.append(
                    (time.perf_counter() - interleave_start) * 1000
                )
            else:
                source_index = startIndex

            # This loop copies data from the source_buffer into
            # our larger, shared application buffers (self.data_buffers).
            while samples_to_process > 0:
                # Determine how much space is left in the current application buffer.
                self.buf_free = self.comp_buffer_size - self.buf_used
                copy_size = min(samples_to_process, self.buf_free)

                # Copy the data slice.
                self.data_buffers[self.buf_idx][
                    self.buf_used : self.buf_used + copy_size
                ] = source_buffer[source_index : source_index + copy_size]

                # Update pointers and remaining sample counts.
                self.buf_used += copy_size
                samples_to_process -= copy_size
                source_index += copy_size

                # If the current application buffer is full...
                if self.buf_used == self.comp_buffer_size:
                    # ...send its index to the consumer.
                    self.data_queue.put(self.buf_idx)
                    try:
                        # ...and get a new empty buffer from the consumer.
                        self.buf_idx = self.empty_queue.get_nowait()
                        self.buf_used = 0
                    except queue.Empty:
                        # This is a critical failure. The consumer is not keeping up.
                        self.empty_pro_queue_count += 1
                        logger.critical(
                            "Producer queue is empty. Consumer cannot keep up. "
                            "Shutting down to prevent data loss."
                        )
                        self.shutdown_event.set()
                        return  # Exit immediately.

            duration_ms = (time.perf_counter() - callback_start_time) * 1000
            self.callback_durations.append(duration_ms)

    def run_capture(self) -> None:
        """The main capture loop for the producer thread."""
        if not self.streaming_configured:
            self.run_streaming()

        # This loop polls the SDK for new data, which triggers the callback.
        while not self.shutdown_event.is_set():
            ps.ps5000aGetStreamingLatestValues(self.handle, self.callbackFuncPtr, None)
            # Yield the GIL to other threads.
            time.sleep(0.001)

        # --- Shutdown and reporting ---
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

        # Report aggregate mode performance if applicable
        if self.downsample_mode == "aggregate" and self.interleave_durations:
            logger.info("--- Aggregate Mode Performance ---")
            logger.info(
                f"Total interleave operations: {len(self.interleave_durations)}"
            )
            logger.info(
                f"Min interleave duration: {min(self.interleave_durations):.3f} ms"
            )
            logger.info(
                f"Max interleave duration: {max(self.interleave_durations):.3f} ms"
            )
            logger.info(
                f"Avg interleave duration: {np.mean(self.interleave_durations):.3f} ms"
            )
            logger.info("----------------------------------")

        self.close_device()

    def close_device(self) -> None:
        """Stops the Picoscope and closes the connection."""
        # Check if handle is valid before trying to close.
        if self.handle.value > 0:
            status_stop = ps.ps5000aStop(self.handle)
            if status_stop != PICO_STATUS["PICO_OK"]:
                logger.warning(f"ps5000aStop failed with status {status_stop}")

            status_close = ps.ps5000aCloseUnit(self.handle)
            if status_close != PICO_STATUS["PICO_OK"]:
                logger.warning(f"ps5000aCloseUnit failed with status {status_close}")

            # Invalidate handle to prevent reuse.
            self.handle.value = 0
            logger.info("Picoscope connection closed.")
