from __future__ import annotations

import os
import queue
import threading
from typing import Any, Dict, List

import h5py
import numpy as np
from loguru import logger


class Consumer:
    """A data consumer that runs in a separate thread.

    This class retrieves data buffers from a queue, writes them to an HDF5 file,
    and then returns the buffer index to an "empty" queue for reuse by the
    producer. It handles file creation, data writing, and metadata storage.
    """

    def __init__(
        self,
        buffer_size: int,
        data_queue: queue.Queue[int],
        empty_queue: queue.Queue[int],
        data_buffers: List[np.ndarray],
        file_name: str,
        shutdown_event: threading.Event,
        metadata: Dict[str, Any],
    ):
        """Initializes the Consumer.

        Args:
            buffer_size: The size of each individual data buffer.
            data_queue: A queue for receiving indices of data-filled buffers.
            empty_queue: A queue for returning indices of processed (empty) buffers.
            data_buffers: A list of pre-allocated NumPy arrays for data.
            file_name: The path to the output HDF5 file.
            shutdown_event: A threading.Event to signal termination.
            metadata: A dictionary of metadata to be saved as HDF5 attributes.
        """
        self.buffer_size = buffer_size
        self.data_queue = data_queue
        self.empty_queue = empty_queue
        self.data_buffers = data_buffers
        self.file_name = file_name
        self.shutdown_event = shutdown_event
        self.metadata = metadata

        self.values_written: int = 0
        self.empty_con_queue_count: int = 0

    def format_sample_count(self, count: int) -> str:
        """Format a large integer count into a human-readable string.

        Uses metric prefixes (K, M, G) for thousands, millions, and billions.

        Args:
            count: The integer number to format.

        Returns:
            A formatted string representation of the count.
        """
        if count >= 1_000_000_000:
            return f"{count / 1_000_000_000:.2f}G"
        if count >= 1_000_000:
            return f"{count / 1_000_000:.2f}M"
        if count >= 1_000:
            return f"{count / 1_000:.2f}K"
        else:
            return str(count)

    def _processing_loop(self, dset: h5py.Dataset) -> None:
        """
        Continuously processes data from the queue and writes to the HDF5 dataset.

        Args:
            dset: The HDF5 dataset to write to.
        """
        while not self.shutdown_event.is_set():
            try:
                # Wait for a buffer index from the producer.
                # A timeout allows the loop to periodically check the shutdown event.
                idx = self.data_queue.get(timeout=0.1)

                # Append the new data to the HDF5 dataset.
                buffer_len = len(self.data_buffers[idx])
                dset.resize((self.values_written + buffer_len,))
                dset[self.values_written :] = self.data_buffers[idx]

                # Return the buffer index to the empty queue for reuse.
                self.empty_queue.put(idx)

                self.values_written += buffer_len

            except queue.Empty:
                # This occurs if the producer hasn't provided data within the timeout.
                self.empty_con_queue_count += 1
                # This is expected when acquisition stops, so no need to log as a warning
                if not self.shutdown_event.is_set():
                    logger.debug("Consumer queue was empty.")

    def consume(self) -> None:
        """The main loop for the consumer thread.

        This method continuously checks for data from the producer, writes it to
        the HDF5 file, and returns the buffer for reuse. It handles file setup,
        the main processing loop, and graceful shutdown.
        """
        try:
            # Ensure a clean slate by removing any pre-existing file.
            if os.path.exists(self.file_name):
                os.remove(self.file_name)
                logger.info(f"Removed existing file: {self.file_name}")

            with h5py.File(self.file_name, "w") as f:
                # Write the collected metadata to the HDF5 file's attributes.
                for key, value in self.metadata.items():
                    if value is not None:
                        f.attrs[key] = value

                # Create a resizable dataset for the ADC data.
                # Chunking is aligned with the buffer size for efficient writes.
                dset = f.create_dataset(
                    "adc_counts",
                    (0,),
                    maxshape=(None,),
                    dtype="int16",
                    chunks=(self.buffer_size,),
                )

                self._processing_loop(dset)
        except (IOError, OSError) as e:
            # A critical file error means we cannot continue.
            logger.critical(f"Failed to create or write to HDF5 file: {e}")
            self.shutdown_event.set()  # Signal other threads to shut down.
            return

        logger.info(
            f"Consumer couldn't obtain data from queue {self.empty_con_queue_count} times."
        )
