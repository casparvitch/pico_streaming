import queue
import h5py
import time
import numpy as np
import os
from loguru import logger


class Consumer:
    def __init__(
        self,
        buffer_size,
        data_queue,
        empty_queue,
        data_buffers,
        file_name,
        shutdown_event,
        metadata,
    ):
        self.buffer_size = buffer_size
        self.data_queue = data_queue
        self.empty_queue = empty_queue
        self.data_buffers = data_buffers
        self.file_name = file_name
        self.shutdown_event = shutdown_event
        self.metadata = metadata

        self.values_written = 0
        self.empty_con_queue_count = 0

    def format_sample_count(self, count):
        """Format large sample counts with appropriate units"""
        if count >= 1_000_000_000:
            return f"{count / 1_000_000_000:.2f}G"
        elif count >= 1_000_000:
            return f"{count / 1_000_000:.2f}M"
        elif count >= 1_000:
            return f"{count / 1_000:.2f}K"
        else:
            return str(count)

    def consume(self):
        total_save_length = 0
        try:
            # Force overwrite of existing file
            if os.path.exists(self.file_name):
                os.remove(self.file_name)
                logger.info(f"Removed existing file: {self.file_name}")

            with h5py.File(self.file_name, "w") as f:
                # Store metadata as root-level attributes
                for key, value in self.metadata.items():
                    if value is not None:
                        f.attrs[key] = value

                dset = f.create_dataset(
                    "adc_counts",
                    (0,),
                    maxshape=(None,),
                    dtype="int16",
                    chunks=(self.buffer_size,),
                )

                while not self.shutdown_event.is_set():
                    try:
                        idx = self.data_queue.get(timeout=0.1)

                        dset.resize(
                            (self.values_written + (len(self.data_buffers[idx])),)
                        )
                        dset[self.values_written :] = self.data_buffers[idx]
                        self.empty_queue.put(idx)

                        self.values_written += len(self.data_buffers[idx])

                    except queue.Empty:
                        self.empty_con_queue_count += 1
                        # This is expected when acquisition stops, so no need to log as a warning
                        if not self.shutdown_event.is_set():
                            logger.debug("Consumer queue was empty.")
        except (IOError, OSError) as e:
            logger.critical(f"Failed to create or write to HDF5 file: {e}")
            self.shutdown_event.set()
            return

        logger.info(
            f"Consumer couldn't obtain data from queue {self.empty_con_queue_count} times."
        )
