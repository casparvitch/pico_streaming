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
        cmaxSamples,
        timeIntervalns,
        chARange,
        maxADC,
    ):
        self.buffer_size = buffer_size
        self.data_queue = data_queue
        self.empty_queue = empty_queue
        self.data_buffers = data_buffers
        self.file_name = file_name
        self.running = True

        self.values_written = 0

        self.cmaxSamples = cmaxSamples
        self.timeIntervalns = timeIntervalns
        self.chARange = chARange
        self.maxADC = maxADC

        self.empty_con_queue_count = 0

    def stop(self):
        self.running = False

    def consume(self):
        total_save_length = 0
        metadata = {
            "cmaxSamples": self.cmaxSamples,
            "timeIntervalns": self.timeIntervalns,
            "chARange": self.chARange,
            "maxADC": self.maxADC,
        }

        # Force overwrite of existing file
        if os.path.exists(self.file_name):
            os.remove(self.file_name)
            logger.info(f"Removed existing file: {self.file_name}")

        with h5py.File(self.file_name, "w") as f:
            metadata_group = f.create_group("metadata")
            for key, value in metadata.items():
                metadata_group.attrs[key] = value

            dset = f.create_dataset(
                "adc_counts",
                (0,),
                maxshape=(None,),
                dtype="int16",
                chunks=(self.buffer_size,),
            )

            while self.running:
                if not self.running:
                    break
                try:
                    idx = self.data_queue.get(timeout=0.1)

                    dset.resize((self.values_written + (len(self.data_buffers[idx])),))
                    dset[self.values_written :] = self.data_buffers[idx]
                    self.empty_queue.put(idx)

                    self.values_written += len(self.data_buffers[idx])

                except queue.Empty:
                    self.empty_con_queue_count += 1
                    # This is expected when acquisition stops, so no need to log as a warning
                    if self.running:
                        logger.debug("Consumer queue was empty.")

        logger.info(
            f"Consumer couldn't obtain data from queue {self.empty_con_queue_count} times."
        )
