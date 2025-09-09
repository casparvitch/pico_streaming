from __future__ import annotations

from typing import Generator, Tuple

import h5py
import numpy as np

from .conversion_utils import adc_to_mV


class PicoStreamReader:
    """A helper class to read and interpret data from picostream HDF5 files.

    This class provides a simple context-manager interface to open HDF5 files,
    read metadata, and retrieve blocks of data as time and voltage arrays.

    It correctly handles the `analog_offset_v` if it is present in the file's
    metadata.

    Example:
        >>> with PicoStreamReader("output.hdf5") as reader:
        ...     print(f"Sample rate: {1e9 / reader.sample_interval_ns:.2f} S/s")
        ...     # Iterate through the whole file
        ...     for times, voltages in reader.get_block_iter(chunk_size=1_000_000):
        ...         # process data
        ...         pass
        ...
        ...     # Or get blocks sequentially
        ...     reader.reset() # Reset internal counter
        ...     while True:
        ...         block = reader.get_next_block(chunk_size=500_000)
        ...         if block is None:
        ...             break
        ...         times, voltages = block
        ...         # process data
    """

    def __init__(self, hdf5_path: str):
        """Initializes the PicoStreamReader.

        Args:
            hdf5_path: Path to the HDF5 file.
        """
        self.hdf5_path = hdf5_path
        self._file: h5py.File | None = None
        self.dset: h5py.Dataset | None = None
        self._current_pos: int = 0

        # Metadata attributes, populated in __enter__
        self.sample_interval_ns: float = 0.0
        self.voltage_range_v: float = 0.0
        self.max_adc_val: int = 0
        self.downsample_mode: str = "average"
        self.hardware_downsample_ratio: int = 1
        self.analog_offset_v: float = 0.0

    def __enter__(self) -> PicoStreamReader:
        """Opens the HDF5 file and reads metadata."""
        self._file = h5py.File(self.hdf5_path, "r")
        self.dset = self._file["adc_counts"]

        # Read metadata from file attributes
        attrs = self._file.attrs
        base_sample_interval_ns = attrs["sample_interval_ns"]
        self.hardware_downsample_ratio = attrs.get("hardware_downsample_ratio", 1)
        self.sample_interval_ns = (
            base_sample_interval_ns * self.hardware_downsample_ratio
        )
        self.voltage_range_v = attrs["voltage_range_v"]
        if "max_adc" in attrs:
            self.max_adc_val = attrs["max_adc"]
        elif "resolution" in attrs:
            # Fallback for older files: calculate max_adc from resolution string
            res_str = attrs["resolution"]  # e.g., "PS5000A_DR_16BIT"
            try:
                # Extract bit depth (e.g., 16) from the string
                res_int = int(res_str.split("_")[-1].replace("BIT", ""))
                self.max_adc_val = (2 ** (res_int - 1)) - 1
            except (ValueError, IndexError):
                raise KeyError(
                    f"Could not parse 'resolution' attribute to determine max_adc: {res_str}"
                )
        else:
            raise KeyError(
                "HDF5 file is missing required 'max_adc' or 'resolution' attribute."
            )
        self.downsample_mode = attrs.get("downsample_mode", "average")
        self.analog_offset_v = attrs.get("analog_offset_v", 0.0)

        self.reset()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Closes the HDF5 file."""
        if self._file:
            self._file.close()

    @property
    def num_samples(self) -> int:
        """Total number of samples in the dataset."""
        if self.dset:
            return self.dset.shape[0]
        return 0

    def reset(self) -> None:
        """Resets the internal position counter for get_next_block."""
        self._current_pos = 0

    def get_next_block(
        self, chunk_size: int
    ) -> Tuple[np.ndarray, np.ndarray] | None:
        """Retrieves the next block of data.

        Args:
            chunk_size: The maximum number of samples to retrieve.

        Returns:
            A (times, voltages) tuple, or None if no more data is available.
        """
        if self._current_pos >= self.num_samples:
            return None

        size = min(chunk_size, self.num_samples - self._current_pos)
        block = self.get_block(size=size, start=self._current_pos)
        self._current_pos += size
        return block

    def get_block_iter(
        self, chunk_size: int = 1_000_000
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Yields data blocks as (times, voltages) tuples for the entire dataset.

        Args:
            chunk_size: The size of each chunk to yield.
        """
        num_samples = self.num_samples
        for start_idx in range(0, num_samples, chunk_size):
            size = min(chunk_size, num_samples - start_idx)
            yield self.get_block(size=size, start=start_idx)

    def get_block(
        self, size: int, start: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieves a specific block of data and converts it to time and voltage.

        Args:
            size: The number of samples to retrieve.
            start: The starting sample index.

        Returns:
            A tuple containing:
                - times (np.ndarray): The time axis in seconds.
                - voltages (np.ndarray): The voltage data in millivolts.
        """
        if not self.dset:
            raise RuntimeError("File not open. Use this class as a context manager.")

        adc_data = self.dset[start : start + size]

        # Convert ADC counts to millivolts
        voltages_mv = adc_to_mV(adc_data, self.voltage_range_v, self.max_adc_val)
        if self.analog_offset_v != 0.0:
            voltages_mv += self.analog_offset_v * 1000

        # Create time axis
        points_per_timestep = 2 if self.downsample_mode == "aggregate" else 1
        time_per_timestep = self.sample_interval_ns * 1e-9

        start_time = (start / points_per_timestep) * time_per_timestep

        num_read_samples = len(adc_data)

        if self.downsample_mode == "aggregate":
            # Data is min/max pairs. Create vertical lines.
            num_pairs = num_read_samples // 2
            time_step_between_pairs = time_per_timestep
            pair_times = start_time + np.arange(num_pairs) * time_step_between_pairs
            times = np.repeat(pair_times, 2)
        else:
            # For linear (non-aggregate) data, create a simple time axis
            time_per_point = time_per_timestep / points_per_timestep
            duration = (
                (num_read_samples - 1) * time_per_point if num_read_samples > 1 else 0
            )
            end_time = start_time + duration
            times = np.linspace(start_time, end_time, num_read_samples)

        return times, voltages_mv
