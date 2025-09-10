import numpy as np
import matplotlib.pyplot as plt
from picostream.reader import PicoStreamReader
from transivent_analysis import (
    configure_logging,
    calculate_smoothing_parameters,
    initialize_state,
    process_chunk,
    get_final_events,
    calculate_initial_background,
)

# --- Analysis Configuration ---
CONFIG = {
    "SMOOTH_WIN_T": 10e-3,
    "SMOOTH_WIN_F": None,
    "DETECTION_SNR": 3,
    "MIN_EVENT_KEEP_SNR": 5,
    "MIN_EVENT_T": 0.75e-6,
    "WIDEN_FRAC": 10,
    "SIGNAL_POLARITY": 1,  # +1 for positive events (current blockades are positive)
    "LOG_LEVEL": "INFO",
    "FILTER_TYPE": "gaussian",
    "FILTER_ORDER": 2,
}

HDF5_PATH = "../data/20250905-haem-picostream/60nM.hdf5"
CHUNK_SIZE = 50_000_000


def analyze_event_properties(reader, events, config, smooth_n):
    """
    Analyzes properties of detected events (duration, amplitude) by re-reading
    data segments from the HDF5 file.
    """
    event_durations_us = []
    event_amplitudes_mv = []
    sampling_interval = reader.sample_interval_ns * 1e-9

    print(f"Analyzing properties of {len(events)} events...")

    for i, (t_start, t_end) in enumerate(events):
        # Duration is the time difference
        duration_us = (t_end - t_start) * 1e6
        event_durations_us.append(duration_us)

        # --- Amplitude Calculation ---
        # To find the amplitude, we need a local baseline. We read a small
        # window of data around the event to calculate this.
        padding = (t_end - t_start) * 5  # Read a window ~11x event duration
        read_start_t = max(0, t_start - padding)
        read_end_t = t_end + padding

        # Convert time back to sample index to use reader.get_block()
        start_idx = int(read_start_t / sampling_interval)
        end_idx = int(read_end_t / sampling_interval)
        size = end_idx - start_idx

        if size <= 0:
            continue

        # Read the local data segment
        t_local, v_local = reader.get_block(size=size, start=start_idx)
        if t_local is None or t_local.size < smooth_n:
            continue

        # Calculate a local background for this segment
        bg_local = calculate_initial_background(
            t_local, v_local, smooth_n, config["FILTER_TYPE"]
        )

        # Find the peak amplitude relative to the local background
        event_mask_local = (t_local >= t_start) & (t_local < t_end)
        if np.any(event_mask_local):
            signal_minus_bg = v_local[event_mask_local] - bg_local[event_mask_local]
            if config["SIGNAL_POLARITY"] < 0:
                amp = np.min(signal_minus_bg)
            else:
                amp = np.max(signal_minus_bg)
            event_amplitudes_mv.append(abs(amp))
        else:
            # This can happen if event is right at the edge of a read block
            event_amplitudes_mv.append(0)

    return np.array(event_durations_us), np.array(event_amplitudes_mv)


def plot_event_waveforms(reader, events, config, max_events=16):
    """
    Plots a grid of individual event waveforms.
    """
    if len(events) == 0 or len(events) > max_events:
        print(
            f"Skipping event waveform plotting ({len(events)} events found, max is {max_events})."
        )
        return

    print(f"Plotting {len(events)} individual event waveforms...")
    sampling_interval = reader.sample_interval_ns * 1e-9
    n_events = len(events)
    cols = 4
    rows = int(np.ceil(n_events / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(3 * cols, 2.5 * rows), sharex=True, sharey=True
    )
    axes = axes.flatten()

    for i, (t_start, t_end) in enumerate(events):
        # Read a window around the event
        duration = t_end - t_start
        padding = duration * 2
        read_start_t = max(0, t_start - padding)
        read_end_t = t_end + padding

        start_idx = int(read_start_t / sampling_interval)
        end_idx = int(read_end_t / sampling_interval)
        size = end_idx - start_idx

        if size <= 0:
            continue

        t_local, v_local = reader.get_block(size=size, start=start_idx)
        if t_local is None or t_local.size == 0:
            continue

        # Plot data and highlight event region
        ax = axes[i]
        ax.plot((t_local - t_start) * 1e6, v_local, lw=1)
        ax.axvspan(0, (t_end - t_start) * 1e6, color="red", alpha=0.2)
        ax.set_title(f"Event {i+1}", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)

    # Add labels to outer plots
    for i in range(rows * cols):
        if i >= n_events:
            axes[i].axis("off")
        if i // cols == rows - 1:
            axes[i].set_xlabel("Time (µs)")
        if i % cols == 0:
            axes[i].set_ylabel("Voltage (mV)")

    fig.suptitle("Detected Event Waveforms")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def main():
    """
    Main function to process a picostream HDF5 file for transient events.
    """
    configure_logging(CONFIG.get("LOG_LEVEL", "INFO"))

    with PicoStreamReader(HDF5_PATH) as reader:
        print(f"File: {HDF5_PATH}")
        print(f"Contains {reader.num_samples:,} samples.")

        # --- Initialize transivent processor ---
        sampling_interval = reader.sample_interval_ns * 1e-9
        CONFIG["sampling_interval"] = sampling_interval

        smooth_n, min_event_n = calculate_smoothing_parameters(
            sampling_interval,
            CONFIG["SMOOTH_WIN_T"],
            CONFIG["SMOOTH_WIN_F"],
            CONFIG["MIN_EVENT_T"],
            CONFIG["DETECTION_SNR"],
            CONFIG["MIN_EVENT_KEEP_SNR"],
            CONFIG["WIDEN_FRAC"],
            CONFIG["SIGNAL_POLARITY"],
        )
        CONFIG["smooth_n"] = smooth_n
        CONFIG["min_event_n"] = min_event_n

        state = initialize_state(CONFIG)

        # --- Process file in chunks ---
        print(f"Processing in chunks of {CHUNK_SIZE:,} samples...")
        for t_chunk, x_chunk in reader.get_block_iter(chunk_size=CHUNK_SIZE):
            results = process_chunk((t_chunk, x_chunk), state)
            state = results["state"]

        # Finalize event list
        final_events = get_final_events(state)
        print(f"\n--- Processing Complete: Found {len(final_events)} events ---")

        if len(final_events) == 0:
            return

        # --- Analyze Event Properties ---
        inter_event_times_ms = np.diff(final_events[:, 0]) * 1000  # In ms
        event_durations_us, event_amplitudes_mv = analyze_event_properties(
            reader, final_events, CONFIG, smooth_n
        )

        total_time_s = reader.num_samples * sampling_interval
        event_rate_hz = len(final_events) / total_time_s
        event_rate_per_min = event_rate_hz * 60

        # --- Print Summary Statistics ---
        print("\n--- Event Statistics ---")
        print(f"Event Rate: {event_rate_hz:.3f} Hz ({event_rate_per_min:.2f} events/min)")
        if len(inter_event_times_ms) > 0:
            print(
                f"Inter-event Time (ms): mean={np.mean(inter_event_times_ms):.2f}, std={np.std(inter_event_times_ms):.2f}"
            )
        if len(event_durations_us) > 0:
            print(
                f"Event Duration (µs):   mean={np.mean(event_durations_us):.2f}, std={np.std(event_durations_us):.2f}"
            )
        if len(event_amplitudes_mv) > 0:
            print(
                f"Event Amplitude (mV):  mean={np.mean(event_amplitudes_mv):.3f}, std={np.std(event_amplitudes_mv):.3f}"
            )
        print("------------------------\n")

        # --- Plotting ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Event Analysis for {HDF5_PATH.split('/')[-1]}", fontsize=16)

        # Inter-event time histogram
        if len(inter_event_times_ms) > 0:
            axes[0].hist(inter_event_times_ms, bins=50, color="skyblue", edgecolor="black")
            axes[0].set_title("Inter-Event Time Distribution")
            axes[0].set_xlabel("Time between events (ms)")
            axes[0].set_ylabel("Frequency")
            axes[0].grid(True, linestyle=":")

        # Event duration histogram
        if len(event_durations_us) > 0:
            axes[1].hist(event_durations_us, bins=50, color="salmon", edgecolor="black")
            axes[1].set_title("Event Duration Distribution")
            axes[1].set_xlabel("Event duration (µs)")
            axes[1].grid(True, linestyle=":")

        # Event amplitude histogram
        if len(event_amplitudes_mv) > 0:
            axes[2].hist(event_amplitudes_mv, bins=50, color="lightgreen", edgecolor="black")
            axes[2].set_title("Event Amplitude Distribution")
            axes[2].set_xlabel("Event amplitude (mV)")
            axes[2].grid(True, linestyle=":")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # Plot individual event waveforms (if not too many)
        plot_event_waveforms(reader, final_events, CONFIG)


if __name__ == "__main__":
    main()
