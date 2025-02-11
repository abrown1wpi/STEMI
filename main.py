import numpy as np
import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes

def collect_data():
    params = BrainFlowInputParams()
    params.serial_port = "COM3"
    sampling_rate = 125
    duration = 10  # seconds

    board = BoardShim(BoardIds.CYTON_BOARD.value, params)

    try:
        # Prepare session and start streaming
        board.prepare_session()
        board.start_stream()
        print("Starting data collection...")

        for i in range(8):
            print(f"Collecting dataset {i}...")
            time.sleep(duration)

            # Retrieve raw data
            raw_data = board.get_board_data()

            # Extract EMG channels
            emg_channels = BoardShim.get_emg_channels(BoardIds.CYTON_BOARD.value)
            emg_data = raw_data[emg_channels, :]
            print(f"Shape of emg_data before clipping: {emg_data.shape}")

            # Clip the first and last second of data
            samples_to_clip = sampling_rate  # 1 second of data
            emg_data_clipped = emg_data[:, samples_to_clip:-samples_to_clip]

            # Store processed data
            np.save("emg_datasets" + str(i) + ".npy", emg_data_clipped)

        print("Data collection complete.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        board.stop_stream()
        board.release_session()

# Collect EMG data
collect_data()

for i in range (7):
    dataset=np.load(f"emg_datasets{i+1}.npy")
    print(dataset.shape)