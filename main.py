import numpy as np
import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes

def collect_data():
    params = BrainFlowInputParams()
    params.serial_port = "COM4"
    sampling_rate = 250
    duration = 10  # seconds

    board = BoardShim(BoardIds.CYTON_BOARD.value, params)

    try:
        # Prepare session and start streaming
        board.prepare_session()
        board.start_stream()
        print("Starting data collection...")

        for i in range(15):
            print(f"Collecting dataset {i}...")
            time.sleep(duration)

            # Retrieve raw data
            raw_data = board.get_board_data()

            # Extract EMG channels
            emg_channels = BoardShim.get_emg_channels(BoardIds.CYTON_BOARD.value)
            emg_data = raw_data[emg_channels, :]
            print(f"Begin next motion. Shape of emg_data: {emg_data.shape}")
             
            #for channel in emg_channels:
            #    DataFilter.perform_notch(emg_data[channel], sampling_rate, 50.0, 4, FilterTypes.BUTTERWORTH.value)
            #    DataFilter.perform_highpass(emg_data[channel], sampling_rate, 20,4, FilterTypes.BUTTERWORTH.value)
            # Store processed data
            np.save("C:\\Users\\AndrewWPI\\Desktop\\STEMI\\Data\\Partcipant1-16f\\test3\\emg_datasets" + str(i) + ".npy", emg_data)
            time.sleep(7)

        print("Data collection complete.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        board.stop_stream()
        board.release_session()

# Collect EMG data
collect_data()