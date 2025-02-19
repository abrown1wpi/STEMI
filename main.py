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
            print(f"Shape of emg_data: {emg_data.shape}")
            np.save("C:\\Users\\AndrewWPI\\Desktop\\STEMI\\Data\\Participant4-16\\test3\\emg_datasets" + str(i) + ".npy", emg_data)
            
            for channel in emg_channels:
                DataFilter.perform_bandpass(raw_data[channel], sampling_rate, 20.0, 450.0, 4, FilterTypes.BUTTERWORTH.value, 0)
        
                # High-pass Filter (20 Hz) to Remove Low-Frequency Drift
                DataFilter.perform_highpass(raw_data[channel], sampling_rate, 20.0, 4, FilterTypes.BUTTERWORTH.value, 0)
            
            # Store processed data
            np.save("C:\\Users\\AndrewWPI\\Desktop\\STEMI\\Data\\Participant4-16\\test3\\emg_datasets_filter" + str(i) + ".npy", emg_data)
            time.sleep(5)
            print("Begin next motion.")
            time.sleep(7)
            raw_data = board.get_board_data()

        print("Data collection complete.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        board.stop_stream()
        board.release_session()

# Collect EMG data
collect_data()