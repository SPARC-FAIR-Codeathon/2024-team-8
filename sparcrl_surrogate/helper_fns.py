
import matplotlib.pyplot as plt
import h5py
import numpy as np
from scipy.interpolate import interp1d
import os 
from PIL import Image, ImageTk, ImageSequence
import tkinter as tk
import openai

# Set the OpenAI API key
openai.api_key = "REPLACE WITH YOUR API_KEY"

# Parse the dataset title from the file path
def get_group_keys(hdf5_file):
    with h5py.File(hdf5_file, "r") as hdf:
        path, _ = os.path.splitext(hdf5_file)
        path_parts = path.split("/")
        subject_name = path_parts[-1]
        subject_group = hdf[subject_name]
        group_keys = list(subject_group.keys())
    return group_keys

# Extract data from the HDF5 file
# This function is specific to the dataset used during prototype development and needs to be 
# adapted to the structure of the data in your own dataset
def extract_data(hdf5_file, group_name):
    with h5py.File(hdf5_file, "r") as hdf:
        path, _ = os.path.splitext(hdf5_file)
        path_parts = path.split("/")
        subject_name = path_parts[-1]        
        subject_group = hdf[subject_name]
        timestamp_group = subject_group[group_name]

        # Load the datasets
        hr_data = timestamp_group["hr"][:]
        neural_data = timestamp_group["neural"][:]
        time_data = neural_data[:, 0]  # Assuming the first column is time
        stim_timing = neural_data[:, -1]  # Assuming the last column is stim_timing

        # Initialize the burst signal with zeros
        burst_signal = np.zeros_like(stim_timing)

        # Detect stimuli and define bursts
        active_indices = np.where(stim_timing > 0)[0]
        bursts = []
        burst_duration = 5.0  # 5 seconds burst duration

        i = 0
        while i < len(active_indices):
            start_index = active_indices[i]  # First occurrence of a spike in this burst
            stim_start_time = time_data[start_index]
            stim_end_time = stim_start_time + burst_duration  # Add 5 seconds to define the end of the burst
            
            # Find the index that corresponds to the end time (or the closest match)
            end_index = np.searchsorted(time_data, stim_end_time)

            # Ensure the end_index does not exceed the length of the dataset
            if end_index >= len(time_data):
                end_index = len(time_data) - 1

            bursts.append((start_index, end_index))
            
            # Set burst signal to 1 for this burst
            burst_signal[start_index:end_index + 1] = 1
            
            # Move i forward to find the next burst (after 5 seconds)
            i = np.searchsorted(active_indices, end_index)

        # Extract stimulation parameters
        stimulation_parameters = {attr_name: attr_value for attr_name, attr_value in timestamp_group.attrs.items()}

    burst_signal_t = time_data - time_data[0]
    hr_signal_t = hr_data[:, 0] - hr_data[0, 0]

    # Interpolate the burst signal to match the HR signal timestamps
    interpolation_function = interp1d(burst_signal_t, burst_signal, kind='nearest', fill_value="extrapolate")
    burst_signal_resampled = interpolation_function(hr_signal_t)

    return hr_signal_t, hr_data, burst_signal_resampled, stimulation_parameters, timestamp_group

# Extract and concatenate data from the HDF5 file for later training
# This function is specific to the dataset used during prototype development and needs to be
def extract_concatenated_data(file_path, plot=False):
    hr_signal_concatenated = []
    frequency_signal_concatenated = []
    current_signal_concatenated = []
    t_concatenated = []

    # Plot the burst signal for each group
    for group_name in get_group_keys(file_path):
        hr_signal_t, hr_data, burst_signal, stimulation_parameters, timestamp_group = extract_data(file_path, group_name)
        
        frequency = stimulation_parameters["frequency_Hz"]
        current = stimulation_parameters["current_uA"]

        frequency_signal = burst_signal * frequency
        current_signal = burst_signal * current

        if plot:
            plt.figure(figsize=(12, 8))
            plt.subplot(3, 1, 1)
            plt.plot(hr_signal_t, hr_data[:, 1], label="Heart Rate (BPM)")
            plt.xlabel("Time (s)")
            plt.ylabel("Heart Rate (BPM)")
            plt.title("Heart Rate Data")
            plt.subplot(3, 1, 2)
            plt.plot(hr_signal_t, frequency_signal, label="Frequency (Hz)")
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")
            plt.title("Stimulation Frequency Signal")
            plt.ylim(0, 30)
            plt.subplot(3, 1, 3)
            plt.plot(hr_signal_t, current_signal, label="Current (uA)")
            plt.xlabel("Time (s)")
            plt.ylabel("Current (uA)")
            plt.ylim(0, 2500)
            plt.title("Stimulation Current Signal")
            plt.suptitle(f"Group: {group_name}, Frequency: {frequency}, Current: {current}")
            plt.tight_layout()
            plt.show()

        # concatenate time
        t_concatenated.extend(hr_signal_t+t_concatenated[-1] if t_concatenated else hr_signal_t)
        # concatenate heart rate signal
        hr_signal_concatenated.extend(hr_data[:, 1])
        # concatenate frequency signal
        frequency_signal_concatenated.extend(frequency_signal)
        # concatenate current signal
        current_signal_concatenated.extend(current_signal)

    return t_concatenated, hr_signal_concatenated, frequency_signal_concatenated, current_signal_concatenated, timestamp_group


def plot_concatenated_data(t_concatenated, hr_signal_concatenated, freq_signal_concatenated, current_signal_concatenated):
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(t_concatenated, hr_signal_concatenated, label="Heart Rate (BPM)")
    plt.xlabel("Time (s)")
    plt.ylabel("Heart Rate (BPM)")
    plt.title("Heart Rate Data (Concatenated)")
    plt.subplot(3, 1, 2)
    plt.plot(t_concatenated, freq_signal_concatenated, label="Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Stimulation Frequency Signal (Concatenated)")
    plt.ylim(0, 30)
    plt.subplot(3, 1, 3)
    plt.plot(t_concatenated, current_signal_concatenated, label="Current (uA)")
    plt.xlabel("Time (s)")
    plt.ylabel("Current (uA)")
    plt.ylim(0, 2500)
    plt.title("Stimulation Current Signal (Concatenated)")
    plt.tight_layout()
    plt.show()

# Function to display an animated GIF in a Tkinter window (used for the loading animation)
class AnimatedGIF(tk.Label):
    def __init__(self, master, gif_path, width=None, height=None, delay=100):
        # Initialize the Label widget
        tk.Label.__init__(self, master)
        
        # Load the GIF using PIL
        self.gif = Image.open(gif_path)
        
        # Resize frames if width and height are provided
        if width and height:
            self.frames = [
                ImageTk.PhotoImage(frame.copy().resize((width, height)))
                for frame in ImageSequence.Iterator(self.gif)
            ]
        else:
            self.frames = [ImageTk.PhotoImage(frame.copy()) for frame in ImageSequence.Iterator(self.gif)]
        
        # Store the delay (time between frames)
        self.delay = delay
        
        # Start the animation
        self.current_frame = 0
        self.animate()

    def animate(self):
        # Update the label with the current frame
        self.config(image=self.frames[self.current_frame])
        
        # Move to the next frame
        self.current_frame = (self.current_frame + 1) % len(self.frames)
        
        # Schedule the next frame update
        self.after(self.delay, self.animate)

# Function to generate a suggestion for the purpose of a given dataset based on LLM
def data_set_purpose_suggestion(dataset_title):
    response = openai.completions.create(
        model="gpt-3.5-turbo-instruct",  # Ensure you're specifying the correct model
        prompt=f"Given the dataset titled '{dataset_title}', provide two sentences on how this dataset could be used to train a reinforcement learning algorithm. The suggestions should be specific to the data and focus on potential applications in optimizing or predicting outcomes.",
        max_tokens=500
    )

    suggestion_text = response.choices[0].text.strip()
    # remove new lines
    suggestion_text = suggestion_text.replace('\n', ' ')
    return suggestion_text
