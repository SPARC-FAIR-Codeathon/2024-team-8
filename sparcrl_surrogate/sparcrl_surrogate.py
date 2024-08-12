import tkinter as tk
from tkinter import ttk, filedialog
from sparc.client import SparcClient
import os
import json
import requests
from PIL import Image, ImageTk, ImageSequence
import zipfile
from helper_fns import extract_concatenated_data, plot_concatenated_data, AnimatedGIF, data_set_purpose_suggestion
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import io
from tkinter import font
import pandas as pd
import numpy as np
import openai 
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError

openai.api_key = "### REPLACE BY YOUR API KEY ###"

# Initialize the SPARC client (assuming connection is not required)
client = SparcClient(connect=False)
# Get the list of datasets
response = client.pennsieve.list_datasets(query='time series', organization='SPARC')
datasets = {dataset['name']: dataset['id'] for dataset in response['datasets']}

# global variable to store app data
app_data = {'trainin_data_1': {}, 'training_data_2': {}, 'training_data_3': {}}

# Function to update the text widget with the selected dataset's JSON details
def update_dataset_info(event=None):
    # enable get dataset button and next button
    ok_button.config(state="normal")
    next_button.config(state="normal")

    selected_dataset = dataset_var.get()
    dataset_id = datasets[selected_dataset]
    
    # Download the dataset files (adjust the query and file_type as needed)
    response = client.pennsieve.list_files(dataset_id=dataset_id, query='manifest', file_type='json')
    
    client.pennsieve.download_file(file_list=response[0], output_name=f'./manifest.json')
    
    with open('manifest.json', 'r') as file:
        data = json.load(file)

    dataset_name_label.config(text=f"Dataset Name: {data['name']}")  # Update the label text
    dataset_description_label.config(text=f"Description: {data['description']}")
    dataset_version_label.config(text=f"Version: {data['version']}")
    dataset_date_label.config(text=f"Published Date: {data['datePublished']}")
    dataset_license_label.config(text=f"License: {data['license']}")
    dataset_doi_label.config(text=f"DOI: {data['@id']}")
    dataset_creator_label.config(text=f"Creator: {data['creator']['first_name']} {data['creator']['last_name']} (ORCID: {data['creator']['orcid']})")
    dataset_keywords_label.config(text=f"Keywords: {', '.join(data['keywords'])}")

    app_data['dataset_folder_name'] = f"Pennsieve-dataset-{dataset_id}-version-{data['version']}"

    # Get the dataset purpose suggestion from OpenAI
    dataset_purpose_suggestion = data_set_purpose_suggestion(data['name'])
    print(f"Dataset Purpose Suggestion: {dataset_purpose_suggestion}")
    usage_sugggestion_label.config(text=f"Usage Suggestion: {dataset_purpose_suggestion}")

    root.update_idletasks()
    root.update()

# Function to download a file from a URL and save it to a specified folder
def download_file(dataset_id, save_folder, progress_label):
    global app_data
    url = f"https://api.pennsieve.io/discover/datasets/{dataset_id}/versions/1/download?downloadOrigin=SPARC"
    # Get the filename from the URL
    local_filename = f'dataset_{dataset_id}.zip'
    # Construct the full path where the file will be saved
    file_path = os.path.join(save_folder, local_filename)

    app_data['base_path'] = save_folder
    
    progress_spinner.grid(column=0, row=15, padx=10, pady=0, sticky=tk.W)

    # Perform the download
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))  # Total size in bytes
        print("Total size:", total_size)
        block_size = 8192  # 8 KB
        
        downloaded_size = 0
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=block_size):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    progress_label.config(text=f"{downloaded_size/1e6:.1f} MB downloaded")
                    root.update_idletasks()
                    root.update()
    
    print(f"File downloaded: {file_path}")
    progress_label.config(text="Download complete!")
    root.update_idletasks()
    root.update()

    progress_label.config(text="Unzipping files...")
    root.update_idletasks()
    root.update()

    # Unzip the downloaded file
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(save_folder)
        print(f"Files extracted to: {save_folder}")
    
    progress_label.config(text="Unzipping complete!")
    root.update_idletasks()
    root.update()

    next_button.config(state="normal")
    progress_spinner.grid_forget()
    progress_label.grid(column=0, row=15, padx=10, pady=0, sticky=tk.W)
    root.update_idletasks()
    root.update()
    
    return file_path

# Function to handle the button click
def on_ok():
    ok_button.config(state="disabled") 
    root.update_idletasks()
    root.update()
    selected_dataset = dataset_var.get()
    dataset_id = datasets[selected_dataset]
    print(f"Selected Dataset: {selected_dataset}")
    print(f"Dataset ID: {dataset_id}")

    # Open a dialog to select the folder where the files will be saved
    folder_selected = filedialog.askdirectory()
    if not folder_selected:
        print("No folder selected. Download canceled.")
        return

    # Download the dataset files (adjust the query and file_type as needed)
    response = client.pennsieve.list_files(dataset_id=dataset_id, query='manifest', file_type='json')
    
    print(f"Downloading {response} to {folder_selected}...")
    client.pennsieve.download_file(file_list=response[0], output_name=f'{folder_selected}/manifest.json')
    
    # Display progress bar and start downloading
    download_file(dataset_id, folder_selected, progress_label)

    print("Download complete.")


def on_next():
    # Open a dialog to select the folder where the files will be saved
    # Load the downloaded JSON file
    with open('manifest.json', 'r') as file:
        data = json.load(file)

    # Hide the main window
    root.withdraw()

    # Show the next window with file selection
    show_file_selection_window(data)

def on_next2():
    global app_data
    # Open a dialog to select the folder where the files will be saved
    # Load the downloaded JSON file
    with open('manifest.json', 'r') as file:
        data = json.load(file)
    
    # Hide the main window
    root.withdraw()
    app_data['new_window'].withdraw()
    show_surrogate_training_config_window(data)


def show_surrogate_training_config_window(data):
    new_window = tk.Toplevel(root)
    new_window.title("SPARC.RL -- Data-Driven Reinforcement Learning for SPARC Datasets")
    new_window.geometry("550x750")

    # Load the image using PIL
    original_image = Image.open("sparcrl_logo.png")

    # Resize the image
    base_width = 150
    w_percent = (base_width / float(original_image.size[0]))
    h_size = int((float(original_image.size[1]) * float(w_percent)))
    resized_image = original_image.resize((base_width, h_size))

    # Convert the resized image to a format tkinter can use
    tk_image = ImageTk.PhotoImage(resized_image)

    # Create a label to display the image
    image_label = tk.Label(new_window, image=tk_image)
    image_label.grid(row=0, padx=20, pady=10, sticky=tk.W)  # Adjust padding as needed

    # Keep a reference to the image to prevent it from being garbage collected
    image_label.image = tk_image

    # Create a label to show the download status
    progress_label = tk.Label(new_window, text="")
    progress_label.grid(column=0, row=12, padx=40, pady=0, sticky=tk.W)

    # Dropdown that allows to select from LSTM, GRU, and BiLSTM and RNN
    model_var = tk.StringVar(new_window)
    model_var.set("LSTM")
    model_var.set("BiLSTM")
    model_var.set("GRU")
    model_var.set("RNN")
    model_dropdown_label = tk.Label(new_window, text="Select a model to train:", font=("Arial", 13, "bold"))
    model_dropdown_label.grid(column=0, row=1, padx=10, sticky=tk.W)

    # create the combo box
    model_dropdown = ttk.Combobox(new_window, textvariable=model_var, width=54)
    model_dropdown['values'] = ["LSTM", "BiLSTM", "GRU", "RNN"]
    model_dropdown.grid(column=0, row=2, padx=10, pady=5, sticky=tk.W)
    model_dropdown.current(0)  # Set the default selection

    # Dropdown to select number of layers
    layer_var = tk.StringVar(new_window)
    layer_var.set("1")
    layer_var.set("2")
    layer_var.set("3")
    layer_var.set("4")
    layer_dropdown_label = tk.Label(new_window, text="Select number of layers:", font=("Arial", 13, "bold"))
    layer_dropdown_label.grid(column=0, row=3, padx=10, sticky=tk.W)

    # create the combo box
    layer_dropdown = ttk.Combobox(new_window, textvariable=layer_var, width=54)
    layer_dropdown['values'] = ["1", "2", "3", "4"]
    layer_dropdown.grid(column=0, row=4, padx=10, pady=5, sticky=tk.W)
    layer_dropdown.current(0)  # Set the default selection

    # Dropdown to select number of units per layer
    units_var = tk.StringVar(new_window)
    units_var.set("64")
    units_var.set("128")
    units_var.set("256")
    units_var.set("512")
    units_dropdown_label = tk.Label(new_window, text="Select number of units per layer:", font=("Arial", 13, "bold"))
    units_dropdown_label.grid(column=0, row=5, padx=10, sticky=tk.W)

    # create the combo box
    units_dropdown = ttk.Combobox(new_window, textvariable=units_var, width=54)
    units_dropdown['values'] = ["64", "128", "256", "512"]
    units_dropdown.grid(column=0, row=6, padx=10, pady=5, sticky=tk.W)
    units_dropdown.current(0)  # Set the default selection

    # Dropdown to select optimizer
    optimizer_var = tk.StringVar(new_window)
    optimizer_var.set("Adam")
    optimizer_var.set("SGD")
    optimizer_var.set("RMSprop")
    optimizer_var.set("Adagrad")
    optimizer_var.set("Adadelta")
    optimizer_var.set("Adamax")
    optimizer_var.set("Nadam")
    optimizer_dropdown_label = tk.Label(new_window, text="Select optimizer:", font=("Arial", 13, "bold"))
    optimizer_dropdown_label.grid(column=0, row=7, padx=10, sticky=tk.W)

    # create the combo box
    optimizer_dropdown = ttk.Combobox(new_window, textvariable=optimizer_var, width=54)
    optimizer_dropdown['values'] = ["Adam", "SGD", "RMSprop", "Adagrad", "Adadelta", "Adamax", "Nadam"]
    optimizer_dropdown.grid(column=0, row=8, padx=10, pady=5, sticky=tk.W)
    optimizer_dropdown.current(0)  # Set the default selection

    # Dropdown to select learning rate
    lr_var = tk.StringVar(new_window)
    lr_var.set("0.0001")
    lr_var.set("0.001")
    lr_var.set("0.01")
    lr_var.set("0.1")
    lr_dropdown_label = tk.Label(new_window, text="Select learning rate:", font=("Arial", 13, "bold"))
    lr_dropdown_label.grid(column=0, row=9, padx=10, sticky=tk.W)

    # create the combo box
    lr_dropdown = ttk.Combobox(new_window, textvariable=lr_var, width=54)
    lr_dropdown['values'] = ["0.0001", "0.001", "0.01", "0.1"]
    lr_dropdown.grid(column=0, row=10, padx=10, pady=5, sticky=tk.W)
    lr_dropdown.current(0)  # Set the default selection

    # Dropdown to select batch size
    batch_size_var = tk.StringVar(new_window)
    batch_size_var.set("32")
    batch_size_var.set("64")
    batch_size_var.set("128")
    batch_size_var.set("256")
    batch_size_dropdown_label = tk.Label(new_window, text="Select batch size:", font=("Arial", 13, "bold"))
    batch_size_dropdown_label.grid(column=0, row=11, padx=10, sticky=tk.W)

    # create the combo box
    batch_size_dropdown = ttk.Combobox(new_window, textvariable=batch_size_var, width=54)
    batch_size_dropdown['values'] = ["32", "64", "128", "256"]
    batch_size_dropdown.grid(column=0, row=12, padx=10, pady=5, sticky=tk.W)
    batch_size_dropdown.current(0)  # Set the default selection

    # Dropdown to select number of epochs
    epochs_var = tk.StringVar(new_window)
    epochs_var.set("10")
    epochs_var.set("20")
    epochs_var.set("50")
    epochs_var.set("100")
    epochs_dropdown_label = tk.Label(new_window, text="Select number of epochs:", font=("Arial", 13, "bold"))
    epochs_dropdown_label.grid(column=0, row=13, padx=10, sticky=tk.W)

    # create the combo box
    epochs_dropdown = ttk.Combobox(new_window, textvariable=epochs_var, width=54)
    epochs_dropdown['values'] = ["10", "20", "50", "100"]
    epochs_dropdown.grid(column=0, row=14, padx=10, pady=5, sticky=tk.W)
    epochs_dropdown.current(0)  # Set the default selection
    
    # Create Checkboxes for Early Stopping and Model Checkpoint
    early_stopping_var = tk.IntVar(new_window)
    early_stopping_checkbox = tk.Checkbutton(new_window, text="Enable Early Stopping", variable=early_stopping_var, font=("Arial", 13, "bold"))
    early_stopping_checkbox.grid(column=0, row=15, padx=10, sticky=tk.W)

    model_checkpoint_var = tk.IntVar(new_window)
    model_checkpoint_checkbox = tk.Checkbutton(new_window, text="Enable Model Checkpoint", variable=model_checkpoint_var, font=("Arial", 13, "bold"))
    model_checkpoint_checkbox.grid(column=0, row=16, padx=10, sticky=tk.W)
    
    # Function to train the model    # Train button function
    def train_model():
        global app_data
        # Get selected values from the GUI
        model_type = model_var.get()
        num_layers = int(layer_var.get())
        units_per_layer = int(units_var.get())
        optimizer_type = optimizer_var.get()
        learning_rate = float(lr_var.get())
        batch_size = int(batch_size_var.get())
        epochs = int(epochs_var.get())
        early_stopping = early_stopping_var.get()
        model_checkpoint = model_checkpoint_var.get()

        # Define the model based on the selected values
        model = tf.keras.Sequential(name="Surrogate_Model")
        model.add(tf.keras.layers.InputLayer((1, 2)))  # Adjusted input shape

        for i in range(num_layers):
            if model_type == "LSTM":
                model.add(tf.keras.layers.LSTM(units_per_layer, return_sequences=True if i < num_layers - 1 else False))
            elif model_type == "BiLSTM":
                model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units_per_layer, return_sequences=True if i < num_layers - 1 else False)))
            elif model_type == "GRU":
                model.add(tf.keras.layers.GRU(units_per_layer, return_sequences=True if i < num_layers - 1 else False))
            elif model_type == "RNN":
                model.add(tf.keras.layers.SimpleRNN(units_per_layer, return_sequences=True if i < num_layers - 1 else False))

        model.add(tf.keras.layers.Dense(8, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='linear'))

        # Choose optimizer
        if optimizer_type == "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_type == "SGD":
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_type == "RMSprop":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer_type == "Adagrad":
            optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
        elif optimizer_type == "Adadelta":
            optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
        elif optimizer_type == "Adamax":
            optimizer = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
        elif optimizer_type == "Nadam":
            optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)

        # Compile the model
        # Compile model
        model.compile(loss=MeanSquaredError(), 
                      optimizer=optimizer, 
                      metrics=[RootMeanSquaredError()])

        # Callbacks
        callbacks = []
        if early_stopping:
            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3))
        if model_checkpoint:
            callbacks.append(tf.keras.callbacks.ModelCheckpoint('model_checkpoint.h5', save_best_only=True))

        # Capture model summary as a string
        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + '\n'))
        summary_str = stream.getvalue()

        # Insert the summary into the Text widget
        summary_text.delete(1.0, tk.END)  # Clear previous content
        summary_text.insert(tk.END, summary_str)

        summary_text.grid(column=0, row=19, padx=10, pady=10, sticky=tk.W)

        new_window.update_idletasks()
        new_window.update()

        def create_feature_matrix(x, y, input_sequence):
            X = []
            Y = []
            
            for i in range(len(x) - input_sequence):
                features = np.hstack((x[i:i + input_sequence], y[i:i + input_sequence]))
                X.append(features)
                label = y[i + input_sequence]
                Y.append(label)
                
            return np.array(X), np.array(Y)


        # Show training progress spinner and label, disable the train button
        progress_spinner.grid(column=0, row=17, padx=10, pady=0, sticky=tk.W)
        progress_label.grid(column=0, row=17, padx=40, pady=0, sticky=tk.W)
        train_button.config(state="disabled")

        new_window.update_idletasks()
        new_window.update()

        print("loading data...")
        x = np.genfromtxt(app_data['training_data1']['path'], delimiter=',', filling_values=np.nan)
        x = x[~np.isnan(x)]
        y = np.genfromtxt(app_data['training_data1']['path'], delimiter=',', filling_values=np.nan)
        y = y[~np.isnan(y)]

        print("done loading data")
        x_u = x.mean()
        x_std = x.std()
        y_u = y.mean()
        y_std = y.std()
        # Normalize
        x = (x - x_u) / x_std
        y = (y - y_u) / y_std

        seq_len = 5000

        # Define input sequence length
        n_input = 100  # You can adjust this based on your requirements

        print("creating feature matrix...")
        # Create feature matrix
        X, Y = create_feature_matrix(x, y, n_input)

        print("done creating feature matrix...")

        print("Reshaping feature matrix...")
        # Adjust the feature shape for LSTM input
        X = X.reshape((X.shape[0], n_input, 2))  # 2 features: x and y

        print("done creating feature matrix...")

        # Split into train, validation, and test sets
        X_train = X[0:int((len(X)/seq_len)*0.8)*seq_len]
        y_train = Y[0:int((len(X)/seq_len)*0.8)*seq_len]
        X_val = X[int((len(X)/seq_len)*0.8)*seq_len+1:int((len(X)/seq_len)*0.9)*seq_len]
        y_val = Y[int((len(X)/seq_len)*0.8)*seq_len+1:int((len(X)/seq_len)*0.9)*seq_len]
        X_test = X[int((len(X)/seq_len)*0.9)*seq_len+1:]
        y_test = Y[int((len(X)/seq_len)*0.9)*seq_len+1:]

        print("Training model...")

        # Train model
        model.fit(X_train, y_train, 
                validation_data=(X_val, y_val),epochs=epochs, batch_size=batch_size, callbacks=callbacks)

        print("Model training complete.")

        print("Evaluating model...")

        # Predict
        test_predictions = model.predict(X_test).flatten()

        # Prepare predictions for plotting
        X_test_list = [X_test[i][0] for i in range(len(X_test))]

        # Create DataFrame for predictions
        test_predictions_df = pd.DataFrame({'X_test': list(X_test_list), 
                                            'LSTM Prediction': list(test_predictions)})

        test_predictions_df.plot(figsize = (15,6))
        
        # save the model
        model.save(f'./training_data/{app_data["dataset_folder_name"]}_model.h5')

        # update progress label
        

    # Create and pack the AnimatedGIF widget
    progress_spinner = AnimatedGIF(new_window, "spinner2.gif", delay=100, width=25, height=25)
    progress_spinner.grid(column=0, row=17, padx=10, pady=10, sticky=tk.W)
    progress_spinner.grid_forget()

    # Create a label to show the download status
    progress_label = tk.Label(new_window, text="Training in progress. Grab a coffee, this will take a while...")
    progress_label.grid(column=0, row=17, padx=40, pady=10, sticky=tk.W)
    progress_label.grid_forget()

    # Create the "Train!" button in the new window on the east
    train_button = tk.Button(new_window, text="Train!", command=train_model)
    train_button.grid(column=0, row=18, padx=17, pady=10, sticky=tk.W)

    # Create a Text widget for displaying the model summary with a monospaced font
    summary_text = tk.Text(new_window, wrap=tk.WORD, height=15, width=100)
    summary_text.grid(column=0, row=19, padx=10, pady=10, sticky=tk.W)

    # Set a monospaced font
    monospace_font = font.Font(family="Courier", size=8)
    summary_text.configure(font=monospace_font)
    summary_text.grid_forget()


# Function to show the next window with a dropdown of available files
def show_file_selection_window(data):
    global app_data
    new_window = tk.Toplevel(root)
    new_window.title("SPARC.RL -- Data-Driven Reinforcement Learning for SPARC Datasets")
    new_window.geometry("550x450")

    # Load the image using PIL
    original_image = Image.open("sparcrl_logo.png")

    # Resize the image
    base_width = 150
    w_percent = (base_width / float(original_image.size[0]))
    h_size = int((float(original_image.size[1]) * float(w_percent)))
    resized_image = original_image.resize((base_width, h_size))

    # Convert the resized image to a format tkinter can use
    tk_image = ImageTk.PhotoImage(resized_image)

    # Create a label to display the image
    image_label = tk.Label(new_window, image=tk_image)
    image_label.grid(row=0, padx=20, pady=10, sticky=tk.W)  # Adjust padding as needed

    # Keep a reference to the image to prevent it from being garbage collected
    image_label.image = tk_image

    file_var = tk.StringVar(new_window)

    # Create a dictionary to map file names to their paths, filtering for .hdf5 files
    file_dict = {file_info['name']: file_info['path'] for file_info in data['files'] if file_info['name'].endswith('.hdf5')}
    file_names = list(file_dict.keys())
    
    if not file_names:
        print("No .hdf5 files found in the dataset.")
        file_names = ["No .hdf5 files found in the dataset."]

    file_dropdown_label = tk.Label(new_window, text="Select a file to load (only .hdf5 supported):", font=("Arial", 13, "bold"))
    file_dropdown_label.grid(column=0, row=1, padx=10, sticky=tk.W)

    # Create the dropdown menu
    file_dropdown = ttk.Combobox(new_window, textvariable=file_var)
    file_dropdown['values'] = file_names
    file_dropdown.grid(column=0, row=2, padx=10, pady=10, sticky=tk.W)
    file_dropdown.current(0)  # Set the default selection

    # Function to handle file selection
    def on_file_select():
        global app_data
        selected_file = file_var.get()
        if selected_file in file_dict:
            selected_path = file_dict[selected_file]
            print(f"Selected File: {selected_file}")
            print(f"File Path: {app_data['base_path']}/{selected_path}")
            app_data['selected_file'] = f"{app_data['base_path']}/{selected_path}"
            selected_file = file_var.get()
            selected_path = file_dict[selected_file]
            t_concatenated_subj1, hr_signal_concatenated_subj1, freq_signal_concatenated_subj1, current_signal_concatenated_subj1, timestamp_group = extract_concatenated_data(f"{app_data['base_path']}/{app_data['dataset_folder_name']}/{selected_path}")
            print(timestamp_group)
            plot_concatenated_data(t_concatenated_subj1, hr_signal_concatenated_subj1, freq_signal_concatenated_subj1, current_signal_concatenated_subj1)
            # check if training_data folder exists if not create
            if not os.path.exists('./training_data'):
                os.makedirs('./training_data')
            

            # that part is specific to the SPARC dataset used during developing the prototype
            # this section needs to be modified to fit the dataset being used
            current_signal_path = f'./training_data/current_{app_data["dataset_folder_name"]}_{selected_file}.csv'
            freq_signal_path = f'./training_data/freq_{app_data["dataset_folder_name"]}_{selected_file}.csv'
            hr_signal_path = f'./training_data/hr_{app_data["dataset_folder_name"]}_{selected_file}.csv'

            app_data['training_data1'] = {
                'file': selected_file,
                'path': current_signal_path,
                'signal': 'current'            
            }

            app_data['training_data2'] = {
                'file': selected_file,
                'path': freq_signal_path,
                'signal': 'freq'
            }

            app_data['training_data3'] = {
                'file': selected_file,
                'path': hr_signal_path,
                'signal': 'hr'
            }

            print("APP DATA SET: ")
            print(app_data)

            np.savetxt(current_signal_path, current_signal_concatenated_subj1, delimiter=',', fmt='%.1f', newline=',')
            np.savetxt(freq_signal_path, freq_signal_concatenated_subj1, delimiter=',', fmt='%.1f', newline=',')
            np.savetxt(hr_signal_path, hr_signal_concatenated_subj1, delimiter=',', fmt='%.1f', newline=',')
        else:
            print("No valid file selected.")

    # Create the "OK" button in the new window
    ok_button = tk.Button(new_window, text="Plot Data!", command=on_file_select)
    ok_button.grid(column=0, row=3, padx=10, pady=10, sticky=tk.W)

    app_data['new_window'] = new_window
    next_button_2 = tk.Button(new_window, text="Next", command=on_next2)
    next_button_2.grid(column=0, row=3, padx=10, pady=10, sticky=tk.E)


# Create the main application window
root = tk.Tk()
root.title("SPARC.RL -- Data-Driven Reinforcement Learning for SPARC Datasets")
root.geometry("550x650")

# Dropdown variable
dataset_var = tk.StringVar(root)

# Load the image using PIL
original_image = Image.open("sparcrl_logo.png")

# Resize the image
# Let's say you want to resize the image to a width of 300 pixels
base_width = 150
w_percent = (base_width / float(original_image.size[0]))
h_size = int((float(original_image.size[1]) * float(w_percent)))
resized_image = original_image.resize((base_width, h_size))

# Convert the resized image to a format tkinter can use
tk_image = ImageTk.PhotoImage(resized_image)

# Create a label to display the image
image_label = tk.Label(root, image=tk_image)
image_label.grid(row=0, padx=20, pady=10, sticky=tk.W)  # Adjust padding as needed

dropdown_label = tk.Label(root, text="Time-Series Datasets:", font=("Arial", 13, "bold"))
dropdown_label.grid(column=0, row=1, padx=10, sticky=tk.W)

# Create the dropdown menu for datasets
dropdown = ttk.Combobox(root, textvariable=dataset_var, width=54)
dropdown['values'] = list(datasets.keys())
dropdown.grid(column=0, row=2, padx=10, pady=5, sticky=tk.W)
dropdown.current(0)  # Set the default selection
dropdown.bind("<<ComboboxSelected>>", update_dataset_info)  # Bind the event

# Labels for Dataset Information
dataset_info_label = tk.Label(root, text="Dataset Information:", font=("Arial", 13, "bold"), wraplength=300, justify="left", padx=0, pady=0)
dataset_info_label.grid(column=0, row=3, padx=10, pady=5, sticky=tk.W)

dataset_name_label = tk.Label(root, text="Dataset Name:", font=("Arial", 12), wraplength=500, justify="left", padx=0, pady=0)
dataset_name_label.grid(column=0, row=4, padx=10,  sticky=tk.W)

dataset_description_label = tk.Label(root, text="Description:", font=("Arial", 12), wraplength=500, justify="left", padx=0, pady=0)
dataset_description_label.grid(column=0, row=5, padx=10, sticky=tk.W)

dataset_version_label = tk.Label(root, text="Version:", font=("Arial", 12), wraplength=500, justify="left", padx=0, pady=0)
dataset_version_label.grid(column=0, row=6, padx=10, sticky=tk.W)

dataset_date_label = tk.Label(root, text="Published Date:", font=("Arial", 12), wraplength=500, justify="left", padx=0, pady=0)
dataset_date_label.grid(column=0, row=7, padx=10, sticky=tk.W)

dataset_license_label = tk.Label(root, text="License:", font=("Arial", 12), wraplength=500, justify="left", padx=0, pady=0)
dataset_license_label.grid(column=0, row=8, padx=10, sticky=tk.W)

dataset_doi_label = tk.Label(root, text="DOI:", font=("Arial", 12), wraplength=500, justify="left", padx=0, pady=0)
dataset_doi_label.grid(column=0, row=9, padx=10, sticky=tk.W)

dataset_creator_label = tk.Label(root, text="Creator:", font=("Arial", 12), wraplength=500, justify="left", padx=0, pady=0)
dataset_creator_label.grid(column=0, row=10, padx=10, sticky=tk.W)

dataset_keywords_label = tk.Label(root, text="Keywords:", font=("Arial", 12), wraplength=500, justify="left", padx=0, pady=0)
dataset_keywords_label.grid(column=0, row=11, padx=10, sticky=tk.W)

usage_suggestion_title_label = tk.Label(root, text="Suggested Usage:", font=("Arial", 13, "bold"), wraplength=300, justify="left", padx=0, pady=0)
usage_suggestion_title_label.grid(column=0, row=12, padx=10, pady=5, sticky=tk.W)

usage_sugggestion_label = tk.Label(root, text="", font=("Arial", 12), wraplength=500, justify="left", padx=0, pady=0)
usage_sugggestion_label.grid(column=0, row=13, padx=10, sticky=tk.W)


# Create and pack the AnimatedGIF widget
progress_spinner = AnimatedGIF(root, "spinner.gif", delay=100, width=25, height=25)
progress_spinner.grid(column=0, row=15, padx=10, pady=0, sticky=tk.W)

progress_spinner.grid_forget()

# Create a label to show the download status
progress_label = tk.Label(root, text="")
progress_label.grid(column=0, row=15, padx=40, pady=0, sticky=tk.W)

# Create the "OK" button for dataset selection
ok_button = tk.Button(root, text="Get Dataset!", command=on_ok)
ok_button.grid(column=0, row=16, padx=10, pady=10, sticky=tk.W)

next_button = tk.Button(root, text="Next", command=on_next, state="disabled")
next_button.grid(column=0, row=16, padx=0, pady=10, sticky=tk.E)

update_dataset_info(event=None)

# Run the application
root.mainloop()
