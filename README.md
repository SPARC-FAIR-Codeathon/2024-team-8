<img src="https://github.com/SPARC-FAIR-Codeathon/2024-team-8/blob/main/sparc_rl_logo.png?raw=true" style="display: block; width: 30%;"/>

## Data-Driven Reinforcement Learning for Medical Device Control Made Easy
SPARC.RL is a first proof-of-concept toolchain designed to enable researchers, even those without specific domain knowledge in reinforcement learning, to effortlessly train sophisticated, state-of-the-art reinforcement learning (RL) agents for robust medical device control, e.g. for closed-loop neuromodulation. It levarges the power of [Stable Baselines 3](https://stable-baselines3.readthedocs.io/), one of the most prominent and powerful reinforcement learning frameworks available. SPARC.RL allows users to integrate and utilize data from the [SPARC](https://sparc.science/) platform and models from [oSPARC](https://osparc.io/) seamlessly into a reinforcement learning pipeline. 

This toolchain was developed during the [2024 SPARC FAIR Codeathon](https://sparc.science/news-and-events/events/2024-sparc-fair-codeathon) by Max Haberbusch and John Bentley.

<i>Note: While this toolchain offers powerful capabilities, please note that it is an initial prototype serving as a proof of concept, and no guarantees are made regarding its bug-freeness and operability with other datasets than those used during development.</i>


### Key Features:
#### Dataset and Model Integration:

SPARC.RL supports the selection and use of time-series datasets directly loaded from the SPARC platform using the [SPARC Python client](https://docs.sparc.science/docs/sparc-python-client).
Users can also work with selected [oSPARC](https://osparc.io/) models, enabling the training of RL agents in a highly flexible and customizable manner.

##### Customizable Inputs and Outputs:
Users can choose from available model inputs (actions) and model outputs (observables) to tailor the reinforcement learning process to their specific needs.
That allows the selection of appropriate actions and observables that the RL agent should focus on.

#### Data-driven Modelling:

SPARC.RL offers multiple deep learning architectures to create surrogate models of experimental data available on [SPARC](https://sparc.science/) or [oSPARC](https://osparc.io) models.
Users can select from various RNNs optimized for time-series modeling, including vanilla recurrent neural networks (RNNs), long short-term memory (LSTM) networks, bi-directional LSTM (BiLSTM) networks, and gated recurrent units (GRUs), providing flexibility in how the models are trained. Users can configure their network and training parameters according to their needs. The adjustable parameters include the number of layers, number of units per layer, optimizer, learning rate, batch size, number of epochs, and early stopping policies.

#### Steps to Generate the Surrogate Model
In the first step, select a dataset from the dropdown menu which is automatically populated with available datasets on the SPARC platform. Currently, the datasets are limited to time series data. Once you have selected a model you can inspect the model metadata like model description, creator, creation date, version, etc. Additionally, a large language model is used to generate suggestions on how to use the dataset for reinforcement learning. Once you have chosen your dataset, you can download and extract the data from SPARC by hitting the 'Get Dataset!' button. You will be asked in what folder to save the data. After that, you can proceed to the next step, to select the file(s) to use for training the surrogate model by hitting the 'Next' button.<br/><br/>
<p align="center">
<img src="https://github.com/SPARC-FAIR-Codeathon/2024-team-8/blob/main/img/sparcrl_load_data_from_sparc.png?raw=true" alt="Select dataset from SPARC platform" width="500"/><br/>
  <b>Figure 1.</b> Select dataset from SPARC platform to train surrogate model.
</p><br/>
Once you have chosen and downloaded an appropriate dataset, you can select one of the available files containing experimental data using the dropdown menu. The data is automatically filtered for appropriate file types. Currently, only the .hdf5 file format is supported. After you have selected a file, the data is pre-processed to bring it in a proper format for training the model. You can display the pre-processeed data using by hitting the 'Plot Data!' button. If you are satisfied with the preprocessed data, you can move to the next step by hitting the 'Next' button.<br/>
<p align="center">
<img src="https://github.com/SPARC-FAIR-Codeathon/2024-team-8/blob/main/img/sparcrl_inspect_preprocessed_dataset.png?raw=true" alt="Select data file from dataset and preprocess" width="800"/><br/>
    <b>Figure 2.</b> Select a file from the dataset for model training and inspect preprocessed data.
</p>
<br/>
After loading the data for training, you can define your model architecture. For now, the tool allows you to use different types of recurrent neural networks including LSTM, Bi-LSTM, GRU and vanilla RNNs. You can adjust the number of layers and units per layer based on the complexity of the dynamics in the data that you are trying to capture. Additionally, you can specify training-related parameters like batch-size, learning rate, optimizer, number of epochs and also early stopping to prevent model overfitting. Once you have defined the parameters you can hit the 'Train!' button to start the training. This will print the final model architecture and start the training. For now, a fixed ratio of 8:1:1 for training, validation, and test datasets is used. Currently, if you want to adjust the ratio, you unfortunately have to dig into the code.<br/><br/>
<i>Note: The status messages about the training are written to the console and not passed on to the graphical user interface for now. If you want to observe the training progress, please check the terminal that you used to start the graphical user interface. Also, during the training, the user interface might get unresponsive. Do not worry, just wait until the training is finished.</i>
<p align="center">
<img src="https://github.com/SPARC-FAIR-Codeathon/2024-team-8/blob/main/img/sparcrl_train_model_2.png?raw=true" alt="Define model architecture and start the training of the surrogate model." height="500"/><br/>
      <b>Figure 3.</b> Define model architecture and set training parameters.
</p><br/>
Now you can sit back and watch Tensorflow doing its magic to train your surrogate model. The trained surrogate model is saved along with the training data into the `training_data` directory in your project folder.
<br/><br/>
<p align="center">
<img src="https://github.com/SPARC-FAIR-Codeathon/2024-team-8/blob/main/img/training_progress.png?raw=true" alt="Training of the model" width="800"/><br/>
        <b>Figure 3.</b> Observe training progress.
</p>

#### Reinforcement Learning on oSPARC:

After training the surrogate model, users can parameterize the RL process by selecting from a range of popular RL algorithms such as A2C, DDPG, DQN, HER, PPO, SAC, and TD3, along with their respective policies.
The tool supports detailed customization, including choosing the type of action space (discrete or continuous), specifying value ranges, and setting the number of actions for discrete spaces.
Advanced Training Customization:

Users can decide whether to use observation and/or action normalization, set batch sizes, and define the number of training steps to optimize performance.
SPARC.RL also allows tracking of a fixed setpoint or varying the setpoint within a specified range, adding an extra layer of flexibility to the training process.
SPARC.RL is designed to democratize the use of reinforcement learning, making it accessible to researchers across various domains. With its intuitive interface and powerful features, SPARC.RL is set to become an essential tool for those looking to harness the full potential of reinforcement learning in their research.
