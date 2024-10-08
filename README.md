<img src="./img/sparc_rl_logo_wb.png" style="display: block; width: 30%;"/><br/>

![Stable Baselines 3](https://img.shields.io/badge/Stable%20Baselines-v3.0.0-green)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29.1-red)
![SB3-Contrib](https://img.shields.io/badge/SB3--Contrib-2.3.0-violet)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange)

# Reinforcement Learning for Medical Device Control Made Easy
The control of medical devices, particularly in applications like neuromodulation, is increasingly focused on the development of closed-loop systems. These systems are gaining attention because they allow for real-time adjustments based on continuous feedback, leading to more precise and personalized treatments. However, developing the adaptive intelligence required for closed-loop control often demands specialized knowledge in sophisticated control theory, e.g. in reinforcement learning, posing a significant barrier for many researchers. 

**SPARC.RL** addresses this challenge by offering a proof of concept toolchain that simplifies the training of state-of-the-art reinforcement learning agents, even for those without deep expertise in the field. By leveraging the powerful [Stable Baselines3](https://stable-baselines3.readthedocs.io/) framework and seamlessly integrating data from the [SPARC](https://sparc.science/) platform with models from [oSPARC](https://osparc.io/), SPARC.RL democratizes access to advanced reinforcement learning techniques. This toolchain empowers researchers to explore and implement sophisticated control strategies, accelerating the development of more effective and personalized medical interventions through closed-loop systems.

To demonstrate how the SPARC.RL toolchain enables researchers, even those without specific domain knowledge in reinforcement learning, to effortlessly train state-of-the-art reinforcement learning (RL) agents for robust medical device control we have implemented a **pipeline that trains a closed-loop vagus nerve stimulation controller for heart rate control**. By allowing users to integrate other data from the SPARC platform and other models from oSPARC seamlessly into a reinforcement learning pipeline running on the oSPARC platform, SPARC.RL provides a framework that can be applied to arbitrary control problems in the biomedical field.

This toolchain and oSPARC pipeline were developed during the [2024 SPARC FAIR Codeathon](https://sparc.science/news-and-events/events/2024-sparc-fair-codeathon) by **Max Haberbusch** and **John Bentley**.

<b>The framework resulting from this codeathon can be accessed on [oSPARC](https://osparc.io/).</b>

<i>Note: While this project offers powerful capabilities, it is an initial prototype serving as a proof of concept and no guarantees are made regarding its absence of bugs and interoperability with datasets other than those used during development.</i>


## Key Features
The SPARC.RL toolchain consists of a stand-alone client that is used to obtain and preprocess appropriate datasets for reinforcement learning from the [SPARC](https://sparc.science/) platform and an oSPARC pipeline that allows the development of reinforcement learning-based control algorithms using Stable Baselines3 (Figure 1). This involves using a large language model (LLM) to present suggestions to the user on how to use the selected dataset in reinforcement learning. The stand-alone client helps to preprocess the data, design a proper neural network architecture, and train a surrogate deep neural network model. The resulting trained model is saved to hard disk. 

The second component is an [oSPARC](https://osparc.io/) application that enables the use of Stable Baselines3 on oSPARC (Figure 2). Here the user can load the trained surrogate model, select the reinforcement learning policy, and set the training hyperparameters. This final stage of the pipeline produces a trained control policy that can then be used in the closed-loop system.

Additionally, SPARC.RL provides a fully integrated reinforcement learning pipeline running on the oSPARC platform that also enables the training of surrogate models to efficiently represent system dynamics (Figure 2).

<br/>
<p align="center">
<img src="./img/toolchain_overview_wb.png" alt="Overview of the SPARC.RL toolchain." width="800"/>
<br/>
  <b>Figure 1.</b> Overview of the SPARC.RL toolchain.
</p>

<br/>
<p align="center">
<img src="./img/osparc_pipeline_animated.gif?raw=true" width="800"/>
<br/>
  <b>Figure 2.</b> Fully integrated SPARC.RL reinforcement pipeline on oSPARC.
</p>
<br/>

## Dataset and Model Integration

SPARC.RL supports the selection and use of time-series datasets directly loaded from the SPARC platform using the [SPARC Python client](https://docs.sparc.science/docs/sparc-python-client).
Users can also work with select [oSPARC](https://osparc.io/) models, enabling the training of RL agents in a highly flexible and customizable manner.
<br/><br/>
<i>Note: During development of our toolchain in the 2024 SPARC FAIR Codeathon, we used the Oliver Armitage et al. "Influence of vagus nerve stimulation on vagal and cardiac activity in freely moving pigs" dataset available on SPARC (DOI: [10.26275/aw2z-a49z](https://doi.org/10.26275/aw2z-a49z)).</i>

### Customizable Inputs and Outputs
The ultimate goal is to allow users to choose from multiple available model inputs (actions) and model outputs (observables) to tailor the reinforcement learning process to their specific needs. However, in the current version, this can only be done by modifying the source code. Later versions will allow the user to pick appropriate actions and observables directly from the graphical user interface.

### Data-driven Modelling

SPARC.RL offers multiple deep learning architectures to create surrogate models of experimental data available on [SPARC](https://sparc.science/) or [oSPARC](https://osparc.io) models.
Users can select from various RNNs optimized for time-series modeling, including vanilla Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM) networks, Bi-directional LSTM (BiLSTM) networks, and Gated Recurrent Units (GRUs), providing flexibility in how the models are trained. Users can configure their network and training parameters according to their needs. The adjustable parameters include the number of layers, number of units per layer, optimizer, learning rate, batch size, number of epochs, and early stopping policies.

### Steps to Generate the Surrogate Model

#### Installation 

##### Clone the Repository
```shell
git clone https://github.com/SPARC-FAIR-Codeathon/2024-team-8.git
cd 2024-team-8/sparcrl_surrogate
```

##### Create and Activate a Conda Environment
If you don't have Conda installed, you can install Miniconda or Anaconda as they come with Conda included.

Create and activate the environment using the provided `environment.yml` file:
```shell
conda env create -f environment.yml
conda activate sparcrl
```

##### Run the Surrogate Modeling Tool 
Now you are all set to run the surrogate modeling tool. To do so run the following command on your command line.
```shell
python sparcrl_surrogate.py
```

In the first step, select a dataset from the dropdown menu which is automatically populated with available datasets on the SPARC platform. Currently, the datasets are limited to time series data. Once you have selected a model, you can inspect the model metadata, including model description, creator, creation date, version, etc. Additionally, a large language model is used to generate suggestions on how to use the dataset for reinforcement learning. Once you have chosen your dataset, you can download and extract the data from SPARC by hitting the 'Get Dataset!' button. You will be asked for the file path to save the data. After that, you can proceed to the next step, which is to select the file(s) to use for training the surrogate model, by hitting the 'Next' button.

<br/>
<p align="center">
<img src="./img/choose_dataset.gif?raw=true" alt="Select dataset from SPARC platform" width="900"/>
<br/>
  <b>Figure 3.</b> Select dataset from SPARC platform to train surrogate model.
</p>

<br/>
Once you have chosen and downloaded an appropriate dataset, you can select one of the available files containing experimental data using the dropdown menu. The data is automatically filtered for appropriate file types. Currently, only the .hdf5 file format is supported. After you have selected a file, the data is pre-processed to bring it in the proper format for training the model. You can display the pre-processeed data by hitting the 'Plot Data!' button. If you are satisfied with the preprocessed data, you can move to the next step by hitting the 'Next' button.
<br/>

<br/>
<p align="center">
<img src="./img/chose_file.gif?raw=true" alt="Select data file from dataset and preprocess" width="1000"/>
<br/>
    <b>Figure 4.</b> Select a file from the dataset for model training and inspect preprocessed data.
</p>

<br/>
After loading the data for training, you can define your model architecture. For now, the tool allows you to use different types of recurrent neural networks including LSTM, Bi-LSTM, GRU and vanilla RNNs. You can adjust the number of layers and units per layer based on the complexity of the dynamics in the data that you are trying to capture. Additionally, you can specify training-related parameters like batch-size, learning rate, optimizer, number of epochs, and also early stopping to prevent model overfitting. Once you have defined the parameters, you can hit the 'Train!' button to start the training. This will print the final model architecture and start the training. For now, a fixed ratio of 8:1:1 for training, validation, and test datasets is used. Currently, if you want to adjust the ratio, you must modify the source code.
<br/>
<br/>
<i>Note: The status messages about the training are written to the console and not passed on to the graphical user interface. If you want to observe the training progress, please check the terminal that you used to start the graphical user interface. Also, during the training, the user interface might become unresponsive. Do not worry, just wait until the training is finished.</i>
<br/>

<br/>
<p align="center">
<img src="./img/model_training.gif?raw=true" alt="Define model architecture and start the training of the surrogate model." height="600"/>
<br/>
      <b>Figure 5.</b> Define model architecture and set training parameters.
</p>

<br/>
Now you can sit back and watch Tensorflow do its magic to train your surrogate model. The trained surrogate model is saved along with the training data into the `training_data` directory in your project folder.
<br/>

<br/>
<p align="center">
<img src="./img/training_progress.png" alt="Training of the model" width="800"/>
<br/>
        <b>Figure 6.</b> Observe training progress.
</p>

<br/>
After the training is completed, you can access the training data (.csv files) and the trained model (.h5 file) that was saved to your hard disk from your project directory.
<br/>

<br/>
<p align="center">
<img src="./img/sparcrl_data_saved_to_hd.png" alt="Data saved to hard disk." width="500"/>
<br/>
        <b>Figure 7.</b> Training data and trained model saved to hard disk.
</p>
<br/>

## Reinforcement Learning Using SPARC.RL on oSPARC

### Using SPARC.RL Train Surrogate Model Node on oSPARC
Training of the surrogate model can also be done on the oSPARC platform. However, this approach does not provide the ability to directly select data from the SPARC platform as can be done in the stand-alone client. The training can be run using the SPARC.RL Train Surrogate Model node. This node tries to approximate the underlying dynamical system based on the relationship of the inputs and outputs from the csv files (input.csv and output .csv) passed to the node. The SPARC.RL Train Surrogate Model node saves the trained deep neural network to a .h5 file (model.h5). 

The trained surrogate model then serves as an input to the SPARC.RL Train Agent node, which is used to train the reinforcement learning agent. The output of this node is a .zip file containing the trained reinforcement learning agent (ppo_cardiovascular.zip), which then can be used as a controller.

<br/>
<p align="center">
<img src="./img/osparc_nodes_overview.png" alt="Data saved to hard disk."/>
<br/>
        <b>Figure 8.</b> Overview of connected SPARC.RL nodes in a Study on oSPARC.
</p>

<br/>
Below, in Figure 9, you can see an example output of training a surrogate model using the SPARC.RL Train Surrogate Model node on a synthetic dataset generated with the model from [Haberbusch et al.](https://sparc.science/datasets/335). Here the stimulation intensity was varied from 0 mA to 5 mA in steps of 0.1 mA and the respective heart rate change was calculated. The model included only a single Long Short-Term Memory (LSTM) layer. The data was split into training, test, and validation sets with a ratio of 8:1:1.
<br/>

<br/>
<p align="center">
<img src="./img/osparc_surrogate_training.png" alt="Surrogate training on oSPARC using SPARC.RL Train Surrogate Model Node"/>
<br/>
        <b>Figure 9.</b> Surrogate training on oSPARC using SPARC.RL Train Surrogate Model node.
</p>

<br/>
The trained surrogate model showed very good performance in reproducing the dynamics of the full in silico model, as illustrated for one example from the test dataset shown below (Figure 10).
<br/>

<br/>
<p align="center">
<img src="./img/osparc_surrogate_training_result.png" alt="Surrogate model predictions compared to ground truth running in SPARC.RL Train Surrogate Model node on oSPARC."/>
<br/>
        <b>Figure 10.</b> Surrogate model predictions compared to ground truth running in SPARC.RL Train Surrogate Model node on oSPARC.
</p>

<br/>
After training the surrogate model, users can parameterize the RL process by selecting from a range of popular RL algorithms such as A2C, DDPG, DQN, HER, PPO, SAC, and TD3, along with their respective policies. The tool supports detailed customization, including choosing the type of action space (discrete or continuous), specifying value ranges, and setting the number of actions for discrete spaces.

### SPARC.RL Train Agent Node

#### Paramterize Reinforcement Learning
The SPARC.RL Train Agent node is designed to allow various aspects of the reinforcement learning setup and testing process to be parameterized. Environment settings, such as the choice between discrete or continuous action spaces and the number of parallel environments for training, can be adjusted. The path to the surrogate model and the specific heart rate targets used during testing are also configurable. PPO model parameters, including the policy type, number of training steps, batch size, and total timesteps, can be defined to optimize the training process. Additionally, testing parameters, such as the number of iterations and the interval for changing heart rate targets, can be customized. Finally, the paths for saving and loading trained models are configurable, enabling the script to be flexible and adaptable to different experimental needs.

The reinforcement learning can be adjusted to ones needs by modifying the rl_config.ini file that is used as input to the SPARC.RL Train agent node. As an example, the parameters used during the codeathon are provided below:

```
[Environment]
discrete = True
model_path = model.h5
env_id = 1337
parallel_envs = 6
target_hrs = 72.0, 74.0, 76.0, 78.0, 80.0, 82.0

[PPO]
policy = MlpPolicy
n_steps = 256
batch_size = 32
total_timesteps = 50000
tensorboard_log = ./tensorboard_logs/
tb_log_name = first_run

[Testing]
test_iterations = 1000
steps_per_target_change = 50
render_mode = human

[SavedModel]
save_path = ppo_cardiovascular
```

An example output of the training to the respective Jupyterlab notebook of the SPARC.RL Train Agent node is shown in Figure 11 below.

<br/>
<p align="center">
<img src="./img/osparc_train_rl_agent.png" alt="Running Proximal Policy Optimization (PPO) with the previously trained surrogate model in SPARC.RL Train Agent node on oSPARC."/>
<br/>
        <b>Figure 11.</b> Running Proximal Policy Optimization (PPO) with the previously trained surrogate model in SPARC.RL Train Agent node on oSPARC.
</p>

<br/>
The same output is visualized in Figure 12 below as a tensorboard depicting the loss function, policy gradient loss, and value loss for the training of the reinforcement learning algorithm over 50,000 total timesteps using the configuration specified in the rl_config.ini file shown above. It shows good convergence of the learning, suggesting proper controller performance.
<br/>

<br/>
<p align="center">
<img src="./img/sparcrl_agent_training_loss.png" alt="Training loss reinforcement learning."/>
<br/>
        <b>Figure 12.</b> Loss for reinforcement learning agent training.
</p>

<br/>
After completion of the training phase, the trained agent was tested on the surrogate model in a heart rate tracking task lasting for 1000 seconds. The agent had to track several randomly changing setpoint heart rates (changes occured every 50 seconds) from the setpoints specified in the .ini file above (72.0 bpm, 74.0 bpm, ..., 82.0 bpm). The controller showed very satisfactory performance in terms of steady state error quantified by the mean squared error between setpoint and measured heart rate of only 1.75 bpm (Figure 13).
<br/>

<br/>
<p align="center">
<img src="./img/osparc_controller_test_on_surrogate_model.png" alt="Testing trained agent on the surrogate model."/>
<br/>
        <b>Figure 13.</b> Testing the trained reinforcement learning agent on the surrogate model in SPARC.RL Train Agent node on oSPARC. Running 1000 seconds of heart rate tracking with random setpoint heart rates resulted in a steady state error quantified by mean squared error between setpoint and measured heart rate of only 1.75 bpm.
</p>
<br/>

### SPARC.RL Controller
The controller can be deployed with the full cardiovascular system model from Haberbusch et al. to form a complete control loop for evualation. It is [available on oSPARC](https://osparc.io/study/fa885632-4b04-11ee-869a-02420a0bd26e).

## License
This project is distributed under the terms of the [MIT License](./LICENSE).
