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
The tool allows precise control over the parameters, including selecting the appropriate actions and observables that the RL agent should focus on.

#### Data-driven Modelling:

SPARC.RL offers multiple deep learning architectures to create surrogate models of experimental data or oSPARC models.
Users can select from various RNNs optimized for time-series modeling, including vanilla recurrent neural networks (RNNs), long short-term memory (LSTM) networks, bi-directional LSTM (BiLSTM) networks, and gated recurrent units (GRUs), providing flexibility in how the models are trained. Users can configure their network and training parameters according to their needs. The adjustable parameters include the number of layers, number of units per layer, optimizer, learning rate, batch size, number of epochs, and early stopping policies.

#### Reinforcement Learning Algorithm Configuration:

After training the surrogate model, users can parameterize the RL process by selecting from a range of popular RL algorithms such as A2C, DDPG, DQN, HER, PPO, SAC, and TD3, along with their respective policies.
The tool supports detailed customization, including choosing the type of action space (discrete or continuous), specifying value ranges, and setting the number of actions for discrete spaces.
Advanced Training Customization:

Users can decide whether to use observation and/or action normalization, set batch sizes, and define the number of training steps to optimize performance.
SPARC.RL also allows tracking of a fixed setpoint or varying the setpoint within a specified range, adding an extra layer of flexibility to the training process.
SPARC.RL is designed to democratize the use of reinforcement learning, making it accessible to researchers across various domains. With its intuitive interface and powerful features, SPARC.RL is set to become an essential tool for those looking to harness the full potential of reinforcement learning in their research.
