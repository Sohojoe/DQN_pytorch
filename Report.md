[DataTable3]: images/DataTable3.png "The effects of replay and separating the target Q-network"

# Udacity Reinforcement Learning Nanodagree - Project One
Joe Booth
Feb 15th, 2019

## Project Objective
The goal of this project is to solve the "Banana" environment using a Deep Q Network (DQN) [CITE], a reinforcement learning algorthem, using PyTorch and Python 3.


## The Banana Environment
The Banana environemnt is a modified version of the ML-Agents [Banana Collector](https://github.com/Unity-Technologies/ml-agents/blob/0.4.0/docs/Learning-Environment-Examples.md#banana-collector) environment.

The environment has 37 scaler observations which contain the agent's velocity along with ray-based perception of objects around the agent's forward direction.

The environment provides the agent with a +1 reward for colleting yellow bananas and -1 when collecting a blue banana.

The envionment provides four descrete actions

 * 0 - Move Forward
 * 1 - Move Backwards
 * 2 - Move Left
 * 3 - Move Right


## The DQN Algorithm
The alogrothem used is based on the DQN Algorthem [CITE] which was successful in using a neural network as a nonlinear function approximator to represent the action value function to master the domain of Atari 2600 video games. 

As well as using a neural network, the DQN algorthem introduces two ideas that address the tendancy of nonlinear function aspproximotors to become unstable or diverge:

1) Experiance Replay: Observations are added to an experiance replay buffer which are then randomly sampled during the learning phase. The authors state that this removes correlations in the observation sequence and also smoothes over changes in the data distribution. This authors note that this idea is inspired by biology.

2) A second function approximator: To further imperove the stability of the algorthem, the authors introduced a seperate network for generating the targets in the Q-learning update. This second network, the **target network**, is copied from the **local network** every ```C``` updates where ```C``` is a hyperparamater which was set to 10,000 paramater update steps in the paper. The **target network** is kept static between these updates.


The authors demonstrate the impact of these two ideas on a subset of the Atari 2600 video games:

![alt text][DataTable3]

### My Implementation of the DQN Algorithm

The implementation of the DQN algorthem I used has a few changes from the DQN Atari paper:

* Update to the target network: In the paper, the target network is held static for ```C```/```10,000``` paramater update steps. The implementation here uses a 'soft update' where the **target network** is moved in the direction of the **local network** each paramater update step (see below).

* Neural Network design: The observation space I use here is much simpler than the pixel inputs used in the atari paper, therefor I use a simpler neuro network which enabled faster training.

* Hyperparamaters: Again, because of the simpler nature of the Banana environment, I use different hyperparamaters to enable faster training. See below for the detail of what hyperparamaters I used.

### Code Organization

The code is organized into the following files

* ```dqn_agent.py```: The implementation of the DQN algorthem including the hyperparamaters
* ```model.py```: The model used for the neuralnetwork
* ```learn.py```: The python script used to learn the environment and save the trained model.
* ```play.py```: The python script used to learn the environment and save the trained model.



### Learning Phases

The algorthem has an action phase and a learning the phase. 

#### Action Phase
In the action phase, the algorthem samples actions using the current ```state``` of the environment. 

To sample an action, the algorthem uses an neural network to estimate the action values for a given state. 

The algorthem uses Epslion Greedy to determin weather to take a random action instead of a sampled action. 

The action is passed to the environment's step function which returns:

* the new state ```next_state```
* the reward value ```reward```
* and a termination flag ```done```. 

Finally, the algorthem uses an Experiance Replay Buffer to store the ```state```, ```reward```, ```next_state```, and ```done``` flag.

#### Learning Phase
In the learning phase, the algorthem randomly samples the experiance replay buffer to gather a set of experiances. It updates the neural network via backpropigation using the loos between the values ```expected_Q``` and the ```target_Q``` values.

The ```expected_Q``` values are calculated using the **local** network, ```qnetwork_local```, with the given ```states``` and selected ```actions``` from the experiance replay sample set

The ```target_Q``` values are calculated using the formula ```rewards + (gamma * max_predicted_Q * (1-dones))``` where

 * ```rewards``` and ```dones``` - are from the experiance replay sample set.
 * ```gamma``` - is the discount factor hyperparameter.
 * ```max_predicted_Q``` - is the maxium Q value of the ```next_state``` as estimated by the **target** network, ```qnetwork_target```.

 A the end of the learning phase a *'soft update'* is used to move the target network closer to the local network using the following: ```qnetwork_target = tau*qnetwork_local + (1.0-TAU)*qnetwork_target``` where:

 * ```qnetwork_target``` is the target network
 * ```qnetwork_local``` is the local network
 * ```TAU``` is amount to move the target network towards the local network.


## Model Architecture 


## Hyperparameters





## Results


TODO

* [ ] Clearly describe the learning algorithm 
* [ ] chosen hyperparameters
* [ ] describes the model architectures for any neural networks.
* [ ] plot of rewards per episode - illustrate that the agent is able to receive an average reward (over 100 episodes) of at least +13
* [ ] reports the number of episodes needed to solve the environment
* [ ] Ideas for Future Work - concrete future ideas for improving the agent's performance.

Optional Goals
* [ ] Include a GIF and/or link to a YouTube video of your trained agent!
* [ ] Solve the environment in fewer than 1800 episodes!
* [ ] Write a blog post explaining the project and your implementation!
* [ ] Implement a double DQN, a dueling DQN, and/or prioritized experience replay!
* [ ] For an extra challenge after passing this project, try to train an agent from raw pixels! Check out (Optional) Challenge: Learning from Pixels in the classroom for more details.
