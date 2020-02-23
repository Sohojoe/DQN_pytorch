[RLLoop]: images/ReinforcementLearningLoop.png "Reinforcement Learning Definition"
[Gt]: images/(3.7)_Gt.png "Goal as sum of future rewards"
[Gt_discounted]: images/(3.8)_Gt_discount_reward.png "Goal as sum of future rewards discounted by gamma"
[vpiFormal]: images/(3.12)_State-Value_Function_Vpi.png "State-Value Function"
[qpiFormal]: images/(3.13)_Action-Value_Function_Qpi.png "Action-Value Function"
[vpi]: images/vpi.png "State-Value Function"
[qpi]: images/qpi.png "Action-Value Function"
[vpi(s)]: images/qpi(s,a).png "Value for policy pie, given state, S"
[qpi(s,a)]: images/qpi(s,a).png "Q value for policy pie, given the state action pair, S, and, A"
[St]: images/qpi(s,a).png "State S, at timestep t."
[At]: images/qpi(s,a).png "Action A, at timestep t."
[q-learningFormal]: images/(6.8)q-learningFormal.png

[DataTable3]: images/DataTable3.png "The effects of replay and separating the target Q-network"

# Udacity Reinforcement Learning Nanodagree - Project One
Joe Booth
Feb 15th, 2019

## Project Objective
The goal of this project is to solve the "Banana" environment using a Deep Q Network (DQN) [CITE], a reinforcement learning algorthem, using PyTorch and Python 3.


## Reinforcement Learning and the Q function

Reinforcement learning is used to address problems of sequencial decision making whereby an agent's action at one timestep influence future situations, or states. For example, each move in chess influences the future state of the chessboard. 

The agents goal in reinforcement learning is to maximize its future reward. For example, +1, -1, or 0 for a win, lose, or draw in chess.

### Markov Decision Process

Reinforcement learning uses a classical formulation of sequencial desicion making called a Markov Decision Process, or MDP. 

![alt text][RLLoop]

 - An MDP has descrete timesteps, denoted as a lowercase "t". 

 - The agent observs the environments state, S, at the current timestep, t, giving us, S t. 

 - The agent then performs an action, A t, on the environment. 

 - This in turm produces a new environment state at the next timestep, S t+1, along with a reward signal, R t+1.

The variables, S t, A t, S t+1, R t+1, make the core inputs into any reinforcment learning algorthem.

### Maximizing Future Rewards

The goal of the agent is to maximize the sum of all future rewards. That is, we may want the agent to learn to incur a small negative reward now if that increases the chance of greater reward in the future. 

The sum of all future reward, G, at timestep, t, is denoted as G t.

!["(G at timestep t, is equivilant to, R, at t+1, +, R, at t+2, + R, at t+3, +, etc)"][Gt]

(G at timestep t, is equivilant to, R, at t+1, +, R, at t+2, + R, at t+3, +, etc)


### Maximizing Discounted Future Rewards

In pratice we want to reduce the impact of rewards further in the future. This is done by introducing a discount rate, γ (gamma), where each step in the future is multiple by a incremental power of γ. Therefore:

!["(G at timestep t, is equivilant to, R, at t+1, +, (gamma, times R, at t+2), +, (gamma squared, times, R, at t+3), +, etc)"][Gt_discounted]

(G at timestep t, is equivilant to, R, at t+1, +, (gamma, times R, at t+2), +, (gamma squared, times, R, at t+3), +, etc)

### The State-Value Function

Now we can imagine that, over time, an agent can learn to predict the future value of each state of the MDP. This can be achived by making an estimate of the value at a given state, then calculating the actual value of that state at the end of the episode to produce a loss that we can apply to the estimate using gradient decent. We call this the State-Value function. 

The State-Value function is a powerful tool, because by learning the value of each state in the MDP, the agent does not have to do a nested tree search. At a given timestep, the agent can simply lookup the value of each state that each avaliable action would result in and select the action that results in the highest state-value.

The State-Value function is denoted as: ![vpi]

The formal definition for the State-Value Function is:

![vpiFormal]

(The Value for policy pie, given state, S, is equivilant to, the expected discounted return, if the agents starts at state, s, and then uses policy, pi, to choose actions for all future time steps)

### The Action-Value Function

There is one problem with the State-Value Function approach: the agent needs access to the dynamics of the MDP in order to calculate the next state produced by each action. For games like Chess or Go, this is trivral. However, for many real world situations, we do not have access to the dynamics.

What if, rather than learn the value of each state, the agent learns the value of every action at each state. Now the agent can simply choose the action with the highest value. This is called the Action-Value Function and is denoted as: ![qpi]

The formal definition for the Action-Value Function is

![qpiFormal]

(The q value for policy pie, given the state action pair, S, and, A, is equivilant to, the expected discounted return, if the agents starts at state, s, chooses action, a, and then uses policy, pi, to choose actions for all future time steps)

The Q-learning algorthem, (Watkins, 1989), was one of the early breakthroughs in reinforcement learning. Q-learning forms the bases for a family of algorthems in Reinforcement Learning and is the foundation for the DQN (Deep Q Network) algorthem.

Formally, the Q-learning algorthem is defined as:

![q-learningFormal]

(Let the Q value for the state-action pair, S and, A, at timestep t, be its current value, plus, the step size, alpha, multiplied by, the reward, R, at timestep, t+1, plus, gamma, multipled by, the maximum state-value, Q, of all actions, for the state, s, at timestep, t+1, minus, the state-value, Q, for the state-action pair, S and A, at timestep ,t)


## The DQN Algorithm
The DQN, (Deep Q Network), Algorthem [CITE] was successful in using a neural network as a nonlinear function approximator to represent the action value function to master the domain of Atari 2600 video games. 

As well as using a neural network, the DQN algorthem introduces two ideas that address the tendancy of nonlinear function approximotors to become unstable or diverge:

1) Experiance Replay: Observations are added to an experiance replay buffer which are then randomly sampled during the learning phase. The authors state that this removes correlations in the observation sequence and also smoothes over changes in the data distribution. This authors note that this idea is inspired by biology.

2) A second function approximator: To further imperove the stability of the algorthem, the authors introduced a seperate network for generating the targets in the Q-learning update. This second network, the **target network**, is copied from the **local network** every ```C``` updates where ```C``` is a hyperparamater which was set to 10,000 paramater update steps in the paper. The **target network** is kept static between these updates.


The authors demonstrate the impact of these two ideas on a subset of the Atari 2600 video games:

![alt text][DataTable3]

### Implementation of the DQN Algorithm

The implementation of the DQN algorthem I used has a few changes from the DQN Atari paper:

* Update to the target network: In the paper, the target network is held static for ```C```/```10,000``` paramater update steps. The implementation here uses a 'soft update' where the **target network** is moved in the direction of the **local network** each paramater update step (see below).

* Neural Network design: The observation space I use here is much simpler than the pixel inputs used in the atari paper, therefor I use a simpler neuro network which enabled faster training.

* Hyperparamaters: Again, because of the simpler nature of the Banana environment, I use different hyperparamaters to enable faster training. See below for the detail of what hyperparamaters I used.

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


### Code Organization

The code is organized into the following files

* ```dqn_agent.py```: The implementation of the DQN algorthem including the hyperparamaters
* ```model.py```: The model used for the neuralnetwork
* ```learn.py```: The python script used to learn the environment and save the trained model.
* ```play.py```: The python script used to learn the environment and save the trained model.

### Model Architecture 




### Hyperparameters





## The Banana Environment
The Banana environemnt is a modified version of the ML-Agents [Banana Collector](https://github.com/Unity-Technologies/ml-agents/blob/0.4.0/docs/Learning-Environment-Examples.md#banana-collector) environment.

The environment has 37 scaler observations which contain the agent's velocity along with ray-based perception of objects around the agent's forward direction.

The environment provides the agent with a +1 reward for colleting yellow bananas and -1 when collecting a blue banana.

The envionment provides four descrete actions

 * 0 - Move Forward
 * 1 - Move Backwards
 * 2 - Move Left
 * 3 - Move Right



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


### The Markov Property

The state is said the have the Markov Property if the state includes all information about all aspects of the past agent–environment interaction that make a difference for the future. For example, in chess, at any timestep the chessboard reveals the full state of the game, where as in Poker, some cards are hidden. Therefore, chess has the Markov Property, whereas, Poker does not.

state has the Markov Property because it represents all the history of that game in terms of influencing future moves.


 (MDP) is defined by:
 a set of states, S
 a set of actions, A
 a set of rewards, R
 one-step dynamics of the environment
 a discount factor of future rewards.

