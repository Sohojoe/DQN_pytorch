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
[plot_of_rewards]: images/plot_of_rewards.png
[banana_good]: images/banana_good.gif
[banana_stuck]: images/banana_stuck.gif

# Udacity Reinforcement Learning Nanodagree - Project One
Joe Booth Febuary 2019

## Project Objective
The goal of this project is to solve the "Banana" environment using a Deep Q Network (DQN) (Mnih et al., 2015), a reinforcement learning algorithm, using PyTorch and Python 3.

Note: This report aims to be accessible via text to speech browser plugins. To that end, I have phonetically typed out the 'alt-text' tag under equations (typically in brackets). I also use spaces between letters so that the text to speech plugin correctly pronounces the phrase. For example, I denote state at timestep t, as, 's t', as opposed to, 'st'

This report covers the following areas:

* An overview of Reinforcement Learning and the Q-learning algorithm. 
* The DQN Algorithm
* The implementation of the DQN Algorithm, including model architecture and hyperparameters.
* The Environment
* Results
* Ideas for Future Work


## Reinforcement Learning and the Q function

Reinforcement learning is used to address problems of sequential decision making whereby an agent's action at one timestep influence future situations, or states. For example, each move in chess influences the future state of the chessboard.

The agent's goal in reinforcement learning is to maximize its future reward. For example, +1, -1, or 0 for a win, lose or draw in chess.

### Markov Decision Process

Reinforcement learning uses a classical formulation of sequential decision making called a Markov Decision Process, or MDP.

![alt text][RLLoop]

 - An MDP has discrete timesteps, denoted as a lowercase "t".

 - The agent observes the environments state, S, at the current timestep, t, giving us, S t.

 - The agent then performs the action, A t, on the environment.

 - This, in turn, produces a new environment state at the next timestep, S t+1, along with a reward signal, R t+1.

The variables, S t, A t, S t+1, R t+1, make the core inputs into any reinforcement learning algorithm.

### Maximizing Future Rewards

The goal of the agent is to maximize the sum of all future rewards. That is, we may want the agent to learn to incur a small negative reward now if that increases the chance of greater reward in the future.

The sum of all future reward, G, at time step, t, is denoted as G t.

!["(G at timestep t, is equivalent to, R, at t+1, +, R, at t+2, + R, at t+3, +, etc)"][Gt]

(G at timestep t, is equivalent to, R, at t+1, +, R, at t+2, + R, at t+3, +, etc)


### Maximizing Discounted Future Rewards

In practice, we want to reduce the impact of rewards further in the future. We achieve this by introducing a discount rate, γ (gamma), where each step in the future is multiplied by an incremental power of γ. Therefore:

!["(G at timestep t, is equivalent to, R, at t+1, +, (gamma, times R, at t+2), +, (gamma squared, times, R, at t+3), +, etc.)"][Gt_discounted]

(G at timestep t, is equivalent to, R, at t+1, +, (gamma, times R, at t+2), +, (gamma squared, times, R, at t+3), +, etc.)

### The State-Value Function

Now we can imagine that, over time, an agent can learn to predict the future value of each state of the MDP. This can be achieved by making an estimate of the value at a given state, then calculating the actual value of that state at the end of the episode to produce a loss that we can apply to the estimate using gradient descent. We call this the State-Value function.

The State-Value function is a powerful tool because by learning the value of each state in the MDP, the agent does not have to do a nested tree search. At a given timestep, the agent can simply look up the value of each state that each available action would result in and select the action that results in the highest state-value.

The State-Value function is denoted as: ![vpi] (lowercase v, subscript pi)

The formal definition for the State-Value Function is:

![vpiFormal]

(The Value for policy pie, given state, S, is equivalent to, the expected discounted return, if the agents starts at state, s, and then uses policy, pi, to choose actions for all future time steps)

### The Action-Value Function

There is one problem with the State-Value Function approach: the agent needs access to the dynamics of the MDP in order to calculate the next state produced by each action. For games like Chess or Go, this is trivial. However, for many real-world situations, we do not have access to the dynamics.

What if, rather than learn the value of each state, the agent learns the value of every action at each state. Now the agent can simply choose the action with the highest value. This is called the Action-Value Function and is denoted as: ![qpi] (lowercase q, subscript pi)

The formal definition for the Action-Value Function is:

![qpiFormal]

(The q value for policy pie, given the state-action pair, S, and, A, is equivalent to, the expected discounted return, if the agents starts at state, s, chooses the action, a, and then uses policy, pi, to choose actions for all future time steps)

The Q-learning algorithm, (Watkins, 1989), was one of the early breakthroughs in reinforcement learning. Q-learning forms the bases for a family of algorithms in Reinforcement Learning and is the foundation for the DQN (Deep Q Network) algorithm.

Formally, the Q-learning algorithm is defined as:

![q-learningFormal]

(Let the Q value for the state-action pair, S and, A, at timestep t, be its current value, plus, the step size, alpha, multiplied by, the reward, R, at timestep, t+1, plus, gamma, multiplied by, the maximum state-value, Q, of all actions, for the state, s, at timestep, t+1, minus, the state-value, Q, for the state-action pair, S and A, at timestep, t)


## The DQN Algorithm
The DQN (Deep Q Network), algorithm (Mnih et al., 2015) was successful in using a neural network as a nonlinear function approximator to represent the action-value function to master the domain of Atari 2600 video games.

As well as using a neural network, the DQN algorithm introduces two ideas that address the tendency of nonlinear function approximators to become unstable or diverge:

1) Experience Replay: Observations are added to an experience replay buffer, which are then randomly sampled during the learning phase. The authors state that this removes correlations in the observation sequence and also smoothes over changes in the data distribution. The authors note that this idea was inspired by biology.

2) A second function approximator: To further improve the stability of the algorithm, the authors introduced a separate network for generating the targets in the Q-learning update. This second network, the **target network**, is copied from the **local network** every ```C``` updates where ```C``` is a hyperparameter, which was set to 10,000 parameter update steps in the paper.  The **target network** is kept static between these updates.


The authors demonstrate the impact of these two ideas on a subset of the Atari 2600 video games:

![alt text][DataTable3]

### Implementation of the DQN Algorithm

The implementation of the DQN algorithm I used has a few changes from the DQN Atari paper:

* Update to the target network: In the paper, the target network is held static for ```C```/```10,000``` parameter update steps. The implementation here uses a 'soft update' where the **target network** is moved in the direction of the **local network** each parameter update step (see below).

* Neural Network design: The observation space I use here is much simpler than the pixel inputs used in the atari paper. Therefore I use a simpler neuro network, which enabled faster training.

* Hyperparameters: Again, because of the more straightforward nature of the Banana environment, I use different hyperparameters to enable faster training. See below for the detail of what hyperparameters I used.

### Update Step Phases

During the update step, the algorithm has an action phase and a learning phase. 

#### Action Phase
In the action phase, the algorithm samples actions using the current ```state``` of the environment.

To sample an action, the algorithm uses a neural network to estimate the action values for a given state.

The algorithm uses Epsilon Greedy to determine whether to take a random action instead of a sampled action.

The action is passed to the environment's step function, which returns:

* the new state ```next_state```
* the reward value ```reward```
* and a termination flag ```done```. 

Finally, the algorithm uses an Experience Replay Buffer to store the ```state```, ```action```,```reward```, ```next_state```, and ```done``` flag.

#### Learning Phase
In the learning phase, the algorithm randomly samples the experience replay buffer to gather a set of experiences. It updates the neural network via backpropagation using the loss between the ```expected_Q``` values and the ```target_Q``` values.

The ```expected_Q``` values are calculated using the **local network**, ```qnetwork_local```, with the given ```states``` and selected ```actions``` from the experience replay sample set.

The ```target_Q``` values are calculated using the formula ```rewards + (gamma * max_predicted_Q * (1-dones))``` where

 * ```rewards``` and ```dones``` - are from the experiance replay sample set.
 * ```gamma``` - is the discount factor hyperparameter.
 * ```max_predicted_Q``` - is the maxium Q value of the ```next_state``` as estimated by the **target network**, ```qnetwork_target```.

A the end of the learning phase, a *'soft update'* is used to move the target network closer to the local network using the following: ```qnetwork_target = TAU * qnetwork_local + (1.0-TAU) * qnetwork_target``` where:

 * ```qnetwork_target``` is the target network
 * ```qnetwork_local``` is the local network
 * ```TAU``` is amount to move the target network towards the local network.


### Code Organization

The code is organized into the following files

* ```dqn_agent.py```: The implementation of the DQN algorithm including the hyperparameters
* ```model.py```: The model used for the neural network
* ```learn.py```: The python script used to learn the environment and save the trained model.
* ```play.py```: The python script used to play back the trained model.

### Model Architecture 

The model architecture uses three network layers, an input layer, a hidden layer, and an output layer. The layers have the following size and activation units.

```
input layer = 37 (number of observations) with relu activation
hidden layer = 32 with relu activation
output layer = 4 (number of actions)
```

### Hyperparameters

```
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 1e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
```


## The Banana Environment
The Banana environment is a modified version of the ML-Agents [Banana Collector](https://github.com/Unity-Technologies/ml-agents/blob/0.4.0/docs/Learning-Environment-Examples.md#banana-collector) environment.

The environment has 37 scaler observations that contain the agent's velocity along with ray-based perception of objects around the agent's forward direction.

The environment provides the agent with a +1 reward for collecting yellow bananas and -1 when collecting a blue banana.

The environment provides four discrete actions:

 * 0 - Move Forward
 * 1 - Move Backwards
 * 2 - Move Left
 * 3 - Move Right

The environment is considered solved when the agent scores an average of 13 or more over 100 episodes.

## Results

I was able to achieve an average score of 13+ over 100 episodes by tuning the learning rate (from 5e-4 to 1e-4). I was able to solve the environment in around 500 episodes. 

![plot_of_rewards] 

Blue = score per episode. 
Green = average score over past 100 episodes

Here is a GIF of the agent scoring 14 points in an episode:

![banana_good] 

Here is a GIF of the agent getting stuck in a loop:

![banana_stuck] 


I tried various strategies to improve the score but was not able to make a significant improvement. The strategies I tried include:

1. Grid search over Hyperparameters
2. Implementing DDQN (Hasselt et al., ‎2015) - I was able to see improvements against the Gym CartPole environment, but it scored less with the Banana environment.
3. Implementing a simplified version of Prioritized Experience Replay (Schaul et al., ‎2015) - Again, I was able to see improvements against the Gym CartPole environment, but not with the Banana environment.

## Ideas for Future Work

I see various ways of improving the agents' performance:

### Algorithmic Improvements

1. Adding an LSTM - Sometimes, the agent gets stuck into an endless loop whereby it goes left for a few frames, then right for a few frames. An LSTM could help the agent to avoid that state.
 
2. Additional DQN improvements - There continue to be algorithmic improvements to the original DQN algorithm. It would be interesting to try Rainbow (Hasselt et al., ‎2018), which combines many improvements into one algorithm.

### Using Background Knowledge of the Environment

3. Reduce Number of Actions - The trained agent only uses 3 actions (left, right, forward). Training with a reduced number of actions should improve performance.

4. Visual Pre-Processing - As well as training from pixels, the environment seems like a good candidate for pre-processing. For example, rather than 3 RGB layers, we could pick out the yellow in the good bananas for one layer, the blue for bad bananas in another layer, and the gray of the wall for a third layer. This should help improve training time over RGB layers.

## Optional Goals

The Optional Goals I Completed Are:

* Include a GIF and/or link to a YouTube video of your trained agent (see above)
* Solve the environment in fewer than 1800 episodes! (solved in ~500)
* Write a blog post explaining the project and your implementation (see: [The Fairly Accessible Guide to the DQN Algorithm](https://medium.com/@Joebooth/the-fairly-accessible-guide-to-the-dqn-algorithm-d497565844b9?sk=90a9811a622c66663777d375b6eb817c))
* Implement a double DQN, a dueling DQN, and/or prioritized experience replay (see results)

