from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch
import random
from sys import platform

if platform == "linux" or platform == "linux2":
    # linux
    env = UnityEnvironment(file_name="Banana.app")
elif platform == "darwin":
    # OS X
    env = UnityEnvironment(file_name="Banana.app")
else: # elif platform == "win32":
    # Windows...
    env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe",no_graphics=True)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# 2. Examine the State and Action Spaces

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)


from dqn_agent import Agent

seed = 0
# seed = 1
# seed = 2
# seed = 3
# seed = 4

env_info = env.reset(train_mode=True)[brain_name]
agent = Agent(state_size=state_size, action_size=action_size, seed=seed)


def dqn(n_episodes=1000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    window_average = []
    eps = eps_start                    # initialize epsilon
    high_score = 5
    has_new_high_score = False
    high_scores_window = deque(maxlen=10)
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)                 # select an action
            action = action.astype(int)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        window_average.append(np.mean(scores_window))
        high_scores_window.append(score)  # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.2f}'.format(i_episode, np.mean(scores_window), eps), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.2f}'.format(i_episode, np.mean(scores_window), eps))
            if has_new_high_score:
                print('\rNew High Score: {:.2f}'.format(high_score))
                has_new_high_score = False
        if np.mean(high_scores_window)>high_score:
            has_new_high_score = True
            high_score = np.mean(high_scores_window)
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
    print('\rHigh Score: {:.2f}'.format(high_score))
    return scores, window_average

scores, window_average = dqn()
env.close()

buff_13 = [13 for i in range(len(scores))]
# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.plot(np.arange(len(buff_13)), buff_13)
plt.plot(np.arange(len(window_average)), window_average)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
            