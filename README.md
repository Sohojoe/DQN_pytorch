# DQN_pytorch
DQN Implementation for Udacity Reinforcement Learning Nanodagree - Project One

* For the report, see: [Report.md](Report.md)
* For a blog post, see: [The Fairly Accessible Guide to the DQN Algorithm](https://medium.com/@Joebooth/the-fairly-accessible-guide-to-the-dqn-algorithm-d497565844b9?sk=90a9811a622c66663777d375b6eb817c)

## Getting Started

Instructions for installing dependencies or downloading needed files.

1. Clone this repository and create terminal / command prompt into the root of the repository

2. Create the Conda environment and install packages

```
conda env create --file environment.yml
conda activate p1_navigation
pip install .
```

3. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)


## Instructions

### Play the Trained Agent
```
python play.py
```

### Retrain the Agent
```
python learn.py
```

## Project Details

This project uses the Banana environment. A modified version of the ML-Agents [Banana Collector](https://github.com/Unity-Technologies/ml-agents/blob/0.4.0/docs/Learning-Environment-Examples.md#banana-collector) environment.

The environment has 37 scaler observations that contain the agent's velocity along with ray-based perception of objects around the agent's forward direction.

The environment provides the agent with a +1 reward for collecting yellow bananas and -1 when collecting a blue banana.

The environment provides four discrete actions:

 * 0 - Move Forward
 * 1 - Move Backwards
 * 2 - Move Left
 * 3 - Move Right

The environment is considered solved when the agent scores an average of 13 or more over 100 episodes.


    