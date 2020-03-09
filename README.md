# DQN_pytorch
DQN Implementation for Udacity Reinforcement Learning Nanodagree - Project One

* For the report, see: [Report.md](Report.md)
* For a blog post, see: [The Fairly Accessible Guide to the DQN Algorithm](https://medium.com/@Joebooth/the-fairly-accessible-guide-to-the-dqn-algorithm-d497565844b9?sk=90a9811a622c66663777d375b6eb817c)

## Install and Setup

1. Create the Conda environment and install packages

```
conda env create --file environment.yml
conda activate p1_navigation
pip install .
```

2. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)


## Play the Trained Agent
```
python play.py
```

## Retrain the Agent
```
python learn.py
```
    

    