[//]: # (Image References)

[image1]: ./ReportImages/ActorModel.png "Actor Model"
[image2]: ./ReportImages/ResultGraph.png "ResultGraph"
[image3]: ./ReportImages/PsuedoCode.png "Psuedo Code"

# Project Report : Continuous Control
This project report is in relation with the second project in DRLND course - Continuous Control. In this environment, there are `20` double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm - 4 action space. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

## Learning Algorithm:
We make use of the Deep Deterministic Policy Gradients - DDPG - algorithm for this project.

As per DDPG which makes use of an Actor and a Critic NN, we use 2 DNN models. We use a similar network architechture for both of them:

**State --> BatchNorm --> 400 --> ReLU --> 300 --> ReLU --> tanh --> action**

Here is screenshot for our Actor Model:
![Actor Model][image1]

this above is for Actor, our critic varies only slightly.

We use the following hyperparameters:
1. Gamma - Discount Factor = 0.99
2. Learning rate for Actor = 5e-4
3. Learning rate for Critic = 1e-3

In addition, we also `SoftUpdates` with a `TAU = 1e-3` in order to calculate target values for both Actor and Critic

### Experience Replay:
With a `BUFFERSIZE = int(1e6)` and a `BATCHSIZE = 1024` we create a data container - the replay buffer. We batch from this random indepenent samples to stabally train the network.

Following a psuedo code for the DDPG algorithm used:
![Psuedo Code][image3]

## Plots of Rewards:
This has been quite a challanging project. Small updates and changes in the model resulted quite a lot of changes in the training result.
Initially without a random seed and batch normalization the model took way too much time to train. However, after adding a random seed and updating the initializer and adding batch normalization we can see that the model was solved in **`243 EPISODES`**.

Below is a graph of the reward as plotted agains episodes:
![Result Graph][image2]

## Ideas for Future Work:
This project implements the RelayBuffer but there is room for trying out other novel methods that were advised in the Project description as well:
1. For future work you could play around with:
    - [Proximal Policy Optimization Algorithm](https://arxiv.org/pdf/1707.06347.pdf)
    - [A3C](https://arxiv.org/pdf/1602.01783.pdf)
    - [Distributed Distributional Deep Deterministic Policy Gradient algorithm](https://openreview.net/pdf?id=SyZipzbCb)
2. I hardly played with the hyperparameters in this one, but I would want to experiment with the hyperparameteres to see how that affects learning and if I can improving the network.
3. Try out the crawler problem as well.