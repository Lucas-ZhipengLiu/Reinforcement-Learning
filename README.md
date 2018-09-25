# Reinforcement-Learning
**This project implements various reinforcement learning algorithms,
including state-of-art approaches DDPG and D4PG (incomplete), and use them
to solve a challenging balancing task. Moreover, a systematic comparison
among implemented algorithms is given. MSc dissertation is available [here](https://github.com/Lucas-ZhipengLiu/Reinforcement-Learning/blob/master/references/Liu-1772961.pdf).**

## Discription of implemented algorithms

- Q Learning
- Policy Gradient
- Actor-critic
- DQN (Deep Q-network)
- Prioritized DQN (DQN + Prioritized experience replay)
- Categorical DQN (DQN + Distributional perspective)
- DDPG (Deep Deterministic Policy Gradient)
- Incomplete D4PG (DDPG + Prioritized experience replay)

The first deep reinforcement learning algorithm, deep Q-network (DQN), was
proposed by DeepMind in 2015. It is based on a basic reinforcement learning
algorithm Q learning. Since then, reinforcement learning can be empowered by
deep neural network directly. One of the most significant achievements is that
reinforcement learning is capable of dealing with high-dimensional states like
raw image pixels. After that, the ideas underlying the success of deep Qnetwork
was adapted to the continuous action space, then we have the deep
deterministic policy gradient (DDPG) algorithm which is presented in 2017. It
relies on actor-critic architecture. Furthermore, distributional distributed deep
deterministic policy gradient (D4PG) algorithm, an extended version of DDPG,
was proposed in 2018. It adopts several very successful improvements (e.g.
distributional perspective) and works within a distributed framework. Below is a flowchart of thoes algorithms.

![alt text](https://github.com/Lucas-ZhipengLiu/Reinforcement-Learning/blob/master/result%20images/29a.PNG)

## Balancing task

It is a balance task for a 2-wheel robot. The robot needs to go down
a slope, climb another slope and keep the body balanced. It is a torque
control robot with two actuated joints on two wheels.  The simulation is powered by pybullet. You can watch the full video [here](https://youtu.be/oOzKpN154ng).

![alt text](https://github.com/Lucas-ZhipengLiu/Reinforcement-Learning/blob/master/result%20images/22a.png)

## Results

Test and compare **DQN**, **Prioritized DQN** and **Categorical DQN** on OpenAI Gym CartPole-v1 task. 

![alt text](https://github.com/Lucas-ZhipengLiu/Reinforcement-Learning/blob/master/result%20images/14.png)

Test and compare **DQN**, **DDPG**, **Incomplete D4PG** on the balancing task.

![alt text](https://github.com/Lucas-ZhipengLiu/Reinforcement-Learning/blob/master/result%20images/23.png)

## Usage and Dependencies

Each python file implements one reinforcement learning algorithm on either
Balancing task (i.e. the designed task) or CartPole task. There are 10 python
files totally including 8 RL algorithms. All implementations are from scratch.

- Ubuntu 14.04, 16.04
- Python 3.6, 3.5
- PyTorch v0.4.1
- OpenAI Gym (Classic control)
- PyBullet 2.1

## Main References

- [Playing Atari with Deep Reinforcement Learning](https://github.com/Lucas-ZhipengLiu/Reinforcement-Learning/blob/master/references/DQN.pdf)
- [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://github.com/Lucas-ZhipengLiu/Reinforcement-Learning/blob/master/references/Policy%20Gradient.pdf)
- [Deterministic Policy Gradient Algorithms](https://github.com/Lucas-ZhipengLiu/Reinforcement-Learning/blob/master/references/DPG.pdf)
- [CONTINUOUS CONTROL WITH DEEP REINFORCEMENTLEARNING](https://github.com/Lucas-ZhipengLiu/Reinforcement-Learning/blob/master/references/DDPG.pdf)
- [DISTRIBUTED PRIORITIZED EXPERIENCE REPLAY](https://github.com/Lucas-ZhipengLiu/Reinforcement-Learning/blob/master/references/d3pg.pdf)
- [DISTRIBUTED DISTRIBUTIONAL DETERMINISTIC POLICY GRADIENTS](https://github.com/Lucas-ZhipengLiu/Reinforcement-Learning/blob/master/references/D4PG.pdf)
- [PyBullet Quickstart Guide](https://github.com/Lucas-ZhipengLiu/Reinforcement-Learning/blob/master/references/pybullet%20quickstart%20guide.pdf)
- [DeepMind Control Suite](https://github.com/Lucas-ZhipengLiu/Reinforcement-Learning/blob/master/references/deepmind.pdf)
- [Deep Reinforcement Learning Hands-On](https://github.com/Lucas-ZhipengLiu/Reinforcement-Learning/blob/master/references/deep-reinforcement-learning-hands.pdf)
- [Morvan tutorials](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/)
