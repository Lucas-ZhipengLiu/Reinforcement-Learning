# Reinforcement-Learning
Implemented various RL algorithms. Test and compare them on both openai gym task and my designed task.

## Implemented algorithms

- Q Learning
- Policy Gradient
- Actor-critic
- Naive DQN (Deep Q-network)
- Prioritized DQN (DQN + Prioritized experience replay)
- Categorical DQN (DQN + Distributional perspective)
- Naive DDPG (Deep Deterministic Policy Gradient)
- Prioritized DDPG (DDPG + Prioritized experience replay)


## Dependency

- Ubuntu 14.04, 16.04
- Python 3.6, 3.5
- Pytorch v0.4.1
- OpenAI Gym (Classic control)
- pybullet

## My task

It is basically a balance task for a 2-wheels robot. However, the robot not only need to keep balance, but also need to come down slope and climb slope. The simulation is based on pybullet. You can watch the video here.

![alt text](https://github.com/Lucas-ZhipengLiu/Reinforcement-Learning/blob/readme-edits/result%20images/9.gif)

## Results

Test and compare DQN, Prioritized DQN and Categorical DQN on OpenAI Gym CartPole-v1 task. 

![alt text](https://github.com/Lucas-ZhipengLiu/Reinforcement-Learning/blob/readme-edits/result%20images/6.png)

Test and compare DQN, DDPG, Prioritized DDPG on my task.

![alt text](https://github.com/Lucas-ZhipengLiu/Reinforcement-Learning/blob/readme-edits/result%20images/5.png)

## References
