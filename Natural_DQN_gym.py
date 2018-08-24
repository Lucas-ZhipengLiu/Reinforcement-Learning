#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 16:02:39 2018
@author: lucas

Implementing Natural Deep Q network (DQN)
On opensi gym CartPole task
"""

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch

from collections import namedtuple
import matplotlib.pyplot as plt
import random
import pickle
import copy
import gym


class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(ob_size, 256)
        self.fc2 = nn.Linear(256, ac_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
    
class DQN():
    
    def __init__(self):
        
        self.greedy_end = 0.1
        self.memory_size = memory_size
        self.anneal = 0.8 / self.memory_size
        
        self.memory = []
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
        
        self.dqnNet = Net()
        self.dqnNet.zero_grad()
        self.optimizor = optim.RMSprop(self.dqnNet.parameters(), lr=0.001)
        #Surprisingly, RMSprop+smooth_l1_loss is much better than SGD+MSELoss, especially the optimizor.
        
        self.targetNet = copy.deepcopy(self.dqnNet)
        self.targetNet.zero_grad()
        
        
    def store_transitions(self, ob, action, reward, ob2):
        
        if(step_count < self.memory_size):
            self.memory.append(None)
            self.memory[step_count] = self.Transition(ob, action, reward, ob2)
        else:
            position = step_count % self.memory_size
            self.memory[position] = self.Transition(ob, action, reward, ob2) 
            
            
    def step_action(self, ob, steps):
    
        a = random.random()
    
        # annealing greedy (0.9 - 0.1)
        if(steps > self.memory_size):
            greedy = greedy_end
        else:
            greedy = 0.9 - self.anneal * steps

        if(a < greedy):
            action = random.randint(0, ac_size - 1)          
        else:
            output = self.dqnNet(ob)
            _, index = torch.max(output, 0)
            action = index.item()

        return action
    
    
    def learn(self):
    
        input_state = torch.zeros(batch_size, ob_size)
        input_nextState = torch.zeros(batch_size, ob_size)
        reward = torch.zeros(batch_size, 1)
        target_Q = torch.zeros(batch_size, 1)
        current_Q = torch.zeros(batch_size, 1)
        nextState_max = torch.zeros(batch_size, 1)
    
        # randomly sampled from experience
        drawn = random.sample(range(self.memory_size), batch_size)
        
        for i in range(batch_size):
            input_state[i] = self.memory[drawn[i]][0]
            input_nextState[i] = self.memory[drawn[i]][3]
    
        # calculate current/target q (action) values
        output_state = self.dqnNet(input_state)
        output_nextState = self.targetNet(input_nextState)  
        output_nextState = output_nextState.detach()
    
        # get current state-action value and maximum target q_value
        for i in range(batch_size):
            a = self.memory[drawn[i]][1]
            current_Q[i] = output_state[i][a]
            
            reward[i] = self.memory[drawn[i]][2]
            _, index = torch.max(output_nextState[i], 0)
            nextState_max[i] = output_nextState[i][index]
            
        target_Q = reward + gamma * nextState_max
    
        # update neural network
        loss = F.smooth_l1_loss(current_Q, target_Q)
        self.optimizor.zero_grad()
        loss.backward()
        self.optimizor.step()
        
        
def main():
    
    global targetNet_step, step_count
    average_reward = 0
    
    env = gym.make('CartPole-v1')
    env = env.unwrapped
    dqn = DQN()
    
    for i in range(episode_count):#episode_count
        
        # for evaluation
        record.append(average_reward)
        print(i, ':', average_reward)
        average_reward = 0 
   
        ob = torch.tensor(env.reset(), dtype = torch.float)
       
        #while True: 
        for j in range(max_episodeSteps):   
           
            #env.render()
            action = dqn.step_action(ob, step_count)
            
            ob2_array, reward, done, _ = env.step(action) 
            ob2 = torch.tensor(ob2_array, dtype = torch.float)
            x, x_dot, theta, theta_dot = ob2
            r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
            reward = r1 + r2
            
            average_reward = average_reward + reward
            
            # store memory
            dqn.store_transitions(ob, action, reward, ob2)
            
            # train
            if(step_count>memory_size):
                if(step_count == memory_size + 1):
                    print('....learning_start....')
                dqn.learn()
                targetNet_step += 1
                if(targetNet_step>target_update):
                    dqn.targetNet = copy.deepcopy(dqn.dqnNet)
                    targetNet_step = 0
            
            ob = ob2
            step_count += 1
            if(done): break    
        
        
if __name__ == '__main__':
    
    ob_size = 4
    ac_size = 2

    # for deep Q network
    max_episodeSteps = 1000
    episode_count = 1000
    memory_size = 10000
    step_count = 0
    batch_size = 32
    greedy_end = 0.1
    gamma = 0.9
    target_update = 100
    targetNet_step = 0
  
    record = []
    
    main()
    
    plt.plot(record)
    file = open('DQN_gym.pickle','wb')
    pickle.dump(record, file)
    file.close()