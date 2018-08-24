#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 17:34:14 2018
@author: Lucas Liu

Implementing Distributional Deep Q network (DDQN)
On openai gym Cartpole model
"""

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch

from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import copy
import gym


class Net(nn.Module): # modified for distributional perspective  
    
    def __init__(self):
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(ob_size, 256)
        self.fc2 = nn.Linear(256, ac_size*n_atoms)
           
    def forward(self, x):
        if (x.size()[0]==4):
            miniBatch = 1
        else:
            miniBatch = batch_size
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x.view(miniBatch, ac_size, n_atoms)
    

class DDQN():
    
    def __init__(self):
        
        self.greedy_end = 0.1
        self.memory_size = 10000
        self.anneal = 0.8 / self.memory_size
        
        self.memory = []
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
        
        self.ddqnNet = Net()
        self.ddqnNet.zero_grad()
        self.optimizor = optim.RMSprop(self.ddqnNet.parameters(), lr=0.001)
        
        self.targetNet = copy.deepcopy(self.ddqnNet)
        self.targetNet.zero_grad()
        
        
    def store_transitions(self, ob, action, reward, ob2):
        
        if(step_count < self.memory_size):
            self.memory.append(None)
            self.memory[step_count] = self.Transition(ob, action, reward, ob2)
        else:
            #re = torch.tensor([self.memory[i][2] for i in range(len(self.memory))])
            position = step_count % self.memory_size
            self.memory[position] = self.Transition(ob, action, reward, ob2) 
    
    # function created for distributional perspective
    def targetDistri_2_originDistri(self, distribution2, rewards, Vmin, Vmax, n_atoms, gamma):
    
        distribution2 = distribution2.numpy()
        rewards = rewards.numpy()
    
        proj_distr = np.zeros((batch_size, n_atoms), dtype=np.float32)
        delta      = (Vmax - Vmin) / (n_atoms - 1)
    
        for atom in range(n_atoms):
        
            r_proj       = np.minimum(Vmax, np.maximum(Vmin, rewards + gamma * (Vmin + atom * delta)))
            target_index = (r_proj - Vmin) / delta
            below        = np.floor(target_index).astype(np.int64)
            above        = np.ceil(target_index).astype(np.int64)
        
            for i in range(batch_size):
            
                if (above[i] != below[i]):
                    proj_distr[i][below[i]] += distribution2[i][atom] * (above[i] - target_index[i])
                    proj_distr[i][above[i]] += distribution2[i][atom] * (target_index[i] - below[i])
                else:
                    proj_distr[i][below[i]] += distribution2[i][atom]
                
        return torch.tensor(proj_distr, dtype=torch.float)
    
    # function created for distributional perspective
    def action_select(self, action_distr):

        # transfer distribution to action value
        action_probs = F.softmax(action_distr.view(-1, n_atoms))
        action_value = action_probs * distribution
        _, index = torch.max(action_value.sum(dim=1), 0)
    
        return index.item()
    
    
    def step_action(self, ob, steps):
    
        a = random.random()
    
        # annealing greedy (0.9 - 0.1)
        if(steps > self.memory_size):
            greedy = self.greedy_end
        else:
            greedy = 0.9 - self.anneal * steps

        if(a < greedy):
            action = random.randint(0, ac_size - 1)          
        else:
            output = self.ddqnNet(ob)
            action = self.action_select(output) # modified for distributional perspective
          
        return action
    
    
    def learn(self):
        
        input_state = torch.zeros(batch_size, ob_size)
        input_nextState = torch.zeros(batch_size, ob_size)
        rewards = torch.zeros(batch_size, 1)
        target_Q = torch.zeros(batch_size, n_atoms)
        current_Q = torch.zeros(batch_size, n_atoms)
        nextState_max = torch.zeros(batch_size, n_atoms)       
        
        drawn = random.sample(range(self.memory_size), batch_size)
        
        for i in range(batch_size):
            input_state[i] = self.memory[drawn[i]][0]
            input_nextState[i] = self.memory[drawn[i]][3]
                       
        output_state = self.ddqnNet(input_state)
        output_nextState = self.targetNet(input_nextState) 
        output_nextState = output_nextState.detach()
        
        for i in range(batch_size):
        
            rewards[i] = self.memory[drawn[i]][2]
            
            # modified for distributional perspective
            index = self.action_select(output_nextState[i])
            nextState_max[i] = output_nextState[i][index]
        
            a = self.memory[drawn[i]][1]
            current_Q[i] = output_state[i][a]
        
        # modified for distributional perspective    
        nextState_max = F.softmax(nextState_max, dim=1)
        current_Q     = F.log_softmax(current_Q, dim=1)
        
        # modified for distributional perspective    
        target_Q = self.targetDistri_2_originDistri(nextState_max, rewards, Vmin, Vmax, n_atoms, gamma)
        
        # modified for distributional perspective    
        loss = - target_Q * current_Q
        loss = loss.sum(dim=1).mean()
        self.optimizor.zero_grad()
        loss.backward()
        self.optimizor.step()
        
        
def main(env, ddqn):
    
    global targetNet_step, step_count
    average_reward = 0
    
    for i in range(episode_maximum):
        
        
        record.append(average_reward)
        print(i, ':', average_reward)
        average_reward = 0 
    
        ob = torch.tensor(env.reset(), dtype = torch.float)
        
        survival_steps = 0
       
        for j in range(max_episodeSteps):          
           
            #env.render()
            action = ddqn.step_action(ob, step_count)
            
            ob2_array, reward, done, _ = env.step(action) 
            ob2 = torch.tensor(ob2_array, dtype = torch.float)
            
            x, x_dot, theta, theta_dot = ob2
            r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
            reward = r1 + r2
            
            average_reward = average_reward + reward
            
            ddqn.store_transitions(ob, action, reward, ob2)
            
            if(step_count>ddqn.memory_size):
                if(step_count == ddqn.memory_size+1):
                    print('....learning_start....')
                ddqn.learn()
                targetNet_step += 1
                if(targetNet_step>target_update):
                    ddqn.targetNet = copy.deepcopy(ddqn.ddqnNet)
                    targetNet_step = 0
            
            ob = ob2
            step_count += 1
            survival_steps += 1
            if(done): break    

    
if __name__ == '__main__':
    
    ob_size = 4
    ac_size = 2

    # for deep Q network
    episode_maximum = 1000
    max_episodeSteps = 1000
    batch_size = 32
    step_count = 0
    target_update = 100
    targetNet_step = 0
    gamma = 0.9
    flag = False

    # distributional hyperparameters
    Vmax = 10
    Vmin = -10
    n_atoms = 51
    delta = (Vmax - Vmin) / (n_atoms - 1)
    distribution = torch.arange(Vmin, Vmax+delta, delta)
    
    
    env = gym.make('CartPole-v1')
    env = env.unwrapped
    
    ddqn = DDQN()
    
    record = []
    
    main(env, ddqn)
    
    plt.plot(record)
    file = open('DDQN_gym.pickle','wb')
    pickle.dump(record, file)
    file.close()
    
    
    
    
