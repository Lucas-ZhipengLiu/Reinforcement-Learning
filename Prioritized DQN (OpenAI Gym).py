#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 17:14:34 2018
@author: lucas

Implementing Priorited Experience Replay Deep Q network (PDQN)
On opensi gym CartPole task
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
        self.prob_alpha = 0.6
        self.beta = 0.4
        self.memory_size = memory_size
        self.anneal = 0.8 / self.memory_size
        
        self.memory = []
        self.priorities = np.zeros((self.memory_size), dtype = np.float32)
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
        
        self.dqnNet = Net()
        self.dqnNet.zero_grad()
        self.optimizor = optim.RMSprop(self.dqnNet.parameters(), lr=0.001)
        #Surprisingly, RMSprop+smooth_l1_loss is much better than SGD+MSELoss, especially the optimizor.
        
        self.targetNet = copy.deepcopy(self.dqnNet)
        self.targetNet.zero_grad()
        
        
    def store_transitions(self, ob, action, reward, ob2):
        
        # store maximum priority of each new sample
        max_prio = self.priorities.max() if self.memory else 1.0
        if(step_count < self.memory_size):
            self.memory.append(None)
            self.memory[step_count] = self.Transition(ob, action, reward, ob2)
            self.priorities[step_count] = max_prio
        else:
            position = step_count % self.memory_size
            self.memory[position] = self.Transition(ob, action, reward, ob2) 
            self.priorities[position] = max_prio
            
            
    def sample_transitions(self, beta):
        
        ob1 = torch.zeros(batch_size, ob_size)
        ob2 = torch.zeros(batch_size, ob_size)
        ac1 = torch.zeros(batch_size)
        rewards = torch.zeros(batch_size, 1)
        
        # compute probabilities of every sample
        probs = self.priorities**self.prob_alpha
        probs /= probs.sum()
        indices = np.random.choice(self.memory_size, batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        
        # compute weights of every batch sample
        batch_weights = (batch_size * probs[indices])**(beta) # w = (N*P(i))^(-β)
        batch_weights /= batch_weights.max()
        
        # unpack from samples
        for i in range(batch_size):
            
            ob1[i]     = samples[i][0]
            ac1[i]     = samples[i][1]
            rewards[i] = samples[i][2]
            ob2[i]     = samples[i][3]
        
        return ob1, ac1, rewards, ob2, indices, torch.tensor(batch_weights)
        
        
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
    
    
    def learn(self, beta):
    
        current_Q = torch.zeros(batch_size, 1)
        nextState_max = torch.zeros(batch_size, 1)
    
        # samples from memory
        ob1, ac1, rewards, ob2, indices, batch_weights = self.sample_transitions(beta)
        batch_weights = batch_weights.unsqueeze(-1)
    
        # calculate current/target q (action) values
        output_state = self.dqnNet(ob1)
        output_nextState = self.targetNet(ob2)  
        output_nextState = output_nextState.detach()
    
        # get current state-action value and maximum target q_value
        for i in range(batch_size):
            
            current_Q[i] = output_state[i][int(ac1[i].item())]           
            _, index = torch.max(output_nextState[i], 0)
            nextState_max[i] = output_nextState[i][index]
            
        target_Q = rewards + gamma * nextState_max
    
        #loss = F.smooth_l1_loss(current_Q, target_Q)
        new_loss = batch_weights * (current_Q - target_Q)**2
        loss = new_loss.mean()
        new_priorities = new_loss + 1e-5
        
        # update neural network
        self.optimizor.zero_grad()
        loss.backward()
        self.optimizor.step()
        
        # priorities update
        for i in range(batch_size):            
            self.priorities[indices[i]] = new_priorities[i]
        
        
def main():
    
    global targetNet_step, step_count
    average_reward = 0
    beta = beta_start
    frame_idx = 0
    
    env = gym.make('CartPole-v1')
    env = env.unwrapped
    dqn = DQN()
    
    for i in range(episode_size):
        
        # for evaluation
        record.append(average_reward)
        print(i, ':', average_reward)
        average_reward = 0 
   
        ob = torch.tensor(env.reset(), dtype = torch.float)
       
        #while True: 
        for j in range(max_episodeSteps):            
           
            #env.render()
            frame_idx += 1
            beta = min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)
            
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
                dqn.learn(beta)
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
    episode_size = 1000
    memory_size = 10000
    step_count = 0
    batch_size = 32
    greedy_end = 0.1
    gamma = 0.9
    target_update = 100
    targetNet_step = 0
    beta_frames = 20000 # back to uniform sampleing after 2e+4 steps
    beta_start = 0.4
  
    record = []
    
    main()
    
    plt.plot(record)
    file = open('PDQN_gym.pickle','wb')
    pickle.dump(record, file)
    file.close()
