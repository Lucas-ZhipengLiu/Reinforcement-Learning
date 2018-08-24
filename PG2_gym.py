#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 23:29:20 2018

@author: lucas
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from collections import namedtuple
from time import sleep
import random
import numpy as np
import matplotlib.pyplot as plt



ob_num = 4
ac_num = 2

episodes_count = 10000

total_steps = 0


class Net(nn.Module):
    
    def __init__(self):
        
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(ob_num, 128)
        self.fc2 = nn.Linear(128, ac_num)
        
    def forward(self, x):
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # no softmax layer here as we will use softmax & log_softmax later
        return x


class Policy():
    
    def __init__(self):
        
        self.policyNet = Net()
        self.memory = []
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward'))
        
        self.policyNet.zero_grad()
        self.optimizer = optim.Adam(self.policyNet.parameters(), lr=0.01)
        
        
    def step_action(self, ob):
        
        output = self.policyNet(ob)
        output = F.softmax(output, dim=1)
        action = random.choices([0, 1], output[0], k=1) # choice with probability

        return action[0]
    

    def step_reward(self):
    
        global ep_rs
        ep_rs = []
    
        ep_rs = [i[2] for i in self.memory]
    
        #print(sum(ep_rs))
        
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(ep_rs)
        running_add = 0
        for t in reversed(range(0, len(ep_rs))):
            running_add = running_add * 0.95 + ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
    
        return discounted_ep_rs


    def memory_store(self, ob, action, reward, position):
    
        self.memory.append(None)
        self.memory[position] = self.Transition(ob, action, reward)
    
    
    def learn(self, discounted_ep_rs):
    
        log_prob = []
        loss = []
    
        input = self.memory[0][0]
        for i in range(len(self.memory)-1):
            input = torch.cat((input, self.memory[i+1][0]), 0)
        
        output = self.policyNet(input)
        output = F.log_softmax(output, dim=1)
    
        for i in range(len(self.memory)):
            log_prob.append(output[i][self.memory[i][1]])
    
        for i in range(len(log_prob)):
            # the core part of policy gradient
            loss.append(- log_prob[i] * discounted_ep_rs[i])
    
        loss = sum(loss)
      
        # mini-batch update neural network at once based on all data of current episode
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.memory.clear()
    
    
def main():
    
    global total_steps, position
    position = 0
    
    for episode in range(episodes_count):
        
        observation = env.reset()
        ob = torch.tensor([observation], dtype=torch.float)
        
        #print(position)
        record.append(position)
        position = 0
        
        while True:
            
            #env.render()
            
            action = policy.step_action(ob)
            
            observation2, reward, done, _ = env.step(action)
            ob2 = torch.tensor([observation2], dtype=torch.float)
            
            policy.memory_store(ob, action, reward, position)
            
            total_steps += 1

            # learn only at the end of each episode
            if(done):
                # normalize reward to easier learning
                discounted_ep_rs = policy.step_reward()
                policy.learn(discounted_ep_rs)
                #policy.memory.clear()
                '''
                # plot the effect of normalized reward
                if episode == 0:
                    plt.plot(discounted_ep_rs)    # plot the episode vt
                    plt.xlabel('episode steps')
                    plt.ylabel('normalized state-action value')
                    plt.show()
                '''    
                break
            
            position += 1
            ob = ob2
            #sleep(0.01)
    return 0


if __name__ == '__main__':
    
    record = []
    
    env = gym.make('CartPole-v1')
    env = env.unwrapped # otherwise, the maximum steps is 500 each episode
    
    policy = Policy()
    
    main()
    
    plt.plot(record)
            
        
        
    
    
    