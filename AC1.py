#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 16:45:48 2018

@author: lucas
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
import gym
import matplotlib.pyplot as plt


class Net(nn.Module):
    
    def __init__(self):
        
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(4, 20)
        self.fc2_action = nn.Linear(20, 2)
        self.fc2_state = nn.Linear(20, 1)
        
    def forward(self, x):
        
        x = self.fc1(x)
        x = F.relu(x)
        action_value = self.fc2_action(x)
        state_value = self.fc2_state(x)
        
        return action_value, state_value


class Actor():
    
    def __init__(self):
        
        self.actorNet = Net()
        self.actorNet.zero_grad()
        self.optimizor = optim.Adam(self.actorNet.parameters(), lr=0.001)
        
    def step_action(self, ob):
        
        action_value, _ = self.actorNet(ob)
        action_prob = F.softmax(action_value, dim=1)
        action = random.choices([0, 1], action_prob[0], k=1)
        
        return action[0]
    
    def learn(self, ob, action, td_error):
        
        action_value, _ = self.actorNet(ob)
        action_prob = F.log_softmax(action_value, dim=1)
        log_prob = action_prob[0][action]
        
        self.loss = -log_prob * td_error 
        
        self.optimizor.zero_grad()
        self.loss.backward()
        self.optimizor.step()
        
        
class Critic():
    
    def __init__(self):
        
        self.gamma = 0.9
        
        self.criticNet = Net()
        self.criticNet.zero_grad()
        self.optimizor = optim.Adam(self.criticNet.parameters(), lr=0.01)
        
    def learn(self, ob, ob2, reward):
        
        _, ob_value = self.criticNet(ob)
        _, ob2_value = self.criticNet(ob2)
        
        td_error = reward + self.gamma * ob2_value - ob_value
        
        self.loss = torch.mul(td_error, td_error)
        
        self.optimizor.zero_grad()
        self.loss.backward()
        self.optimizor.step()
        
        return td_error
    
    
def main():
    
    position = 0
    
    for episode in range(episode_count):
        
        observation = env.reset()
        ob = torch.tensor([observation], dtype=torch.float)
        
        #print(position)
        record.append(position)
        position = 0
        
        while True:
            
            #env.render()
            
            action = actor.step_action(ob)
            
            observation2, reward, done, _ = env.step(action)
            
            ob2 = torch.tensor([observation2], dtype=torch.float)
            
            if(done): reward = -20
            
            td_error = critic.learn(ob, ob2, reward)
            TD_error = td_error.detach()
            actor.learn(ob, action, TD_error)
            
            ob = ob2
            position += 1
            
            if(done): break



if __name__ == '__main__':
    
    episode_count = 1000
    record = []
    
    env = gym.make('CartPole-v1')
    env = env.unwrapped
    
    actor = Actor()
    critic = Critic()
    
    main()
    
    plt.plot(record)
    