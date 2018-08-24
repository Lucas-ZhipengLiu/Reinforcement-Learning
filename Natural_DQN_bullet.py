#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 17:35:48 2018
@author: Lucas Liu

Implementing Deep Q Network (DQN)
On my designed task.
"""

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch

import pybullet as p
import pybullet_data

from collections import namedtuple
import matplotlib.pyplot as plt
from math import exp
import numpy as np
import random
import copy


class Net(nn.Module):
    
    def __init__(self):
        
        super(Net, self).__init__()
        
        self.bn0 = nn.BatchNorm1d(ob_size)
        self.fc1 = nn.Linear(ob_size, 400)
        self.bn1 = nn.BatchNorm1d(400)
        self.fc2 = nn.Linear(400, 300)
        self.bn2 = nn.BatchNorm1d(300)
        self.fc3 = nn.Linear(300, ac_size)
        
    def forward(self, x):
        
        x = self.bn0(x)
        x = F.relu( self.bn1(self.fc1(x)) )
        x = F.relu( self.bn2(self.fc2(x)) )
        x = self.fc3(x)
        
        return x
 

class Env():
    
    def __init__(self):
        
        self.physicsClient = p.connect(p.DIRECT)   # p.DIRECT or p.GUI
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=90, cameraPitch=-30, cameraTargetPosition=[0,1,0])
        self.base_pos = []
        
        self.angle1 = np.deg2rad(20)
        self.length1 = 2
        self.angle2 = np.deg2rad(10)
        self.length2 = 1
        self.radius = 15
        self.plane_urdf = '/home/lucas/modules/summer_project/bullet3-master/pybullet/gym/pybullet_data/plane.urdf'
        
        self.action_delta = 2 * action_bound / (ac_size-1)
        self.action_range = [-action_bound+self.action_delta*i for i in range(ac_size)]
        
    def reset(self):
        
        self.vt = 0
        
        p.resetSimulation()
        p.setGravity(0,0,-9.8)
        #self.planeId = p.loadURDF("plane.urdf")
        self.plane1 = p.loadURDF(self.plane_urdf,
                                 [0, -13, 0], 
                                 p.getQuaternionFromEuler([0, 0, 0]))

        self.plane2 = p.loadURDF(self.plane_urdf,
                                 [0, 2+self.radius*np.cos(self.angle1), -self.radius*np.sin(self.angle1)],
                                 p.getQuaternionFromEuler([-self.angle1, 0, 0]))

        self.plane3 = p.loadURDF(self.plane_urdf,
                                 [0, 3+self.length1*np.cos(self.angle1), -self.length1*np.sin(self.angle1)],
                                 p.getQuaternionFromEuler([0, 0, 0]))
        '''
        self.plane4 = p.loadURDF(self.plane_urdf,
                                 [0, 4+self.length1*np.cos(self.angle1)+self.length2*np.cos(self.angle2)-self.radius*np.cos(self.angle2), 
                                  self.length2*np.sin(self.angle2)-self.length1*np.sin(self.angle1)-self.radius*np.sin(self.angle2)],
                                 p.getQuaternionFromEuler([self.angle2, 0, 0]))

        self.plane5 = p.loadURDF(self.plane_urdf,
                                 [0, 4+self.length1*np.cos(self.angle1)+self.length2*np.cos(self.angle2)+self.radius, 
                                  self.length2*np.sin(self.angle2)-self.length1*np.sin(self.angle1)],
                                 p.getQuaternionFromEuler([0, 0, 0]))
        '''
        self.botId = p.loadURDF("/home/lucas/modules/summer_project/my_models/bot_origin2.urdf", StartPos, StartOrien)
        
        p.stepSimulation()
        
        observation = self.step_observation()
        
        return observation
      
        
    def step(self, action, ob):
        
        self.step_action(action)
        
        p.stepSimulation()
        
        ob2 = self.step_observation()
        
        reward = self.step_reward(ob2)
        
        done = self.step_done(ob2)
        
        return (ob2, reward, done, ob2[0][4])
    
    
    def step_action(self, action):
        
        acceleration = self.action_range[action]
        self.vt += acceleration
        self.vt = max(-25, min(self.vt, 25))
        # execute the action
        p.setJointMotorControl2(bodyUniqueId = self.botId, 
                                jointIndex = 0, 
                                controlMode = p.VELOCITY_CONTROL, 
                                targetVelocity = self.vt)
        p.setJointMotorControl2(bodyUniqueId = self.botId, 
                                jointIndex = 1, 
                                controlMode = p.VELOCITY_CONTROL, 
                                targetVelocity = -self.vt)
        
        
    def step_observation(self):
        
        base_pos, base_orn = p.getBasePositionAndOrientation(self.botId)
        base_orn = p.getEulerFromQuaternion(base_orn)
        base_linV, bas_anguV = p.getBaseVelocity(self.botId)
        _, joint_V, _, _ = p.getJointState(self.botId, 0)
        
        observation = torch.tensor([base_orn[0], bas_anguV[0], base_linV[1], joint_V, base_pos[1]], dtype=torch.float)
        observation = torch.unsqueeze(observation, 0)
        #print(observation)
        
        return observation
    
    
    def step_reward(self, ob2):
        #-exp(2*(x+0.2)^2)+1
        reward = -exp(2*(ob2[0][0]+0.05)**2)+1 + 2*ob2[0][4] + 4*ob2[0][2]
        reward = torch.tensor([reward], dtype=torch.float)
        reward = torch.unsqueeze(reward, 0)
        
        return reward
    
    
    def step_done(self, ob2):
        
        done = abs(ob2[0][0]) > 1.2
        
        return done

    
class DQN():
    
    def __init__(self):
        
        self.greedy_end = 0.05
        self.memory_size = memory_size
        self.anneal = 0.05 / self.memory_size # check!!!!
        
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
            greedy = self.greedy_end
        else:
            greedy = 0.9 - self.anneal * steps

        if(a < greedy):
            action = random.randint(0, ac_size - 1)          
        else:
            self.dqnNet = self.dqnNet.eval()
            output = self.dqnNet(ob)
            _, index = torch.max(output, dim=1)
            action = index.item()
            self.dqnNet = self.dqnNet.train()

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
    
    global targetNet_step, step_count, train_flag
    
    env1 = Env()
    
    dqn = DQN()
    
    for episode in range(episode_count):#episode_count
   
        ob = env1.reset()
        
        survival = 0
       
        while True: 

            action = dqn.step_action(ob, step_count)
            
            ob2, reward, done, base_pos = env1.step(action, ob)
            
            # store memory
            dqn.store_transitions(ob, action, reward, ob2)
            
            # train
            if(step_count>memory_size and train_flag):
                if(step_count == memory_size + 1):
                    print('-------------learning_start-------------')
                dqn.learn()
                targetNet_step += 1
                if(targetNet_step>target_update):
                    dqn.targetNet = copy.deepcopy(dqn.dqnNet)
                    targetNet_step = 0
            
            ob = ob2
            survival += 1
            step_count += 1
            if(done or survival>1999 or base_pos>9):
                print('Episode:', episode, ' survival:', survival, ' position: %.2f' % base_pos, 'action: %.2f' % action,
                      'base_linV: %.2f' % ob2[0][2], 'joint_V: %.2f' % ob2[0][3],)
                record_survival.append(survival)
                record_position.append(base_pos)
                if(base_pos>7):
                   train_flag = False
                   print('--------------Training STOP--------------')
                break    
        
        
if __name__ == '__main__':
    
    random.seed(1) #2 # 1
    torch.manual_seed(1) # 2 # 1
    
    ob_size = 5
    ac_size = 21
    
    # for bullet environment
    StartOrien = p.getQuaternionFromEuler([0, 0, 0])
    StartPos = [0, 0, 0]
    action_bound = torch.tensor([[4]], dtype=torch.float)

    # for deep Q network
    episode_count = 1000 # 2000 # 1000
    memory_size = 10000 # 100000 # 10000
    step_count = 0
    batch_size = 32 # 64 # 32
    gamma = 0.99
    target_update = 100
    targetNet_step = 0
  
    train_flag = True
    record_survival = []
    record_position = []
    
    main()