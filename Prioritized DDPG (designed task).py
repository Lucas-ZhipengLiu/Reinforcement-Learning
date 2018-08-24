#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 09:03:23 2018
@author: Lucas Liu

Implementing Priorited Deep Deterministic Policy Gradient (PDDPG)
On my designed task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pybullet as p
import pybullet_data

from collections import namedtuple
import matplotlib.pyplot as plt
from math import exp
import numpy as np
#from time import sleep
#import random
import copy


class Net1(nn.Module): # Actor nn
    
    def __init__(self):
        
        super(Net1, self).__init__()
        
        self.bn0 = nn.BatchNorm1d(ob_num)
        self.fc1 = nn.Linear(ob_num, 400)
        self.bn1 = nn.BatchNorm1d(400)
        self.fc2 = nn.Linear(400, 300)
        self.bn2 = nn.BatchNorm1d(300)
        self.fc3 = nn.Linear(300, ac_num)
        self.bn3 = nn.BatchNorm1d(ac_num)
        
    def forward(self, x):
        
        x = self.bn0(x)
        x = F.relu( self.bn1(self.fc1(x)) )
        x = F.relu( self.bn2(self.fc2(x)) )
        x = F.tanh( self.bn3(self.fc3(x)) )
            
        return x
    
    
class Net2(nn.Module): # Critic nn
    
    def __init__(self):
        
        super(Net2, self).__init__()
        
        self.bn0 = nn.BatchNorm1d(ob_num)
        self.fc1 = nn.Linear(ob_num, 400)
        self.bn1 = nn.BatchNorm1d(400)
        self.fc2_state = nn.Linear(400, 300)
        self.fc2_action = nn.Linear(ac_num, 300)
        self.fc3 = nn.Linear(300, 1)
     
    def forward(self, x, a):
        
        x = self.bn0(x)
        x = F.relu( self.bn1(self.fc1(x)) )
        # action included at second hidden layer
        x = F.relu( self.fc2_state(x) + self.fc2_action(a) )
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
        self.length2 = 2
        self.radius = 15
        self.plane_urdf = '/home/lucas/modules/summer_project/bullet3-master/pybullet/gym/pybullet_data/plane.urdf'
        
        
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

        self.plane4 = p.loadURDF(self.plane_urdf,
                                 [0, 4+self.length1*np.cos(self.angle1)+self.length2*np.cos(self.angle2)-self.radius*np.cos(self.angle2), 
                                  self.length2*np.sin(self.angle2)-self.length1*np.sin(self.angle1)-self.radius*np.sin(self.angle2)],
                                 p.getQuaternionFromEuler([self.angle2, 0, 0]))

        self.plane5 = p.loadURDF(self.plane_urdf,
                                 [0, 4+self.length1*np.cos(self.angle1)+self.length2*np.cos(self.angle2)+self.radius, 
                                  self.length2*np.sin(self.angle2)-self.length1*np.sin(self.angle1)],
                                 p.getQuaternionFromEuler([0, 0, 0]))
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
    
    
    def step_action(self, acceleration):
        
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
        reward = -exp(2*(ob2[0][0]+0.1)**2)+1 + 2*ob2[0][4] + 4*ob2[0][2]
        reward = torch.tensor([reward], dtype=torch.float)
        reward = torch.unsqueeze(reward, 0)
        
        return reward
    
    
    def step_done(self, ob2):
        
        done = abs(ob2[0][0]) > 1.2
        
        return done
        
    
    
class DDPG():
    
    def __init__(self):
        
        # initialize memory
        self.prob_alpha = 0.6
        self.memory_size = memory_count
        self.memory = []
        self.priorities = np.zeros((self.memory_size), dtype = np.float32)
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
        
        # for OU noise
        self.N = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        
        # initialize actor network
        self.actorNet = Net1()
        self.params_A = list(self.actorNet.parameters())
        # initialize each layers' parameters
        nn.init.uniform(self.params_A[2], - ob_num ** (-0.5), ob_num ** (-0.5))
        nn.init.uniform(self.params_A[3], - ob_num ** (-0.5), ob_num ** (-0.5))
        nn.init.uniform(self.params_A[6], - 400 ** (-0.5), 400 ** (-0.5))
        nn.init.uniform(self.params_A[7], - 400 ** (-0.5), 400 ** (-0.5))
        nn.init.uniform(self.params_A[10], -0.003, 0.003) 
        nn.init.uniform(self.params_A[11], -0.003, 0.003) 
        
        self.actorNet.zero_grad()
        self.optimizor_A = optim.Adam(self.actorNet.parameters(), lr=0.0001)
        
        
        # initialize critic network
        self.criticNet = Net2()
        self.params_C = list(self.criticNet.parameters())
        # initialize each layers' parameters
        nn.init.uniform(self.params_C[2], - ob_num ** (-0.5), ob_num ** (-0.5))
        nn.init.uniform(self.params_C[3], - ob_num ** (-0.5), ob_num ** (-0.5))
        nn.init.uniform(self.params_C[6], - 400 ** (-0.5), 400 ** (-0.5))
        nn.init.uniform(self.params_C[7], - 400 ** (-0.5), 400 ** (-0.5))
        nn.init.uniform(self.params_C[8], - ac_num ** (-0.5), ac_num ** (-0.5))
        nn.init.uniform(self.params_C[9], - ac_num ** (-0.5), ac_num ** (-0.5))
        nn.init.uniform(self.params_C[10], -0.003, 0.003) 
        nn.init.uniform(self.params_C[11], -0.003, 0.003) 
        
        self.criticNet.zero_grad()
        self.optimizor_C = optim.Adam(self.criticNet.parameters(), lr=0.001, weight_decay=0.01)
        # MSELoss equals '1/N * (q_eval-q_target)^2', but has better effect
        self.criterion = nn.MSELoss()
        
        # initialize target networks
        self.criticNet_target = copy.deepcopy(self.criticNet)
        self.actorNet_target = copy.deepcopy(self.actorNet)
        
        
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
        
        ob1 = torch.zeros(miniBatch_num, ob_num)
        ob2 = torch.zeros(miniBatch_num, ob_num)
        ac1 = torch.zeros(miniBatch_num, 1)
        rewards = torch.zeros(miniBatch_num, 1)
        
        # compute probabilities of every sample
        probs = self.priorities**self.prob_alpha
        probs /= probs.sum()
        indices = np.random.choice(self.memory_size, miniBatch_num, p=probs)
        samples = [self.memory[idx] for idx in indices]
        
        # compute weights of every batch sample
        batch_weights = (miniBatch_num * probs[indices])**(beta) # w = (N*P(i))^(-β)
        batch_weights /= batch_weights.max()
        
        # unpack from samples
        for i in range(miniBatch_num):
            
            ob1[i]     = samples[i][0]
            ac1[i]     = samples[i][1]
            rewards[i] = samples[i][2]
            ob2[i]     = samples[i][3]
        
        return ob1, ac1, rewards, ob2, indices, torch.tensor(batch_weights)
    
    
    def selectAction(self, ob, action_bound):
        
        self.actorNet = self.actorNet.eval()
        action = self.actorNet(ob)
        action = action * action_bound
        self.actorNet = self.actorNet.train()
        # the variable requires grad cannot be used in env.step()
        return action.detach() 
        
    
    def learn(self, beta):
        
        # samples from memory
        ob1, ac1, R, ob2, indices, batch_weights = self.sample_transitions(beta)
        batch_weights = batch_weights.unsqueeze(-1)
    
        # calculate loss_C
        ac2 = self.actorNet_target(ob2)
        ac2 = ac2.detach() # targetNet dosen't update
        
        ob1_value = self.criticNet(ob1, ac1)
        ob2_value = self.criticNet_target(ob2, ac2)
        ob2_value = ob2_value.detach() # targetNet dosen't update
        
        q_target = R + gamma * ob2_value
        q_eval   = ob1_value
        #loss_C   = self.criterion(q_eval, q_target) 
        new_loss = batch_weights * (q_eval - q_target)**2
        loss_C = new_loss.mean()
        new_priorities = new_loss + 1e-5
        
        # update critic nn 
        # (the order of following three commomds cannot be changed)
        self.optimizor_C.zero_grad()
        loss_C.backward()
        self.optimizor_C.step()
        
        # calculate loss_A      
        ac      = self.actorNet(ob1)
        q_value = self.criticNet(ob1, ac)      
        loss_A  = - torch.mean(q_value) # gradient acent
        
        # update actor nn
        # (the order of following three commomds cannot be changed)
        self.optimizor_A.zero_grad()
        loss_A.backward()
        self.optimizor_A.step()
               
        # priorities update
        for i in range(miniBatch_num):            
            self.priorities[indices[i]] = new_priorities[i]
        
        
        
    def targetNets_softUpdate(self):
        
        self.actorNet_target.fc1.weight.data = TAU * self.actorNet.fc1.weight.data + (1-TAU) * self.actorNet_target.fc1.weight.data
        self.actorNet_target.fc1.bias.data   = TAU * self.actorNet.fc1.bias.data   + (1-TAU) * self.actorNet_target.fc1.bias.data
        self.actorNet_target.fc2.weight.data = TAU * self.actorNet.fc2.weight.data + (1-TAU) * self.actorNet_target.fc2.weight.data
        self.actorNet_target.fc2.bias.data   = TAU * self.actorNet.fc2.bias.data   + (1-TAU) * self.actorNet_target.fc2.bias.data
        self.actorNet_target.fc3.weight.data = TAU * self.actorNet.fc3.weight.data + (1-TAU) * self.actorNet_target.fc3.weight.data
        self.actorNet_target.fc3.bias.data   = TAU * self.actorNet.fc3.bias.data   + (1-TAU) * self.actorNet_target.fc3.bias.data
        
        self.actorNet_target.bn0.weight.data = self.actorNet.bn0.weight.data
        self.actorNet_target.bn0.bias.data   = self.actorNet.bn0.bias.data
        self.actorNet_target.bn1.weight.data = self.actorNet.bn1.weight.data
        self.actorNet_target.bn1.bias.data   = self.actorNet.bn1.bias.data
        self.actorNet_target.bn2.weight.data = self.actorNet.bn2.weight.data
        self.actorNet_target.bn2.bias.data   = self.actorNet.bn2.bias.data
        self.actorNet_target.bn3.weight.data = self.actorNet.bn3.weight.data
        self.actorNet_target.bn3.bias.data   = self.actorNet.bn3.bias.data
        
        self.criticNet_target.fc1.weight.data        = TAU * self.criticNet.fc1.weight.data        + (1-TAU) * self.criticNet_target.fc1.weight.data
        self.criticNet_target.fc1.bias.data          = TAU * self.criticNet.fc1.bias.data          + (1-TAU) * self.criticNet_target.fc1.bias.data
        self.criticNet_target.fc2_state.weight.data  = TAU * self.criticNet.fc2_state.weight.data  + (1-TAU) * self.criticNet_target.fc2_state.weight.data
        self.criticNet_target.fc2_state.bias.data    = TAU * self.criticNet.fc2_state.bias.data    + (1-TAU) * self.criticNet_target.fc2_state.bias.data
        self.criticNet_target.fc2_action.weight.data = TAU * self.criticNet.fc2_action.weight.data + (1-TAU) * self.criticNet_target.fc2_action.weight.data
        self.criticNet_target.fc2_action.bias.data   = TAU * self.criticNet.fc2_action.bias.data   + (1-TAU) * self.criticNet_target.fc2_action.bias.data
        self.criticNet_target.fc3.weight.data        = TAU * self.criticNet.fc3.weight.data        + (1-TAU) * self.criticNet_target.fc3.weight.data
        self.criticNet_target.fc3.bias.data          = TAU * self.criticNet.fc3.bias.data          + (1-TAU) * self.criticNet_target.fc3.bias.data
        
        self.criticNet_target.bn0.weight.data = self.criticNet.bn0.weight.data
        self.criticNet_target.bn0.bias.data   = self.criticNet.bn0.bias.data
        self.criticNet_target.bn1.weight.data = self.criticNet.bn1.weight.data
        self.criticNet_target.bn1.bias.data   = self.criticNet.bn1.bias.data
        
        
    def ou_noise(self, action, theta=0.15, mu=0, sigma=0.2):
        
        noise = theta * (mu - action) + sigma * self.N.sample()
        
        return noise
    
    
    
def main():
    
    global step_count, epsilon, train_flag
    beta = beta_start
    frame_idx = 0
    
    for episode in range(episode_count):
        
        #ep_reward = 0
        
        ob = env.reset()
        
        survival = 0
        
        while True:
            
            frame_idx += 1
            beta = min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)
            epsilon -= 1.0 / explore # for ou noise
            
            action = ddpg.selectAction(ob, action_bound)
            noise = ddpg.ou_noise(action)
            action = max(-action_bound, min(action + (max(epsilon, 0) * noise), action_bound)) # action clamp
        
            ob2, reward, done, base_pos = env.step(action, ob)
        
            ddpg.store_transitions(ob, action, reward, ob2)
        
            if (step_count > memory_count and train_flag):
            
                if(step_count == memory_count + 1):
                    print('------------------learning_start------------------')
                    
                ddpg.learn(beta)
            
                ddpg.targetNets_softUpdate()
            
            #ep_reward += reward
            survival += 1
            step_count += 1
            ob = ob2
            
            if (done or survival > 1999 or base_pos>13): 
                #ep_reward = ep_reward/survival
                print('Episode:', episode, ' survival:', survival, ' position: %.2f' % base_pos, 'action: %.2f' % action.item(),
                      'base_linV: %.2f' % ob2[0][2], 'joint_V: %.2f' % ob2[0][3], 'explore: %.4f' % epsilon,) 
                record_survival.append(survival)
                record_position.append(base_pos)
                if (base_pos>10):
                    train_flag = False
                    print('------------------Training STOP-------------------')    
                break
        
    
if __name__ == '__main__':

    np.random.seed(12)
    torch.manual_seed(12)
    
    episode_count = 1000 
    memory_count = 100000 # 200000 100000
    miniBatch_num = 64
    step_count = 0
    gamma = 0.99
    TAU = 0.001
    ob_num = 5 
    ac_num = 1
    
    StartOrien = p.getQuaternionFromEuler([0, 0, 0])
    StartPos = [0, 0, 0]
    action_bound = torch.tensor([[4]], dtype=torch.float)
    explore = 300000
    epsilon = 1
    
    record_survival = []
    record_position = []
    
    train_flag = True
    beta_frames = 100000 # back to uniform sampleing after 2e+4 steps
    beta_start = 0.4
    
    env = Env()
    
    ddpg = DDPG()
    
    main()
    
    plt.plot(record_survival)
    plt.plot(record_position)
