
# coding: utf-8

# A one dimension world:    ' *----T ',
# 
# where T is the target position, and '*' represents robot position.

# In[5]:


import numpy as np
import time

np.random.seed(2) # reproducible

env_columns = 6
env_rows = 1
lr = 0.01
gamma = 0.9
episode = 1
initial_pos = 0

Q_table = np.zeros((env_rows, env_columns))
reward = np.zeros((env_rows, env_columns))
reward[0][env_columns-1] = 100

print(Q_table)
print(reward)


# In[6]:


def env_update(current_pos):
    global episode
    global initial_pos
    if current_pos==(env_columns-1):
        terminal = True
        env = ['-'] * (env_columns - 1) + ['*']
        env_array = ''.join(env)
        print('\r{}'.format(env_array), end='')
        print(' Episode',episode, 'total_steps = ', total_steps)
        episode += 1
        time.sleep(1)
    else:
        terminal = False
        env = ['-'] * (env_columns - 1) + ['T']
        env[current_pos] = '*'
        env_array = ''.join(env)
        print('\r{}'.format(env_array), end='')
        time.sleep(0.2)
    
    return(terminal)


# In[7]:


for i in range(10):
    initial_pos = np.random.randint(0,5)
    current_pos = np.copy(initial_pos)
    total_steps = 0
    terminal = False
    #print(Q_table)

    while not terminal:
        
        terminal = env_update(current_pos)
    
        pos_left = current_pos - 1
        if pos_left<0: pos_left = 0
        pos_right = current_pos + 1
        if pos_right>(env_columns-1): pos_right = np.copy(env_columns - 1)

        Q_table[0][current_pos] = Q_table[0][current_pos] + lr * (reward[0][current_pos] + gamma * max(Q_table[0][pos_left], Q_table[0][pos_right]) - Q_table[0][current_pos])
                                                 
        if Q_table[0][pos_right]>Q_table[0][pos_left]:
            current_pos = pos_right
        elif Q_table[0][pos_left]>Q_table[0][pos_right]:
            current_pos = pos_left
        else:
            current_pos = np.random.choice([pos_left, pos_right])
            
        total_steps += 1
    
    
    


# In[8]:


np.set_printoptions(suppress=True)
print('', 'Q_table:', '\n', np.round(Q_table, decimals = 7))

