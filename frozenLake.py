#!/usr/bin/env python
# coding: utf-8

# In[142]:


import gym
env = gym.make('FrozenLake-v0')
env.reset()
print('State space: ', env.observation_space)
print('Action space: ', env.action_space)
print(env.P[6])

# In[147]:


cur_state, reward, done, prob=env.step(1)
if(prob["prob"]==1.0):
    print(prob)
else:
    print(":(")


# In[148]:


env.render()


# In[63]:


import gym
import random
env = gym.make("FrozenLake-v0")
env.reset()
env.render()
reward=0.00

forbidden=[5,7,11,12]

actions = {
    'Left': 0,
    'Down': 1,
    'Right': 2,
    'Up': 3
}

#counter=0
counter2=0
bool=True
while(bool):
    #counter=counter+1
    winning_sequence=[random.choice(["Left","Down","Right"]),random.choice(["Left","Down","Right"]),random.choice(["Left","Down"]),random.choice(["Left","Down","Right","Up"])]
    for a in winning_sequence:
        new_state, reward, done, info = env.step(actions[a])
        counter2=counter2+1
        env.render()
        print("Reward: {:.2f}".format(reward))
        if new_state in forbidden:
            env.reset()
            break
        if new_state==15:
            bool=False
            break
#print("no.of attempts",counter)
print(counter2)
print("the winning sequence",winning_sequence)


# In[68]:


import gym
import random
import numpy as np

def frozenAI(env,hole_array,rows,columns):
    env.reset()
    #env.render()
    
    reward=0.00
    
    actions = {
        "Left": 0,
        "Down": 1,
        "Right": 2,
        "Up": 3
    }      
    
    counter=0
    counter_swim=0
    old_state=0    #Startposition
    bool=True
    goal=0
    alreadyDone=False
    move=random.choice(["Left","Down","Right","Up"])
    cur_state,reward,done,prob=env.step(actions[move])
    
    
    for x in range(rows):
        for y in range(columns):
            if hole_array[x][y]==2.0:
                alreadyDone=True
                goal=x*rows+y
    
    while(bool):
    
        cur_column=cur_state%rows
        cur_row=(cur_state-cur_column)//rows
        if(reward==1.0):
            bool=False
            hole_array[cur_row][cur_column]=2
            break;
        
       # env.render()
        counter=counter+1
        if(reward==0 and prob["prob"]==1.0):     #Not moved last round-> stuck in hole
            hole_array[cur_row][cur_column]=1
            env.reset()
            counter=counter-1
            counter_swim=counter_swim+1
        elif(hole_array[cur_row][max(cur_column-1,0)]==1): #Hole left-> we turn right to avoid it
            move="Right"
        elif(hole_array[cur_row][min(cur_column+1,columns-1)]==1):
            move="Left"
        elif(hole_array[min(cur_row+1,rows-1)][cur_column]==1): 
            move="Up"
        elif(hole_array[max(cur_row-1,0)][cur_column]==1):
            move="Down"
        elif(alreadyDone):
            if((goal-rows+1)>cur_state):
                move="Down"
            if(rows>cur_state):
                move="Right"
            if((goal+rows-1)<cur_state):
                move="Up"
            if(goal<cur_state):
                move="Left"
        else:
            if(old_state==cur_state-1):              #when there is no hole, we dont wont to go back-> we choose same option as before
                move="Right"
            elif(old_state==cur_state+1): 
                move="Left"
            elif(old_state==cur_state+rows):  
                move="Down"
            elif(old_state==cur_state-rows):  
                move="Up"    
            else:                               #we must have walked against the wall-> we go in direction away from wall
                if(cur_row==0):
                    move="Down"
                if(cur_column==0):
                    move="Right"
                if(cur_row==rows-1):    
                    move="Up"
                if(cur_column==columns-1):
                    move="Left"
        
        old_state=cur_state
        cur_state,reward,done,prob=env.step(actions[move])
    
    
    print(goal)                  
    print("Steps taken",counter)
    print("Went swimming:",counter_swim)
    print("Field:\n",hole_array)
    
    return hole_array

env = gym.make("FrozenLake8x8-v0")
array=frozenAI(env,np.zeros((8,8)),8,8)
array2=frozenAI(env,array,8,8)
# In[ ]
import gym
import random
env = gym.make("FrozenLake8x8-v0")
env.reset()
#env.render()
reward=0.00

forbidden=[19,29,35,41,42,46,49,52,54,59]

actions = {
    'Left': 0,
    'Down': 1,
    'Right': 2,
    'Up': 3
}
#counter=0
counter2=0
bool=True
while(bool):
    #counter=counter+1
    winning_sequence=[random.choice(["Left","Down","Right"]),random.choice(["Left","Down","Right"]),random.choice(["Left","Down"]),random.choice(["Left","Down","Right","Up"])]
    for a in winning_sequence:
        new_state, reward, done, info = env.step(actions[a])
        counter2=counter2+1
        #env.render()
        #print("Reward: {:.2f}".format(reward))
        if new_state in forbidden:
            env.reset()
            break
        if new_state==63:
            bool=False
            break
#print("no.of attempts",counter)
print(counter2)
print("the winning sequence",winning_sequence)
#%%
#In[]
import numpy as np
import gym
env = gym.make('FrozenLake8x8-v0')
env.seed(0)

def evaluate_policy(env, policy):
  total_rewards = 0.0
  for _ in range(100):
    obs = env.reset()
    while True:
      action = policy[obs]
      obs, reward, done, info = env.step(action)
      total_rewards += reward
      if done:
        break
  return total_rewards/100

def crossover(policy1, policy2):
  new_policy = policy1.copy()
  for i in range(16):
      rand = np.random.uniform()
      if rand > 0.5:
          new_policy[i] = policy2[i]
  return new_policy

def mutation(policy):
  new_policy = policy.copy()
  for i in range(64):
    rand = np.random.uniform()
    if rand < 0.05:
      new_policy[i] = np.random.choice(4)
  return new_policy


k=25
policy_pop = [np.random.choice(4, size=((64))) for _ in range(100)]
for idx in range(25):
  policy_scores = [evaluate_policy(env, pp) for pp in policy_pop]
  policy_ranks = list(reversed(np.argsort(policy_scores)))
  elite_set= [policy_pop[x] for x in policy_ranks[:k]]
  select_probs = np.array(policy_scores) / np.sum(policy_scores)
  child_set = [crossover(
      policy_pop[np.random.choice(range(100), p=select_probs)], 
      policy_pop[np.random.choice(range(100), p=select_probs)])
      for _ in range(100 - k)]
  k-=1
  mutated_list = [mutation(c) for c in child_set]
  policy_pop = elite_set
  policy_pop += mutated_list
policy_score = [evaluate_policy(env, pp) for pp in policy_pop]
best_policy = policy_pop[np.argmax(policy_score)]
print('Best actions score =', (np.max(policy_score)),'best actions =', best_policy.reshape(8,8))
env.close()