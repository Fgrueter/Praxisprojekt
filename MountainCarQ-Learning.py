#!/usr/bin/env python
# coding: utf-8

# In[10]:


import gym
env = gym.make('MountainCar-v0')
env.reset()
print('State space: ', env.observation_space)
print('Action space: ', env.action_space)


# In[11]:


print(env.observation_space.low)
print(env.observation_space.high)


# In[1]:


import numpy as np
import gym
import matplotlib.pyplot as plt

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()

# Define Q-learning function
def QLearning(env, learning, discount, epsilon, min_eps, episodes):
    # Determine size of discretized state space
    num_states = (env.observation_space.high - env.observation_space.low)*np.array([10, 100])
    num_states = np.round(num_states, 0).astype(int) + 1

#Testvariablen
    early_peak=0
    high_peak=200
    goal_reached=0
    total_reward=0
    total_reward_after_peak=0
    
    # Initialize Q table
    Q = np.random.uniform(low = -1, high = 1, 
                          size = (num_states[0], num_states[1], 
                                  env.action_space.n))
    # Initialize variables to track rewards
    # Q 19x15x3=285 Möglichkeiten
    reward_list = []
    ave_reward_list = []
    # Calculate episodic reduction in epsilon
    reduction = (epsilon - min_eps)/(episodes)
    
    # Run Q learning algorithm
    for i in range(episodes):
        # Initialize parameters
        done = False
        tot_reward, reward = 0,0
        state = env.reset()
        
        # Discretize state
        state_adj = (state - env.observation_space.low)*np.array([10, 100])
        state_adj = np.round(state_adj, 0).astype(int)
        counter=0
        while done != True:   
            # Render environment for last five episodes
            if i >= (episodes - 19):
               env.render()
                
            # Determine next action - epsilon greedy strategy
            if np.random.random() < 1 - epsilon:
                action = np.argmax(Q[state_adj[0], state_adj[1]]) #Best Mögliche Aktion für den Wert
            else:
                action = np.random.randint(0, env.action_space.n)
                
            # Get next state and reward
            state2, reward, done, info = env.step(action) 
            
            # Discretize state2
            state2_adj = (state2 - env.observation_space.low)*np.array([10, 100])
            state2_adj = np.round(state2_adj, 0).astype(int)
            
            #Allow for terminal states
            if done and state2[0] >= 0.5:
                Q[state_adj[0], state_adj[1], action] = reward
                
                if(early_peak==0):
                    early_peak=i
                goal_reached+=1
                if((200-counter)<high_peak):
                    high_peak=counter
                
                break;
            # Adjust Q value for current state
            else:
                delta = learning*(reward + 
                                 discount*np.max(Q[state2_adj[0],               #Hier wird die beste Aktion für state ausgewählt
                                                   state2_adj[1]]) - 
                                 Q[state_adj[0], state_adj[1],action])
                Q[state_adj[0], state_adj[1],action] += delta
            # Update 
            tot_reward += reward
            state_adj = state2_adj
            
            counter+=1
            total_reward+=reward
            if(i>=early_peak and early_peak>0):
                total_reward_after_peak+=reward
            
            
                
        # Decay epsilon
        if epsilon > min_eps:
            epsilon -= reduction
        
        # Track rewards
        reward_list.append(tot_reward)

        if (i+1) % 100 == 0:
            ave_reward = np.mean(reward_list)
            ave_reward_list.append(ave_reward)
            reward_list = []
            
        if (i+1) % 100 == 0:   
            print('Episode {} Average Reward: {}'.format(i+1, ave_reward))
        
            
    env.close()
    
    print(early_peak)
    print(high_peak)
    print(goal_reached)
    print(total_reward)
    print(total_reward_after_peak)

    return ave_reward_list

# Run Q-learning algorithm
rewards = QLearning(env, 0.2, 0.9, 0.8, 0, 1500)#learning, discount, epsilon, min_eps, episodes

#Plot Rewards
plt.plot(100*(np.arange(len(rewards)) + 1), rewards)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Episodes')
plt.show()

# In[1]:


import numpy as np
import gym
import matplotlib.pyplot as plt

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()

# Define Q-learning function
def QLearning(env, learning, discount, epsilon, min_eps, episodes):
    # Determine size of discretized state space
    num_states = (env.observation_space.high - env.observation_space.low)*np.array([10, 100])
    num_states = np.round(num_states, 0).astype(int) + 1

#Testvariablen
    early_peak=0
    high_peak=200
    goal_reached=0
    total_reward=0
    total_reward_after_peak=0
    
    # Initialize Q table
    Q = np.random.uniform(low = -1, high = 1, 
                          size = (num_states[0], num_states[1], 
                                  env.action_space.n))
    # Initialize variables to track rewards
    # Q 19x15x3=285 Möglichkeiten
    reward_list = []
    ave_reward_list = []
    # Calculate episodic reduction in epsilon
    reduction = (epsilon - min_eps)*10/(episodes)
    
    # Run Q learning algorithm
    for i in range(episodes):
        # Initialize parameters
        done = False
        tot_reward, reward = 0,0
        state = env.reset()
        
        # Discretize state
        state_adj = (state - env.observation_space.low)*np.array([10, 100])
        state_adj = np.round(state_adj, 0).astype(int)
        counter=0
        while done != True:   
            # Render environment for last five episodes
           # if i >= (episodes - 19):
            #   env.render()
                
            # Determine next action - epsilon greedy strategy
            if np.random.random() < 1 - epsilon:
                action = np.argmax(Q[state_adj[0], state_adj[1]]) #Best Mögliche Aktion für den Wert
            else:
                action = np.random.randint(0, env.action_space.n)
                
            # Get next state and reward
            state2, reward, done, info = env.step(action) 
            
            # Discretize state2
            state2_adj = (state2 - env.observation_space.low)*np.array([10, 100])
            state2_adj = np.round(state2_adj, 0).astype(int)
            
            #Allow for terminal states
            if done and state2[0] >= 0.5:
                Q[state_adj[0], state_adj[1], action] = reward
                
                if(early_peak==0):
                    early_peak=i
                goal_reached+=1
                if((200-counter)<high_peak):
                    high_peak=counter
                
            # Adjust Q value for current state
            else:
                delta = learning*(reward + 
                                 discount*np.max(Q[state2_adj[0],               #Hier wird die beste Aktion für state ausgewählt
                                                   state2_adj[1]]) - 
                                 Q[state_adj[0], state_adj[1],action])
                Q[state_adj[0], state_adj[1],action] += delta
            # Update 
            tot_reward += reward
            state_adj = state2_adj
            
            counter+=1
            total_reward+=reward
            if(i>=early_peak and early_peak>0):
                total_reward_after_peak+=reward
            
            
                
        # Decay epsilon
        if epsilon > min_eps:
            epsilon -= reduction
        
        # Track rewards
        reward_list.append(tot_reward)
        
        if (i+1) % 100 == 0:
            ave_reward = np.mean(reward_list)
            ave_reward_list.append(ave_reward)
            reward_list = []
            
        if (i+1) % 100 == 0:   
            print('Episode {} Average Reward: {}'.format(i+1, ave_reward))
        
            
    env.close()
    
    print(early_peak)
    print(high_peak)
    print(goal_reached)
    print(total_reward)
    print(total_reward_after_peak)

    return Q

def QModel(env,Q,episodes):
    # Determine size of discretized state space
    num_states = (env.observation_space.high - env.observation_space.low)*np.array([10, 100])
    num_states = np.round(num_states, 0).astype(int) + 1

#Testvariablen
    early_peak2=-1
    high_peak2=200
    goal_reached2=0
    total_reward2=0
    total_reward_after_peak2=0
    
    reward_list2 = []
    ave_reward_list2 = []
    # Run Q learning algorithm
    for i in range(episodes):
        # Initialize parameters
        done = False
        tot_reward2, reward2 = 0,0
        state = env.reset()
        
        # Discretize state
        state_adj = (state - env.observation_space.low)*np.array([10, 100])
        state_adj = np.round(state_adj, 0).astype(int)
        counter2=0
        while done != True:   
            # Render environment for last five episodes
            #if i >= (episodes - 19):
            #   env.render()
                
            # Determine next action - epsilon greedy strategy
            action2 = np.argmax(Q[state_adj[0], state_adj[1]]) #Best Mögliche Aktion für den Wert
            
            # Get next state and reward
            state2, reward2, done, info = env.step(action2) 
            
            # Discretize state2
            state2_adj = (state2 - env.observation_space.low)*np.array([10, 100])
            state2_adj = np.round(state2_adj, 0).astype(int)
            
            if done and state2[0] >= 0.5:
                if(early_peak2==-1):
                    early_peak2=i
                goal_reached2+=1
                if((200-counter2)<high_peak2):
                    high_peak2=counter2
                                
            # Update 
            tot_reward2 += reward2
            state_adj = state2_adj
            
            counter2+=1
            total_reward2+=reward2
            if(i>=early_peak2 and early_peak2>=0):
                total_reward_after_peak2+=reward2
        
        # Track rewards
        reward_list2.append(tot_reward2)

        if (i+1) % 100 == 0:
            ave_reward2 = np.mean(reward_list2)
            ave_reward_list2.append(ave_reward2)
            reward_list2 = []
            
        if (i+1) % 100 == 0:   
            print('Episode {} Average Reward: {}'.format(i+1, ave_reward2))
        
            
    env.close()
    
    print(early_peak2)
    print(high_peak2)
    print(goal_reached2)
    print(total_reward2)
    print(total_reward_after_peak2)

    return ave_reward_list2
# Run Q-learning algorithm
Qtable = QLearning(env, 0.2, 0.9, 0.8, 0, 10000)#learning, discount, epsilon, min_eps, episodes
rewards=QModel(env,Qtable,500)
#Plot Rewards
plt.plot(100*(np.arange(len(rewards)) + 1), rewards)
rewards=QModel(env,Qtable,500)
#Plot Rewards
plt.plot(100*(np.arange(len(rewards)) + 1), rewards)
rewards=QModel(env,Qtable,500)
#Plot Rewards
plt.plot(100*(np.arange(len(rewards)) + 1), rewards)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Episodes')
plt.show()
# In[1]:

