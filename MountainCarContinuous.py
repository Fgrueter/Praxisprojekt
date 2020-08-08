import numpy as np
import gym
import matplotlib.pyplot as plt
import math

# Import and initialize Mountain Car Environment

env = gym.make('MountainCarContinuous-v0')
env.reset()

print('State space: ', env.observation_space)
print('Action space: ', env.action_space)

print(env.action_space.low)
print(env.action_space.high)

# Define Q-learning function
def QLearning(env, learning, discount, epsilon, min_eps, episodes):
    # Determine size of discretized state space
    num_states = (env.observation_space.high - env.observation_space.low)*np.array([10, 100])
    num_states = np.round(num_states, 0).astype(int) + 1
    
#Testvariable
    goal_reached=0
    
    num_acts=5
    # Initialize Q table
    Q = np.random.uniform(low = -0.01, high = 0.01,size = (num_states[0], num_states[1],num_acts))
    # Initialize variables to track rewards
    reward_list = []
    ave_reward_list = []
    # Calculate episodic reduction in epsilon
    reduction = (epsilon - min_eps)*8/episodes  
    # Run Q learning algorithm
    for i in range(episodes):
        # Initialize parameters
        done = False
        tot_reward, reward = 0,0
        state = env.reset()
        
        # Discretize state
        state_adj = (state - env.observation_space.low)*np.array([10, 100])
        state_adj = np.round(state_adj, 0).astype(int)
        
        action = np.ndarray(1)
        while done != True:   
            # Render environment for last five episodes
            #if i >= (episodes - 19):
            #   env.render()
                
            # Determine next action - epsilon greedy strategy
            if np.random.random() < 1 - epsilon:
                action[0]= np.argmax(Q[state_adj[0], state_adj[1]])/(num_acts-1)*(env.action_space.high-env.action_space.low)+env.action_space.low #Best Mögliche Aktion für den Wert
                actionCell= int((num_acts-1)/(env.action_space.high-env.action_space.low)*(action[0]-env.action_space.low)) #Wert wird für die Tabelle diskretisiert,da in ihr nur Integer gespeichert werden können
            else:
               # action = np.random.randfloat(env.action_space.low,env.action_space.high)
                action[0]=2*np.random.random()-1
                actionCell= int((num_acts-1)/(env.action_space.high-env.action_space.low)*(action[0]-env.action_space.low))
            # Get next state and reward
            state2, reward, done, info = env.step(action) 
            reward += math.pow(action[0], 2) * 0.1              #Der reward wird in der Umgebung um math.pow(action[0],2)*0.1 verringert. Da dadurch nur noch zu kleine Bewegungen ausgewählt wurden, wird der reward wieder um diesen Wert erhöht

            # Discretize state2
            state2_adj = (state2 - env.observation_space.low)*np.array([10, 100])
            state2_adj = np.round(state2_adj, 0).astype(int)
            
            #Allow for terminal states
            if done and state2[0] >= 0.45:          #Wurde in CarContinous geändert
                #Q[state_adj[0], state_adj[1], action] = reward
                delta = learning*(reward - Q[state_adj[0],state_adj[1],actionCell])
                Q[state_adj[0], state_adj[1],actionCell] += delta
                goal_reached+=1

                
            # Adjust Q value for current state
            else:
                delta = learning*(reward+discount*np.max(Q[state2_adj[0],state2_adj[1]])-Q[state_adj[0],state_adj[1],actionCell])  #Hier wird die beste Aktion für state ausgewählt
                Q[state_adj[0], state_adj[1],actionCell] += delta
            # Update variables
            tot_reward += reward
            state_adj = state2_adj           
                
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
    
    print(goal_reached)
   
    return Q,ave_reward_list
            

#Diese Funktion testet, wie gut die Ergebnisse einer erlernten QTabelle sind. Dabei wird die Tabelle nicht weiter verändert und nur noch der optimale Wert für den Zustand aus der Tabelle verwendet
#Für Q wird die Q-Tabelle von Q-Learning übergeben und diese dann über die übergebene Episodenanzahl getestet
def QModel(env,Q,episodes):
    # Determine size of discretized state space

   
#Testvariablen
   
    goal_reached2=0

    
    num_acts2=5
    action2 = np.ndarray(1)

    # Initialize variables to track rewards
    reward_list2 = []
    ave_reward_list2 = []
    # Calculate episodic reduction in epsilon
    # Run Q learning algorithm
    for i in range(episodes):
        # Initialize parameters
        done = False
        tot_reward2, reward2 = 0,0
        state = env.reset()
        
        # Discretize state
        state_adj = (state - env.observation_space.low)*np.array([10, 100])
        state_adj = np.round(state_adj, 0).astype(int)
        
        while done != True:   
            # Render environment for last five episodes
            #if i >= (episodes - 19):
            #   env.render()
                
            # Determine next action - epsilon greedy strategy
            action2[0]= np.argmax(Q[state_adj[0], state_adj[1]])/(num_acts2-1)*(env.action_space.high-env.action_space.low)+env.action_space.low #Best Mögliche Aktion für den Wert             
            # Get next state and reward
            state2, reward2, done, info = env.step(action2) 

            # Discretize state2
            state2_adj = (state2 - env.observation_space.low)*np.array([10, 100])
            state2_adj = np.round(state2_adj, 0).astype(int)
            
            #Allow for terminal states
            if done and state2[0] >= 0.45: # The goal now is on the position 0.45

                goal_reached2+=1

            # Update variables
            tot_reward2 += reward2
            state_adj = state2_adj

        
        # Track rewards
        reward_list2.append(tot_reward2)

        if (i+1) % 100 == 0:
            ave_reward2 = np.mean(reward_list2)
            ave_reward_list2.append(ave_reward2)
            reward_list2 = []
            
        if (i+1) % 100 == 0:   
            print('Episode {} Average Reward: {}'.format(i+1, ave_reward2))
        
            
    env.close()
    print(goal_reached2)


    return ave_reward_list2

# Run Q-learning algorithm
num_states = (env.observation_space.high - env.observation_space.low)*np.array([10, 100])
num_states = np.round(num_states, 0).astype(int) + 1
Q = np.random.uniform(low = -0.01, high = 0.01, size = (num_states[0], num_states[1],5))
Q,rewards = QLearning(env, 0.1, 0.9, 0.8, 0, 10000)#learning, discount, epsilon, min_eps, episodes

plt.plot(100*(np.arange(len(rewards)) + 1), rewards)

rewards = QModel(env,Q,500)#learning, discount, epsilon, min_eps, episodes

#Plot Rewards
plt.plot(100*(np.arange(len(rewards)) + 1), rewards)
rewards=QModel(env,Q,500)
#Plot Rewards
plt.plot(100*(np.arange(len(rewards)) + 1), rewards)
rewards=QModel(env,Q,500)
#Plot Rewards
plt.plot(100*(np.arange(len(rewards)) + 1), rewards)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Episodes')
plt.show()

