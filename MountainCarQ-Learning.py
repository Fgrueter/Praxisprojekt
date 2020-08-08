import numpy as np
import gym
import matplotlib.pyplot as plt

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()

# Define Q-learning function
def QLearning(env, learning, discount, epsilon, min_eps, episodes):
    # Determine size of discretized state space
    array=np.array([10, 100])
    num_states = (env.observation_space.high - env.observation_space.low)*array
    num_states = np.round(num_states, 0).astype(int) + 1

#Testvariable
    goal_reached=0

    
    # Initialize Q table
    Q = np.random.uniform(low = -0.01, high = 0.01, size = (num_states[0], num_states[1],env.action_space.n))
    # Initialize variables to track rewards
    # Q 19x15x3=855 Möglichkeiten
    reward_list = []
    ave_reward_list = []
    # Calculate episodic reduction in epsilon
    reduction = (epsilon - min_eps*10)/(episodes)
    # Run Q learning algorithm
    for i in range(episodes):
        # Initialize parameters
        done = False
        tot_reward, reward = 0,0
        state = env.reset()
        
        # Discretize state
        state_adj = (state - env.observation_space.low)*array
        state_adj = np.round(state_adj, 0).astype(int)
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
            state2_adj = (state2 - env.observation_space.low)*array
            state2_adj = np.round(state2_adj, 0).astype(int)
            
            #Allow for terminal states
            if done and state2[0] >= 0.5:
               # Q[state_adj[0], state_adj[1], action] = reward
                delta = learning*(reward - Q[state_adj[0], state_adj[1],action])
                Q[state_adj[0], state_adj[1],action] += delta

                goal_reached+=1

                
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
    array=np.array([10, 110])
    num_states = (env.observation_space.high - env.observation_space.low)*array
    num_states = np.round(num_states, 0).astype(int) + 1

#Testvariablen

    goal_reached2=0
    stehen=0
    rechts=0
    links=0
    reward_list2 = []
    ave_reward_list2 = []
    # Run Q learning algorithm
    for i in range(episodes):
        # Initialize parameters
        done = False
        tot_reward2, reward2 = 0,0
        state = env.reset()
        
        # Discretize state
        state_adj = (state - env.observation_space.low)*array
        state_adj = np.round(state_adj, 0).astype(int)
        while done != True:   
            # Render environment for last five episodes
            #if i >= (episodes - 19):
            #   env.render()
                
            # Determine next action - epsilon greedy strategy
            action2 = np.argmax(Q[state_adj[0], state_adj[1]]) #Best Mögliche Aktion für den Wert
            if(action2==2):                     
                rechts=rechts+1
            elif(action2==1):               #Kontrolle, welche Aktionen ausgewählt wurden
                stehen=stehen+1
            elif(action2==0):
                links=links+1
            # Get next state and reward
            state2, reward2, done, info = env.step(action2) 
            # Discretize state2
            state2_adj = (state2 - env.observation_space.low)*array
            state2_adj = np.round(state2_adj, 0).astype(int)
            
            if done and state2[0] >= 0.5:
                goal_reached2+=1
                                
            # Update 
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
    print("Rechts ausgewählt:",rechts)
    print("Stehen bleiben ausgewählt:",stehen)
    print("Links ausgewählt:",links)
    gesamt=rechts+stehen+links
    print("Gesamt:",gesamt)
    return ave_reward_list2                     #Ausgabe der, über jeweils 100 Episoden durchschnittlichen, Rewards im Verlaufe der gesamten Episodenanzahl 
# Run Q-learning algorithm
Qtable,rewards = QLearning(env, 0.1, 0.9, 0.8, 0, 10000)#learning, discount, epsilon, min_eps, episodes
plt.plot(100*(np.arange(len(rewards)) + 1), rewards)

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
