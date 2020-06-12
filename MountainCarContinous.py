import numpy as np
import gym
import matplotlib.pyplot as plt

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
    num_states2 = (env.action_space.high - env.action_space.low)*np.array([1])
    num_states2 = np.round(num_states2, 0).astype(int)+1 #Array mit Größe 3
#Testvariablen
    early_peak=0
    high_peak=200
    goal_reached=0
    total_reward=0
    total_reward_after_peak=0
    
    # Initialize Q table
    Q = np.random.uniform(low = -1, high = 1, 
                          size = (num_states[0], num_states[1], 
                                  num_states2[0]))
    # Initialize variables to track rewards
    reward_list = []
    ave_reward_list = []
    # Calculate episodic reduction in epsilon
    reduction = (epsilon - min_eps)/episodes
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
            #if i >= (episodes - 19):
            #   env.render()
                
            # Determine next action - epsilon greedy strategy
            if np.random.random() < 1 - epsilon:
                action = np.argmax(Q[state_adj[0], state_adj[1]]) #Best Mögliche Aktion für den Wert
                action=action*np.array([1])-1
               # print(action)
            else:
                action = np.random.randint(env.action_space.low,env.action_space.high)
                
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
                                 Q[state_adj[0], state_adj[1],action+1])
                Q[state_adj[0], state_adj[1],action+1] += delta
            # Update variables
            tot_reward += reward
            state_adj = state2_adj
            
            counter+=1
            total_reward+=reward
            if(i>=early_peak and early_peak>0):
                total_reward_after_peak+=reward
            
            
                
        # Decay epsilon
     #   if epsilon > min_eps:
     #       epsilon -= reduction
        
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
rewards = QLearning(env, 0.2, 0.9, 1, 0, 1500)#learning, discount, epsilon, min_eps, episodes

#Plot Rewards
plt.plot(100*(np.arange(len(rewards)) + 1), rewards)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Episodes')
plt.show()
