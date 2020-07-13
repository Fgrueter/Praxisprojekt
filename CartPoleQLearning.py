# In[ ]:
import numpy as np
import gym
import matplotlib.pyplot as plt


env = gym.make('CartPole-v1')
env.reset()
print('State space: ', env.observation_space)
print('Action space: ', env.action_space)

print(env.observation_space.low)
print(env.observation_space.high)

# Import and initialize Mountain Car Environment

# Define Q-learning function
def QLearning(env, learning, discount, epsilon, min_eps, episodes):
    # Determine size of discretized state space
    
    array=np.array([20,10,20,30])
    num_states = np.array([env.observation_space.high[0]- env.observation_space.low[0],20,env.observation_space.high[2]- env.observation_space.low[2],20])*array
    num_states = np.round(num_states, 0).astype(int) + 1
  #  num_states2 = (env.action_space.high - env.action_space.low)*np.array([10])
 #   num_states2 = np.round(num_states2, 0).astype(int) + 1
    
#Testvariable
    goal_reached=0
    
    # Initialize Q table
    Q = np.random.uniform(low = -1, high = 1,size =(num_states[0],num_states[1],num_states[2],num_states[3],env.action_space.n))
    # Initialize variables to track rewards
    # Q 19x15x3=285 Möglichkeiten
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
        state_adj = np.array([state[0]- env.observation_space.low[0],state[1]-10,state[2]-env.observation_space.low[2],state[3]-10])*array
        state_adj = np.round(state_adj, 0).astype(int)
        while done != True:   
            # Render environment for last five episodes
            if i >= (episodes - 19):
               env.render()
                    
            # Determine next action - epsilon greedy strategy
            if np.random.random() < 1 - epsilon:
                action = np.argmax(Q[state_adj[0], state_adj[1],state_adj[2],state_adj[3]]) #Best Mögliche Aktion für den Wert
            else:
                action =np.random.randint(0, env.action_space.n)
                
            # Get next state and reward
            state2, reward, done, info = env.step(action) 
            
            # Discretize state2
            state2_adj = (state2 - (env.observation_space.low[0],-10,env.observation_space.low[2],-10))*array
            state2_adj = np.round(state2_adj, 0).astype(int)
            
            #Allow for terminal states
            if done and (i+1) % 100 == 0 and np.mean(reward_list)>=195.0:
                Q[state_adj[0], state_adj[1],state_adj[2],state_adj[3], action] = reward
                goal_reached+=1     
                break;
            # Adjust Q value for current state
            else:
                delta = learning*(reward + discount*np.max(Q[state2_adj[0],state2_adj[1],state2_adj[2],state2_adj[3]])- Q[state_adj[0], state_adj[1],state_adj[2],state_adj[3],action])
                Q[state_adj[0], state_adj[1],state_adj[2],10,action] += delta
               # print(Q[state_adj[0], state_adj[1],action])
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

    return ave_reward_list

# Run Q-learning algorithm
rewards = QLearning(env, 0.2, 0.9, 0.8, 0, 10000)#learning, discount, epsilon, min_eps, episodes

#Plot Rewards
plt.plot(100*(np.arange(len(rewards)) + 1), rewards)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Episodes')
plt.show()


