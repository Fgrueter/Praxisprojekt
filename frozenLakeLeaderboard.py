# In[63]:

# Dieser Code entstammt dem Leaderboard von OpenAI Gym https://github.com/openai/gym/wiki/Leaderboard und wurde von dem Nutzer Nitish tom michael verfasst.
#Der Code wurde durch eine Variable Counter2 ergänzt, um die Anzahl der Aktionen zu zählen.

import gym
import random
env = gym.make("FrozenLake-v0")
env.reset()
env.render()
reward=0.00

forbidden=[5,7,11,12]           #Hier werden die Löcher gespeichert

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
    for a in winning_sequence:                                          #Es werden die vier zuvor ausgewählten Aktionen hintereinander ausgeführt
        new_state, reward, done, info = env.step(actions[a])
        counter2=counter2+1
        env.render()
        print("Reward: {:.2f}".format(reward))
        if new_state in forbidden:
            counter2=counter2-1                             #Wenn man an diesem Punkt ist, hat man sich nicht bewegt, aber der Counter wurde trotzdem erhöht
            env.reset()
            break
        if new_state==15:                                   #Zielpunkt
            bool=False
            break
#print("no.of attempts",counter)
print(counter2)
print("the winning sequence",winning_sequence)




