# Dies ist ein Abänderung des Codes von dem Leadboard von OpenAI Gym zu der FrozenLake4x4 Umgebung des Nutzers Nitish tom michael für die Frozenlake8x8 Umgebung
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
            counter2=counter2-1
            env.reset()
            break
        if new_state==63:
            bool=False
            break
#print("no.of attempts",counter)
print(counter2)
print("the winning sequence",winning_sequence)