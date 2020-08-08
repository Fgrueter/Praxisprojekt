import gym
env = gym.make("FrozenLake-v0")
env.reset()

l=0             
u=0                 #Das sind die möglichen Ausgänge, wenn man nach links geht
d=0
for i in range(10000):
      new_state, reward, done, info = env.step(0)  #Links
      if new_state==0:
          l=l+1                 #Da beim Start nach links oder oben gegen eine Wand gegangen wird, und man dann an der Startposition bleibt
          u=u+1                 #Sind die beiden Positionen nicht zu unterscheiden
      elif new_state==4:
          d=d+1
      env.reset()    
print("Links:",l/2,"Oben:",u/2,"Unten:",d)  #Da links und oben bei beiden erhöht werden, wird es durch zwei geteilt. 
print("Linksunterschied:",l/2-(10000/3),"Obenunterschied:",u/2-(10000/3),"Untenunterschied:",d-(10000/3))  

l=0
r=0
d=0
for i in range(10000):
      new_state, reward, done, info = env.step(1)  #Unten
      if new_state==0:
          l=l+1
      elif new_state==1:
          r=r+1                 #Hier sind alle Ergebnisse klar zuordbar-> Ergebnisse zeigen:Alle Richtungen ungefähr gleich oft ausgewählt
      elif new_state==4:
          d=d+1
      env.reset()     
print("Links:",l,"Rechts:",r,"Unten:",d)     
print("Linksunterschied:",l-(10000/3),"Rechtsunterschied:",r-(10000/3),"Untenunterschied:",d-(10000/3))     

r=0
u=0
d=0
for i in range(10000):
      new_state, reward, done, info = env.step(2) #Rechts
      if new_state==0:
          u=u+1
      elif new_state==1:
          r=r+1    
      elif new_state==4:
          d=d+1
      env.reset()       
print("Rechts:",r,"Oben:",u,"Unten:",d)     
print("Rechtsunterschied:",r-(10000/3),"Obenunterschied:",u-(10000/3),"Untenunterschied:",d-(10000/3))     

l=0
r=0
u=0
for i in range(10000):
      new_state, reward, done, info = env.step(3) #Oben
      if new_state==0:
          l=l+1
          u=u+1
      elif new_state==1:
          r=r+1    
      env.reset()     
print("Links:",l/2,"Rechts:",r,"Oben:",u/2)     
print("Linksunterschied:",l/2-(10000/3),"Rechtsunterschied:",r-(10000/3),"Obenunterschied:",u/2-(10000/3))     
