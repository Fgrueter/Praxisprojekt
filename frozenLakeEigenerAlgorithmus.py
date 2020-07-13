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
    move=random.choice(["Left","Down","Right","Up"])   # No informations-> random action
    cur_state,reward,done,prob=env.step(actions[move])
    
    
    for x in range(rows):                   #If the goal was found before, save position, to move in the direction to the goal
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
        elif(alreadyDone):                                # No holes and we now where the goal is-> we move in the direction of the goal
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