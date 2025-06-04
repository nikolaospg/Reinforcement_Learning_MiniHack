import numpy as np
from commons import AbstractAgent
import minihack_env as me
from gym_exp import RLTask

### FILE IN WHICH I EXPERIMENT WITH OUR MINIHACK ENVIRONMENTS, USING A FIXED AGENT ### 


class FixedAgent(AbstractAgent):
    def __init__(self, id, action_space):
        super().__init__(id, action_space)                  # calls AbstractAgent's constructor

        self.down_blocked_flag = 0                          #if the down direction has been blocked once, then it is set true and the agent always goes right

    def act(self, state=None, reward=0):                    #The state on which the fixed agent acts is ("chars" option, 21x79 array with characters)
        
        BLOCKING_CODES = {                  #I model these as blocking (i.e. the agent cannot go further), for the codes check minihack_env.py documentation
            125,                               #Lava
            32,                                #blank
            45,                                # horizontal wall
            124,                                #vertical wall   
            35                                  #tree
        }


        #detecting agent and down element#
        agent_loc = np.argwhere(state == ord('@'))              #finding the agent!
        if(len(agent_loc)!=1):
            print("The location of Fixed Agent has not been found exactly 1 times, exiting!")
            exit()
        agent_loc = agent_loc[0]
        down_loc = agent_loc + np.array([1,0])
        #Finished detecting the down element#

        #Detecting whether going down is not blocked: 
        if (down_loc[0] >= 21 or state[tuple(down_loc)] in BLOCKING_CODES):
            # print("detected blocking element!")
            self.down_blocked_flag = 1
        
        #Taking down only if we have never been blocked 
        UP = 0; RIGHT = 1; DOWN = 2; LEFT = 3
        if(self.down_blocked_flag == 0):
            action = DOWN
        else:
            action = RIGHT
        # print("action chosen", action)

        return action
                


def main():

    
    #Initialisation of objects, setting variables 
    size = 5
    id = me.ROOM_WITH_MONSTER                      #EMPTY_ROOM, ROOM_WITH_LAVA, ROOM_WITH_MONSTER, ROOM_WITH_MULTIPLE_MONSTERS, CLIFF
    my_env = me.get_minihack_envirnment(id, size=size, add_pixel=True, random=False)
    my_agent = FixedAgent("agent", my_env.action_space)
    my_RLTask = RLTask(my_env, my_agent, minihack_id = id)

    my_RLTask.visualize_episode_minihack(10)


if __name__ == "__main__":
    main()

