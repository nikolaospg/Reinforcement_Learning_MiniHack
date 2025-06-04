import numpy as np
import gymnasium as gym 
from commons import AbstractAgent, AbstractRLTask, get_crop_pixel_from_observation
import matplotlib.pyplot as plt 


### FILE IN WHICH I EXPERIMENT WITH GYMNASIUM. I IMPLEMENT A SIMPLE (SQUARE GRID WITH GOAL) ENVIRONMENT AND RUN A SIMPLE (RANDOM) AGENT ON IT ###


#I will implement the nxm grid based environment! The rules are:
#The environment has no walls, we begin from (0,0) and try to reach (n-1,m-1). Every action gives -1 reward. If we take an action that would bring of the grid then we just don't move but still get the reward.
#We will just 1)inherit the gym.Env class, 2)define self.action_space and self.observation_space, 3) implement reset() step() and render().
class Env11(gym.Env):   

    def __init__(self, n: int, m: int):

        #Setting attributes#
        self.n = n
        self.m = m
        #Finished Setting attributes#

        #Defining agent and target location setting initial values- specific according to the description.#
        #Using _name convention to show soft encapsulation privacy:
        self._agent_location   = np.array([0, 0], dtype=np.int32)         
        self._target_location  = np.array([self.n -1, self.m -1], dtype=np.int32)
        #Finished defining initial and target location#

        #Defining action_space and observation_space#
        #They are defined with specific gym.spaces ! 
        self.action_space = gym.spaces.Discrete(4)          #The four moves, with the way they should be defined. It is A wrapper over range(n) with functions .sample(), .contains(x) and .n
        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # down
        }

        self.observation_space = gym.spaces.Dict({"agent_location": gym.spaces.Box(low=np.array([0, 0]),high=np.array([n-1, m-1]), dtype=np.int32)})    #can also similarly add "target_loc" if it is not given!
        #Finished defining action_space and observation_space#

    #Resetting _agent_location and returning just the observation (a dictionary with "agent_location" as we defined), empty dictionary for info.
    def reset(self):
        self._agent_location = np.array([0, 0], dtype=np.int32)
        return {"agent_location": self._agent_location}, {}                     

    #should take step according to action_key, do whatever is needed and return observation, reward, terminated, truncated, info
    def step(self, action):

        #Taking action, finding new agent location:#
        action_direction = self._action_to_direction[action]
        if(self._agent_location[0]+action_direction[0]<0 or self._agent_location[0]+action_direction[0]>self.n-1):
            next_x = self._agent_location[0]
        else:
            next_x = self._agent_location[0]+action_direction[0]
        if(self._agent_location[1]+action_direction[1]<0 or self._agent_location[1]+action_direction[1]>self.m-1):
            next_y = self._agent_location[1]
        else:
            next_y = self._agent_location[1]+action_direction[1]
        self._agent_location = np.array([next_x, next_y],  dtype=np.int32)
        #Finished finding new location#

        #Setting the values to be returned#
        if(np.array_equal(self._agent_location, self._target_location)):
            terminated = True
        else:
            terminated = False

        truncated = False
        reward = -1                  #"receives a negative reward of -1 for each action"
        info={}
        observation = {"agent_location": self._agent_location}
        #Finished Setting the values to be returned#

        return observation, reward, terminated, truncated, info

    def render(self):
        grid = np.full((self.n, self.m), ".", dtype=str)
        grid[tuple(self._agent_location)]   = 'A'
        grid[tuple(self._target_location)]  = 'T'
        print(grid)



#Implementing a random agent that inherits AbstractAgent and just does random action. Nothing more complex.
class RandomAgent(AbstractAgent):
    def __init__(self, id, action_space):
        super().__init__(id, action_space)                  # calls AbstractAgent's constructor

    def act(self, state=None, reward=0):     
        return self.action_space.sample()                   #Our action space (the one we have in the environment) has the .sample() function to get random action               




#Inheriting the AbstractRL Task, showing how one episode goes (visualise_episode) and how the average return changes for a large amount of episodes (interact) 
class RLTask(AbstractRLTask):
    def __init__(self, env, agent, minihack_id =None):
        super().__init__(env, agent)
        print("Task initialised fine!")
        self.minihack_id = minihack_id

    def interact(self, n_episodes):
        print("\nRunning the RL algorithm for ", n_episodes, "episodes:")

        returns_list      = []
        avg_returns_list  = []
        sum_returns       = 0
        for current_episode in range(n_episodes):
            self.env.reset()
            current_return = 0
            while(True):                            #Running episode until we terminate
                current_action = self.agent.act()
                observation, reward, terminated, truncated, info = self.env.step(current_action)

                current_return += reward
                if(terminated or truncated):
                    break
            sum_returns += current_return
            returns_list.append(current_return)
            avg_returns_list.append(sum_returns/(current_episode+1))
        self.env.reset()
        # print("The returns are:", returns_list)
        # print("The avg returns are", avg_returns_list)
        plt.plot(avg_returns_list)
        plt.title("RL task interaction for %d Episodes Final Avg return=%.3f" % (n_episodes, avg_returns_list[-1]), fontsize=13)
        plt.show()

    def visualize_episode(self, max_number_steps = 10):
        print("\nVisualising One Episode for max steps", max_number_steps)
        print("Initial Game state:")
        self.env.render()
        print("Consecutive States:\n")
        for current_step in range(max_number_steps):
            print("Applying Step ",current_step)
            current_action = self.agent.act()
            observation, reward, terminated, truncated, info = self.env.step(current_action)
            self.env.render()
            if(terminated):
                print("Game terminated on step", current_step)
                break
        self.env.reset()
    

#------------------------------ MINIHACK VARIANTS, FOR TASK 1.2 ----------------------------------#

    def visualize_episode_minihack(self, max_number_steps = 10):
        print("RUNNING %s " % (self.minihack_id))
        print("\nVisualising One Episode for max steps", max_number_steps)

        list_size_to_show = 6              #How many images to show in one figure
        init_observation, _ = self.env.reset()
        im_list = []
        current_im = get_crop_pixel_from_observation(init_observation)
        im_list.append(current_im)

        observation = init_observation
        #Running max_number_steps, keeping images on list. Showing lots of useful information for the episode!!
        for current_step in range(max_number_steps):
            print("Applying Step ",current_step)
            current_action = self.agent.act(state = observation["chars"])
            observation, reward, terminated, truncated, info = self.env.step(current_action)
            print("current reward is ", reward)
            print("end status", info["end_status"])
            if(truncated):
                print("Game truncated on step", current_step)
                break
            if(terminated):
                print("Game terminated on step", current_step)
                break
            current_im = get_crop_pixel_from_observation(observation)
            im_list.append(current_im)
            
            #If the agent dies then show it so that we show teleportation:#
            if(reward == -101):
                im_list = im_list[-list_size_to_show:]
 
                fig, axes = plt.subplots(2, 3, figsize=(20, 6))  
                fig.suptitle("Death on Environment %s," % self.minihack_id, fontsize=16)

                for index, (im, ax) in enumerate(zip(im_list, axes.flat)):
                    ax.imshow(im)
                    ax.set_title("Step = %d" % index)
                    ax.axis('off')

                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.subplots_adjust(hspace=0.15, wspace=0.15)
                plt.show()
                #Finished showing death and teleportation#
                print("\n")
            
        #Showing the last 10 steps before termination-truncation#
        im_list = im_list[-list_size_to_show:]

        fig, axes = plt.subplots(2, 3, figsize=(20, 6))  
        fig.suptitle("Last %d steps, Environment %s" % (list_size_to_show, self.minihack_id), fontsize=16)

        for index, (im, ax) in enumerate(zip(im_list, axes.flat)):
            ax.imshow(im)
            ax.set_title("Step = %d" % index)
            ax.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.subplots_adjust(hspace=0.15, wspace=0.05)
        plt.show()
        self.env.reset()

    # def interact_minihack(self, n_episodes):
    #     print("\nRunning the RL algorithm for ", n_episodes, "episodes:")

    #     returns_list      = []
    #     avg_returns_list  = []
    #     sum_returns       = 0
    #     for current_episode in range(n_episodes):
    #         print("Running Episode", current_episode)
    #         observation, _ = self.env.reset()
    #         current_return = 0
    #         current_step = 0
    #         while(True):                            #Running episode until we terminate
    #             current_action = self.agent.act(state=observation["chars"])
    #             observation, reward, terminated, truncated, info = self.env.step(current_action)

    #             current_return += reward
    #             current_step +=1 
    #             if(terminated or truncated):
    #                 if(truncated):
    #                     print("truncated step",current_step)
    #                 if(terminated):
    #                     print("terminated step",current_step)
    #                 break
    #         print("Episode = %d current return is %d" % (current_episode, current_return))
    #         print("end status", info["end_status"])
    #         print("\n")
            
    #         sum_returns += current_return
    #         returns_list.append(current_return)
    #         avg_returns_list.append(sum_returns/(current_episode+1))
    #     self.env.reset()                                        #defensive programming, resetting again
    #     # print("The returns are:", returns_list)
    #     # print("The avg returns are", avg_returns_list)
    #     plt.plot(avg_returns_list)
    #     plt.title("RL task interaction for %d Episodes Final Avg return=%.3f" % (n_episodes, avg_returns_list[-1]))
    #     plt.show()





def main():


    n = 5
    m = 5
    max_number_steps_visualise = 10
    total_interaction_episodes = 10000


    my_env    = Env11(n,m)
    my_agent  = RandomAgent("agent", my_env.action_space)
    my_RLTask = RLTask(my_env, my_agent)
    print("Running RL Task with n=%d. m=%d" % (n,m))

    #Visualising one episode
    my_RLTask.interact(total_interaction_episodes)
    my_RLTask.visualize_episode(max_number_steps_visualise)

if __name__ == "__main__":
    main()