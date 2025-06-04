import numpy as np
from commons import AbstractAgent, get_crop_chars_from_observation, get_crop_pixel_from_observation
import matplotlib.pyplot as plt 
import minihack_env as me
import random
import time 
import math

class RLAgent(AbstractAgent):

    def __init__(self, id, action_space):
        super().__init__(id, action_space)                  # calls AbstractAgent's constructor. 
        self.num_actions = self.action_space.n              # To get number of actions -> self.action_space.n the actions are just integers!


    ## TO DO! CHANGE SO THAT IT USES DERIVED POLICY! ## -> NVM not needed!!!
    def act(self, state=None, reward=0):                    #The state on which the fixed agent acts is ("chars" option, 21x79 array with characters)
        pass
                

    #Mechanism that gets observation and gives us agent state representation:
    def get_agent_state(self, observation, option = "crop"):

        if(option == "naive"):
            array_repr = observation["chars"].copy()            #I use .copy() to make sure that a new array is allocated, because I want to log them. If they use same array (essentially same pointer) for every observation the content will be overwritten!
        elif(option == "crop"):
            array_repr = get_crop_chars_from_observation(observation).copy()
        
        array_repr[array_repr == 60] = 46                           #get rid of the upward stair case, which appears randomly when Random=True on empty_room (treated as artifact, and causes state space numerosity to explode!)
        # total = np.sum(array_repr == 62)                            #Checking if the number of downwards ladders (goal) is different than one!
        # if(total !=1):
        #     print(total, "total ladders found, should be 1")
            # exit()
       
        hash_repr = array_repr.tobytes()                            # to inverse if i want to np.frombuffer(hash_repr, dtype=np.uint8).reshape(array_repr.shape). Also only works if the shape is constant, if it is not i might have to add it as part of the key like hash_repr = (array_repr.shape, array_repr.tobytes())
        return array_repr, hash_repr

    #Implements eps greedy choice based on estimate Q values 
    def eps_greedy_choice(self, current_epsilon, current_state_hash, Q):
        
        action_values = Q.setdefault(current_state_hash, np.zeros(self.num_actions, dtype=np.float32))          #If the value with this specific key exists just give it. if it does not, initialise with zeros!
        if np.random.rand() < current_epsilon:
            action = np.random.randint(self.num_actions)        # explore: random
        else:
            action = np.argmax(action_values)            # exploit: greedy

        return action

    #I use this function to plot trajectories in cliff for sarsa and q learning agents, showcase the difference
    #method="Q_lear" or "SARSA"
    def plot_trajectory(self, env, Q_est, eps, episode_num, method="Q_lear"):
        
        im_sequence = []
        current_observation, _ = env.reset()            #initial observation
        im_sequence.append(get_crop_pixel_from_observation(current_observation))
        current_state_array, current_state_bytes  = self.get_agent_state(observation=current_observation, option="crop") 

        #Running the episode 
        while(True):                            #Running episode until we terminate or truncate

            if(method=="SARSA"):
                current_action = self.eps_greedy_choice(eps, current_state_bytes, Q_est)
            else:
                current_action = self.eps_greedy_choice(0, current_state_bytes, Q_est)              #getting the actual greedy choice, no eps
            

            current_observation, current_reward, terminated, truncated, info = env.step(current_action)
            im_sequence.append(get_crop_pixel_from_observation(current_observation))
            current_state_array, current_state_bytes  = self.get_agent_state(observation=current_observation, option="crop")
            if(terminated or truncated):
                break
        #finished running the episode


        #Now plotting the whole sequence, The user will check which frames to include in the final figure
        n = len(im_sequence)
        cols = min(6, n)                      # up to 6 frames per row
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(2.2*cols, 2.2*rows))
        if rows == 1:
            axes = np.atleast_1d(axes)        # make iterable even for a single row

        for idx, img in enumerate(im_sequence):
            r, c = divmod(idx, cols)
            ax = axes[r][c] if rows > 1 else axes[c]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"t={idx}", fontsize=8)

        # hide any empty subplots
        for ax in axes.flat[n:]:
            ax.axis('off')

        fig.suptitle(f"{method} trajectory at episode {episode_num}", fontsize=14)
        print("Note down the steps you will want to be shown separately!")
        plt.tight_layout()
        plt.show()
        #Finished Plotting the whole trajectory
        
        #Now plotting the frames specified 
        # Ask user for specific step indices
        step_input = input("Give a sequence of step numbers you want showed (e.g., 0 5 10 15 20 25): ")
        try:
            step_indices = list(map(int, step_input.strip().split()))
        except ValueError:
            print("Invalid input. Expected space-separated integers.")
            return
        print("I got step indices", step_indices)
        m = len(step_indices)
        if m == 0:
            print("No steps selected.")
            return

        rows = 2
        cols = math.ceil(m / rows)

        fig, axes = plt.subplots(rows, cols, figsize=(20, 6))
        fig.suptitle(f"Selected steps for {method} at episode {episode_num}", fontsize=16)

        for i, step in enumerate(step_indices):
            if step >= len(im_sequence):
                print(f"Step {step} is out of range.")
                continue
            r, c = divmod(i, cols)
            ax = axes[r][c] if rows > 1 else axes[c]
            ax.imshow(im_sequence[step])
            ax.set_title(f"Step = {step}")
            ax.axis('off')

        for ax in axes.flat[m:]:
            ax.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.subplots_adjust(hspace=0.15, wspace=0.05)
        plt.show()

    #Implementing on policy first visit monte carlo control, with eps greedy choice, incremental updates based on alpha (learning rates) and linear adaptation of eps.
    #following sutton barto, second edition 101 page
    def MC_control(self, env, eps_start, eps_end, alpha, num_episodes, gamma):
        t1=time.time()
        print("running MC control! e_start=%.3f e_end=%.3f alpha=%.3f episodes=%d gamma=%.3f" % (eps_start, eps_end, alpha, num_episodes, gamma))
        #Initialisation hash maps for policy, Q and Returns#
        MC_policy       = {}                      #hash table, key is the space representation and values are integers showing the corresponding action
        Q               = {}                      #This will be a hash table, where key will be state representation and value will be 1D array with Q values for every discrete action!
        episode_returns = []
        episode_disc_returns =[]                    #The discounted and undiscounted (exactly above) returns in each episode, saved in lists
        #Finished initialisations#

        for current_episode in range(num_episodes):

            #First initialising stuff for this specific episode#
            # print("running episode", current_episode)
            state_sequence  = []
            action_sequence = []
            reward_sequence = []
            G               =  0                        #the G variable used in book
            eps = eps_start  - current_episode*(eps_start-eps_end)/(num_episodes-1)                         
            #finished initialising for this specific episode
            

            #Generating the episode by exploring with e greedy based on current Q estimates#
            current_observation, _ = env.reset()            #initial observation
            current_state_array, current_state_bytes  = self.get_agent_state(observation=current_observation, option="crop")       
            state_array_shape = current_state_array.shape               # I will later check if it is constant! if not i have to change the byte representation
            state_sequence.append(current_state_bytes)
            self.state_array_shape = state_array_shape
            while(True):                            #Running episode until we terminate or truncate
                current_action = self.eps_greedy_choice(eps, current_state_bytes, Q)
                action_sequence.append(current_action)

                current_observation, current_reward, terminated, truncated, info = env.step(current_action)
                reward_sequence.append(current_reward)
                if(terminated or truncated):
                    break
                    
                current_state_array, current_state_bytes  = self.get_agent_state(observation=current_observation, option="crop")
                if not np.array_equal(state_array_shape, current_state_array.shape):
                    print(current_state_array)
                    raise Exception("The state_array_shape are not constant!")
                state_sequence.append(current_state_bytes)
            
            #Now getting the discounted and undiscounted rewards
            undiscounted = sum(reward_sequence)
            discounted = 0.0
            gamma_pow = 1.0
            for r in reward_sequence:          # forward pass
                discounted += gamma_pow * r
                gamma_pow *= gamma
            episode_returns.append(undiscounted)
            episode_disc_returns.append(discounted)
            if(current_episode%25 ==0 or current_episode==num_episodes-1):
                print("current_episode=%d, discounted=%.3f, undiscounted%.3f"  % (current_episode, discounted, undiscounted))
                # print(current_state_array)
                print("Number of unique states visited:", len(Q), "\n")
            #Finished generating the episode!#


            #Now i will backtrack on this episode to implement the incremental estimation of the average return which will help update the q values
            visited_state_actions = set()           # I will keep a set of state-action pairs! the state is passed in the hashable byte version! This offers fast lookup! 
            T = len(reward_sequence)
            for t in range(T-1, -1, -1):         #Doing the loop like in sutton barto book
                G = gamma*G + reward_sequence[t]
                
                current_state  = state_sequence[t]
                current_action = action_sequence[t]
                if (current_state, current_action) not in visited_state_actions:
                    visited_state_actions.add((current_state, current_action))
                    corresponding_q_value = Q[current_state][current_action]
                    Q[current_state][current_action] = corresponding_q_value + alpha * (G - corresponding_q_value)          #Implementing the incremental update with the alpha!!!!
            #Finished backtracking on this episode and updating Q value estimations!#
        
        #Now doing policy improvement (essentially getting the final greedy policy)
        
        for s in Q:
            MC_policy[s] = int(np.argmax(Q[s]))
        
        self.MC_policy = MC_policy
        t2=time.time()
        print("Total time for MC control ", t2-t1)
        return episode_disc_returns

    #Implementing on policy TD control (SARSA), with eps greedy choice, incremental updates based on alpha (learning rates) and linear adaptation of eps.
    #Following sutton barto, second edition 130 page
    def SARSA_control(self, env, eps_start, eps_end, alpha, num_episodes, gamma, env_pixel=None):
        t1=time.time()
        print("running SARSA control! e_start=%.3f e_end=%.3f alpha=%.3f episodes=%d gamma=%.3f" % (eps_start, eps_end, alpha, num_episodes, gamma))
        #Initialisation hash maps for policy, Q and Returns#
        SARSA_policy       = {}                      #hash table, key is the space representation and values are integers showing the corresponding action
        Q                  = {}                      #This will be a hash table, where key will be state representation and value will be 1D array with Q values for every discrete action!
        episode_returns    = []
        episode_disc_returns =[]                    #The discounted and undiscounted (exactly above) returns in each episode, saved in lists
        #Finished initialisations#

        for current_episode in range(num_episodes):
            if((current_episode==500 or current_episode==num_episodes-1) and env_pixel is not None):
                self.plot_trajectory(env_pixel, Q, eps, current_episode, method="SARSA")
            #First initialising stuff for this specific episode#
            #keeping the sequences, like i did in MC, for logging purposes
            state_sequence  = []                    
            action_sequence = []
            reward_sequence = []
            eps = eps_start  - current_episode*(eps_start-eps_end)/(num_episodes-1)   

            #Getting initial S:
            current_observation, _ = env.reset()            #initial observation
            current_state_array, current_state_bytes  = self.get_agent_state(observation=current_observation, option="crop")       
            state_array_shape = current_state_array.shape               # I will later check if it is constant! if not i have to change the byte representation
            self.state_array_shape = state_array_shape
            state_sequence.append(current_state_bytes)

            #Choosing initial action (e greedy based on the Q values):
            current_action = self.eps_greedy_choice(eps, current_state_bytes, Q)
            action_sequence.append(current_action)
            #finished initialising for this specific episode#
            
            
            #Now running the trajectory and doing updates for this specific episode#
            while(True):                            #Running episode until we terminate or truncate
                
                #First taking previous action and observing result:
                current_observation, current_reward, terminated, truncated, info = env.step(current_action)
                reward_sequence.append(current_reward)

                
                #Converting the observation to state
                next_state_array, next_state_bytes  = self.get_agent_state(observation=current_observation, option="crop")
                if not np.array_equal(state_array_shape, next_state_array.shape) and not (terminated or truncated):
                    raise Exception("The state_array_shape are not constant!")
                state_sequence.append(next_state_bytes)

                #Now choosing the next action, again e greedy through Q
                next_action = self.eps_greedy_choice(eps, next_state_bytes, Q)
                action_sequence.append(next_action)

                #Finally doing update in the Q estimates!
                current_pair_Q  =  Q[current_state_bytes][current_action]
                next_state_action_vals = Q.setdefault(next_state_bytes, np.zeros(self.num_actions, dtype=np.float32))               #in case this specific hash key has not yet been observed
                next_pair_Q     =  next_state_action_vals[next_action]

                Q[current_state_bytes][current_action]  =  current_pair_Q + alpha*(current_reward + gamma*next_pair_Q - current_pair_Q)         #SARSA update equation

                #Resetting the current state/action <-next state/action
                if(terminated or truncated):                #The terminal check must be done in the end!
                    break   
                current_state_array = next_state_array
                current_state_bytes = next_state_bytes
                current_action      = next_action


            #Finished running trajectory and doing updates for this episode#

            #Getting the discounted and undiscounted returns, useful for plots etc:
            undiscounted = sum(reward_sequence)
            discounted = 0.0
            gamma_pow = 1.0
            for r in reward_sequence:          # forward pass
                discounted += gamma_pow * r
                gamma_pow *= gamma
            episode_returns.append(undiscounted)
            episode_disc_returns.append(discounted)
            if(current_episode%25 ==0 or current_episode==num_episodes-1):
                print("current_episode=%d, discounted=%.3f, undiscounted%.3f"  % (current_episode, discounted, undiscounted))
                # print(current_state_array)
                print("Number of unique states visited:", len(Q), "\n")

        #Now doing policy improvement (essentially getting the final greedy policy)
        for s in Q:
            SARSA_policy[s] = int(np.argmax(Q[s]))
        
        self.SARSA_policy = SARSA_policy
        t2=time.time()
        print("Total time for SARSA control ", t2-t1)
        return episode_disc_returns

    #Implementing off policy TD control (Tabular Q learning), with eps greedy choice, incremental updates based on alpha (learning rates) and linear adaptation of eps.
    #Following sutton barto, second edition 131 page
    def Q_learning_control(self, env, eps_start, eps_end, alpha, num_episodes, gamma, env_pixel=None):
        t1=time.time()
        print("running Q learning control! e_start=%.3f e_end=%.3f alpha=%.3f episodes=%d gamma=%.3f" % (eps_start, eps_end, alpha, num_episodes, gamma))
        #Initialisation hash maps for policy, Q and Returns#
        tabular_Q_policy     = {}                      #hash table, key is the space representation and values are integers showing the corresponding action
        Q                    = {}                      #This will be a hash table, where key will be state representation and value will be 1D array with Q values for every discrete action!
        episode_returns      = []
        episode_disc_returns =[]
        #Finished initialisations#

        for current_episode in range(num_episodes):
            if((current_episode==500 or current_episode==num_episodes-1) and env_pixel is not None):
                self.plot_trajectory(env_pixel, Q, eps, current_episode, method="Q_lear")
            #First initialising stuff for this specific episode#
            #keeping the sequences, like i did in MC, for logging purposes
            state_sequence  = []                    
            action_sequence = []
            reward_sequence = []
            eps = eps_start  - current_episode*(eps_start-eps_end)/(num_episodes-1)   

            #Getting initial S:
            current_observation, _ = env.reset()            #initial observation
            current_state_array, current_state_bytes  = self.get_agent_state(observation=current_observation, option="crop")       
            state_array_shape = current_state_array.shape               # I will later check if it is constant! if not i have to change the byte representation
            state_sequence.append(current_state_bytes)
            #Finished initialising stuff for this specific episode
            
            
            #Now running the trajectory and doing updates for this specific episode#
            while(True):                            #Running episode until we terminate or truncate
                
                #First choosing action e greedy based on Q estimated vals:
                current_action = self.eps_greedy_choice(eps, current_state_bytes, Q)
                action_sequence.append(current_action)

                #Taking action and observing result from environment:
                current_observation, current_reward, terminated, truncated, info = env.step(current_action)
                reward_sequence.append(current_reward)
                
                #Converting the observation to state:
                next_state_array, next_state_bytes  = self.get_agent_state(observation=current_observation, option="crop")
                if not np.array_equal(state_array_shape, next_state_array.shape) and not (terminated or truncated):
                    print(current_state_array)
                    raise Exception("The state_array_shape are not constant!")
                state_sequence.append(next_state_bytes)


                #Finally doing update in the Q estimates!:
                current_pair_Q  =  Q[current_state_bytes][current_action]
                next_state_action_vals = Q.setdefault(next_state_bytes, np.zeros(self.num_actions, dtype=np.float32))               #in case this specific hash key has not yet been observed
                max_pair_Q     =  np.max(next_state_action_vals)

                Q[current_state_bytes][current_action]  =  current_pair_Q + alpha*(current_reward + gamma*max_pair_Q - current_pair_Q)         #Tabular Q learning update equation

                #Resetting the current state <-next state:
                if(terminated or truncated):
                    break
                current_state_array = next_state_array
                current_state_bytes = next_state_bytes

            #Finished running trajectory and doing updates for this episode#

            #Getting the discounted and undiscounted returns, useful for plots etc:
            T = len(reward_sequence)
            undiscounted = sum(reward_sequence)
            discounted = 0.0
            gamma_pow = 1.0
            for r in reward_sequence:          # forward pass
                discounted += gamma_pow * r
                gamma_pow *= gamma
            episode_returns.append(undiscounted)
            episode_disc_returns.append(discounted)
            if(current_episode%25 ==0 or current_episode==num_episodes-1):
                print("current_episode=%d, discounted=%.3f, undiscounted%.3f"  % (current_episode, discounted, undiscounted))
                # print(current_state_array)
                print("Number of unique states visited:", len(Q), "\n")

        #Now doing policy improvement (essentially getting the final greedy policy)
        for s in Q:
            tabular_Q_policy[s] = int(np.argmax(Q[s]))
        
        self.tabular_Q_policy = tabular_Q_policy
        t2=time.time()
        print("Total time for Q learning control ", t2-t1)
        return episode_disc_returns

    #Implementing Dyna Q control, with eps greedy choice, incremental updates based on alpha (learning rates) and linear adaptation of eps.
    #Again following sutton barto
    def dyna_Q_control(self, env, eps_start, eps_end, alpha, num_episodes, gamma, n):
        t1=time.time()
        print("running Q learning control! e_start=%.3f e_end=%.3f alpha=%.3f episodes=%d gamma=%.3f" % (eps_start, eps_end, alpha, num_episodes, gamma))
        #Initialisation hash maps for policy, Q and Returns#
        dyna_Q_policy     = {}                      #hash table, key is the space representation and values are integers showing the corresponding action
        Q                 = {}                      #This will be a hash table, where key will be state representation and value will be 1D array with Q values for every discrete action!
        model             = {}                      #hash table that models the environment. the key is a tuple (state_bytes_representation, action_integer) and the value is a tuple of (state_bytes_representation, reward_float)
        episode_returns      = []
        episode_disc_returns =[]
        #Finished initialisations#

        for current_episode in range(num_episodes):
            #First initialising stuff for this specific episode#
            #keeping the sequences, like i did in MC, for logging purposes
            state_sequence  = []                    
            action_sequence = []
            reward_sequence = []
            eps = eps_start  - current_episode*(eps_start-eps_end)/(num_episodes-1)   

            #Getting initial S:
            current_observation, _ = env.reset()            #initial observation
            current_state_array, current_state_bytes  = self.get_agent_state(observation=current_observation, option="crop")       
            state_array_shape = current_state_array.shape               # I will later check if it is constant! if not i have to change the byte representation
            state_sequence.append(current_state_bytes)
            #Finished initialising stuff for this specific episode
            
            
            #Now running the trajectory and doing updates for this specific episode#
            while(True):                            #Running episode until we terminate or truncate
                
                #First choosing action e greedy based on Q estimated vals:
                current_action = self.eps_greedy_choice(eps, current_state_bytes, Q)
                action_sequence.append(current_action)

                #Taking action and observing result from environment:
                current_observation, current_reward, terminated, truncated, info = env.step(current_action)
                reward_sequence.append(current_reward)
                
                #Converting the observation to state:
                next_state_array, next_state_bytes  = self.get_agent_state(observation=current_observation, option="crop")
                if not np.array_equal(state_array_shape, next_state_array.shape) and not (terminated or truncated):
                    print(current_state_array)
                    raise Exception("The state_array_shape are not constant!")
                state_sequence.append(next_state_bytes)


                #Finally doing update in the Q estimates (and the model)!:
                current_pair_Q  =  Q[current_state_bytes][current_action]

                next_state_action_vals = Q.setdefault(next_state_bytes, np.zeros(self.num_actions, dtype=np.float32))               #in case this specific hash key has not yet been observed
                max_pair_Q     =  np.max(next_state_action_vals)

                Q[current_state_bytes][current_action]  =  current_pair_Q + alpha*(current_reward + gamma*max_pair_Q - current_pair_Q)         #Tabular Q learning update equation

                model[(current_state_bytes, current_action)] = (next_state_bytes, current_reward)
                
                #n dyna q steps
                keys_list = list(model.keys())
                for current_dyna_step in range(n):
                    simulated_s, simulated_a = random.choice(keys_list)             #i just pick a key -> an observed state/action pair
                    simulated_next_s, simulated_r = model[(simulated_s, simulated_a)]

                    current_pair_Q  =  Q[simulated_s][simulated_a]
                    max_sim_pair_Q  =  np.max(Q.setdefault(simulated_next_s, np.zeros(self.num_actions, dtype=np.float32)))
                    Q[simulated_s][simulated_a]  =  current_pair_Q + alpha*(simulated_r + gamma*max_sim_pair_Q - current_pair_Q)
                    

                #Resetting the current state <-next state:
                if(terminated or truncated):
                    break
                current_state_array = next_state_array
                current_state_bytes = next_state_bytes




            #Finished running trajectory and doing updates for this episode#

            #Getting the discounted and undiscounted returns, useful for plots etc:
            T = len(reward_sequence)
            undiscounted = sum(reward_sequence)
            discounted = 0.0
            gamma_pow = 1.0
            for r in reward_sequence:          # forward pass
                discounted += gamma_pow * r
                gamma_pow *= gamma
            episode_returns.append(undiscounted)
            episode_disc_returns.append(discounted)
            if(current_episode%25 ==0 or current_episode==num_episodes-1):
                print("current_episode=%d, discounted=%.3f, undiscounted%.3f"  % (current_episode, discounted, undiscounted))
                # print(current_state_array)
                print("Number of unique states visited:", len(Q), "\n")
        #Now doing policy improvement (essentially getting the final greedy policy)
        for s in Q:
            dyna_Q_policy[s] = int(np.argmax(Q[s]))
        
        self.dyna_Q_policy = dyna_Q_policy

        t2=time.time()
        print("Total time for Dyna q control ", t2-t1)
        return episode_disc_returns
                


def plot_returns(all_disc_returns_lists, legends, title):


    #1) raw returns (skip first 50) 
    plt.figure(figsize=(10, 5))
    for disc_returns in all_disc_returns_lists:
        plt.plot(disc_returns[50:])
    plt.xlabel("Episode")
    plt.ylabel("Return (Gₖ)")
    plt.title(f"{title} — Actual Returns (from Episode 50 Onwards)")
    plt.legend(legends)
    plt.grid(True)
    plt.tight_layout()
    # plt.show()

    #2) running average
    plt.figure(figsize=(10, 5))
    for disc_returns in all_disc_returns_lists:
        G_hat = np.cumsum(disc_returns) / (np.arange(len(disc_returns)) + 1)
        plt.plot(G_hat)
    plt.xlabel("Episode")
    plt.ylabel("Running Avg Return (Ĝₖ)")
    plt.title(f"{title} — Running Average Returns")
    plt.legend(legends)
    plt.grid(True)
    plt.tight_layout()
    # plt.show()

    #3) 25-episode rolling average 
    window = 25
    plt.figure(figsize=(10, 5))
    for disc_returns in all_disc_returns_lists:
        # rolling mean with a width of 25 episodes
        if len(disc_returns) >= window:
            kernel = np.ones(window) / window
            rolling = np.convolve(disc_returns, kernel, mode="valid")
            plt.plot(np.arange(window-1, len(disc_returns)), rolling)
        else:
            # too few episodes: just plot a flat line at overall mean
            plt.plot([0, len(disc_returns)-1],
                     [np.mean(disc_returns)]*2, linestyle="--")
    plt.xlabel("Episode")
    plt.ylabel("25-Episode Avg Return")
    plt.title(f"{title} — Rolling Average (25 episode window)")
    plt.legend(legends)
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def main():



    #Initialisation of objects, setting variables#
    size = 5
    max_episode_steps = 250 
    e_start, e_end = 0.99, 0.01                       #linear decay scheduling. Set e_start equal to e_end to have constant epsilon
    # e_start, e_end = 0.1, 0.1
    alpha   = 0.1
    num_episodes = 1000
    gamma = 1
    n = 10                               #dyna q reps parameter

    id = me.ROOM_WITH_LAVA                      #EMPTY_ROOM, ROOM_WITH_LAVA, ROOM_WITH_MONSTER, ROOM_WITH_MULTIPLE_MONSTERS, CLIFF
    my_env = me.get_minihack_envirnment(id, size=size, add_pixel=False, random=False, max_episode_steps=max_episode_steps)
    my_agent = RLAgent("agent", my_env.action_space)
    #Finished initialising objects and variables#


    #Run this piece of code to get experiments for MC SARSA Q Learning on the specified environemtn and get ploted cuvres:
    returns1 = my_agent.MC_control(my_env, e_start, e_end, alpha, num_episodes, gamma)
    returns2 =  my_agent.SARSA_control(my_env, e_start, e_end, alpha, num_episodes, gamma)
    returns3 =  my_agent.Q_learning_control(my_env, e_start, e_end, alpha, num_episodes, gamma)
    #returns4 =  my_agent.dyna_Q_control(my_env, e_start, e_end, alpha, num_episodes, gamma, n)
    plot_returns([returns1, returns2, returns3], ["MC", "SARSA", "Q Lear."], "")

    # #Run this piece of code to make a Q learning vs Dyna q comparison!
    # returns1 =  my_agent.Q_learning_control(my_env, e_start, e_end, alpha, num_episodes, gamma)
    # returns2 =  my_agent.dyna_Q_control(my_env, e_start, e_end, alpha, num_episodes, gamma, n)
    # plot_returns([returns1, returns2], ["Q learning", "Dyna Q"], "")

    # # Run this piece of code to get trajectories for SARSA and Q learning on cliff! 
    # my_env_pixel = me.get_minihack_envirnment(id, size=size, add_pixel=True, random=False, max_episode_steps=max_episode_steps)
    # returns1 =  my_agent.SARSA_control(my_env, e_start, e_end, alpha, num_episodes, gamma, my_env_pixel)
    # returns2 =  my_agent.Q_learning_control(my_env, e_start, e_end, alpha, num_episodes, gamma, my_env_pixel)
    # # plot_returns([returns1, returns2], ["SARSA", "Q Lear."], "")

if __name__ == "__main__":
    main()

