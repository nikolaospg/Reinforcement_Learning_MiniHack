import numpy as np
import gymnasium as gym 
from commons import get_crop_chars_from_observation
import minihack_env as me
from classical_RL import RLAgent, plot_returns
import time 


from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback


#Callback to use with model.learn() in order to get episode returns
class EpisodeReturnLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info:
                reward = info["episode"]["r"]
                self.episode_rewards.append(reward)
                print(f"Episode {len(self.episode_rewards)}: return = {reward}")
        return True


#Wrapper class i use in order to crop and flatten the observations (["chars"]), so i can pass as input to MLP implementation
class CropFlattenWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        random_observation, _ = env.reset()                     #getting one random observation to transform and estimate the dimension 
        random_sample = self.observation(random_observation) 
        flat_shape = random_sample.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=flat_shape,
            dtype=np.uint8
        )

    def observation(self, observation):
        if(np.all(observation["chars"] == 0)):                  #special handling of the terminal state! I just keep the top left part, with dimensionality equal to the non terminal cropped
            cropped = observation["chars"][0:self.cropped_shape[0], 0:self.cropped_shape[1]]
        else:                       
            cropped = get_crop_chars_from_observation(observation)
            self.cropped_shape = cropped.shape
        cropped_flat = cropped.copy().flatten().astype(np.uint8) 
        return  cropped_flat
    

def main():


    #Initialisation of objects, setting variables #
    size = 7
    max_episode_steps = 250 
    e_start, e_end = 0.99, 0.01                       #linear decay scheduling. Set e_start equal to e_end to have constant epsilon
    alpha   = 0.1
    num_episodes = 100
    gamma = 1
    n = 10      

    id = me.ROOM_WITH_MULTIPLE_MONSTERS                      #EMPTY_ROOM, ROOM_WITH_LAVA, ROOM_WITH_MONSTER, ROOM_WITH_MULTIPLE_MONSTERS, CLIFF
    my_env = me.get_minihack_envirnment(id, size=size, add_pixel=False, random=False, max_episode_steps=max_episode_steps)
    my_agent = RLAgent("agent", my_env.action_space)
    my_env = my_env                             
    #Finished with the initial objects, variables etc#

    #Working with tabular Q learning (to compare with deep RL#
    Q_learning_returns =  my_agent.Q_learning_control(my_env, e_start, e_end, alpha, num_episodes, gamma)
    #Finished with tabular q learning#

    #Working with Deep Q network#
    policy_kwargs = dict(
        net_arch=[256, 128]                             #my architecture
    )
    my_env = CropFlattenWrapper(my_env)                #Wrapper so i can use MLP architecture

    model_DQN = DQN(                                                #Set the parameters you want 
        policy="MlpPolicy",       
        env=my_env,               
        learning_rate=5e-4,       
        buffer_size=50000,                      #default value
        learning_starts=1000,                   #default value 
        batch_size=32,                          #default value
        gamma=gamma,                                #this is what we have on the experiments
        train_freq=1,                           #default                           
        target_update_interval=500,             #how often to update the target network, default             
        verbose=0,
        policy_kwargs = policy_kwargs
    )

    callback = EpisodeReturnLogger()                #Callback so that the learning process gives the episode returns
    t1=time.time()
    model_DQN.learn(total_timesteps=7000,  callback=callback)
    t2=time.time()
    print("total time for Deep q learning", t2-t1)
    deep_q_returns = callback.episode_rewards
    #Finished with the Deep Q network#
    
    #PPO Algorithm#
    model_PPO = PPO(
        policy="MlpPolicy",
        env=my_env,                     
        learning_rate=3e-4,             # default
        n_steps=1024,                   # smaller than default due to small map                    
        batch_size=64,                  # mini-batch size used in each epoch
        n_epochs=10,                    # how many times each rollout is reused
        gamma=gamma,                    # same discount factor as DQN
        clip_range=0.2,                 # clipping default
        ent_coef=0.01,                  # coefficient for max entropy regularisation
        verbose=0,
        policy_kwargs=policy_kwargs,
    )


    callback = EpisodeReturnLogger()
    t1=time.time()
    model_PPO.learn(total_timesteps=7000, callback=callback)
    t2=time.time()
    print("total time for ppo", t2-t1)
    ppo_returns = callback.episode_rewards
    #Finished with the PPO algorirthm# 

    #Now plotting the results:
    plot_returns([Q_learning_returns, deep_q_returns[0:num_episodes], ppo_returns[0:num_episodes]], ["Tabular Q", "Deep Q"], "PPO")

if __name__ == "__main__":
    main()

