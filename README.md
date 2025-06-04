# Reinforcement_Learning_MiniHack
In this project I use MiniHack to test RL algorithms. I implement On-Policy MC, SARSA, Tabular Q-Learning and Dyna-Q from scratch, following the algorithm descriptions of Sutton and Barto. I also use stable_baselines3 to implement a DQN and a PPO agent.

Commons.py      -> File contraining helper functions and abstract classes.  
minihack_env.py -> File containing descriptions and initialisation functions for the MiniHack environments I am going to be using.  
quickstart.py   -> File showcasing the use of functions to initialise the environments, process the observations and render.  

gym_exp.py      -> Simple file for familiarity with gymnasium. I implement a simple grid environment with one goal and a random agent. I run the agent on the environment.  
minihack_exp.py -> File in which I actually initialise the MiniHack environments I am going to be using, and run a FixedAgent on them.

classical_RL.py -> Here I implement the 4 classical ML algorithms I mention above, run experiments and do comparisons.
deep_RL.py      -> I use stable_baselines3 to run DQN and PPO agents. Again run experiments and comparisons.

To run code, just run the files with main functions.  
For detailed information about the implementations, the environments as well as the experiments and the corresponding conclusions please check the report.pdf!
