import minihack_env as me
import matplotlib.pyplot as plt
import commons



### SIMPLE FILE SHOWCASING HOW TO HANDLE OBSERVATIONS FROM THE MINIHACK ENVS ###


# mpl.use('MacOSX')   #uncomment this in some MacOSX machines for matplotlib
# How to get a minihack environment from the minihack_env utility.

size = 5 

#Handling the chars representation:#
id = me.EMPTY_ROOM
env = me.get_minihack_envirnment(id, size=size)
state = env.reset()                                 #This state is state[0]=chars representation (2D array) inside dictionary, state[1]=info dictionary
next_state = env.step(1)                            #state[0]["char"] char representation, [1] reward int, [2] bool terminated, [3] bool truncated, [4] info dictionary
obs0 = state[0]["chars"]
obs1 = next_state[0]["chars"]
#Finished showcasing how to handle the chars representation#


#Showing how to handle the pixel representation and also how to get the central part!#
id = me.EMPTY_ROOM
env = me.get_minihack_envirnment(id, add_pixel=True)
state, info = env.reset()
print("Initial state", state)
plt.imshow(state["pixel"])
plt.show()
#Finished showing how to handle the pixel representation and the observation depiction#

#Crop representations to non-empty part#
#get_crop_chars_from_observation -> receives the observation, extracts the char part, finds which of the char part is not blank (e.g. !=32) and RETURNS THE NON BLACK PART
#get_crop_pixel_from_observation -> receives the observation, finds the non blank part BUT returns the image representation
id = me.EMPTY_ROOM
env = me.get_minihack_envirnment(id, add_pixel=True)
state, info = env.reset()
print("Initial state", commons.get_crop_chars_from_observation(state))
plt.imshow(commons.get_crop_pixel_from_observation(state))
plt.show()
#Finished showcasing how to work with the cropped versions#