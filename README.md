# Super Hexagon AI

 - Based on Reinforcement Learning (Q-Learning)
 - Beats all six levels easily end to end (i. e. from pixel input)
 - Fast C++ implementation that hooks into the game and hands the screen over to the python process
 

# Approach
This AI is based on Reinforcement Learning more specifically Deep Q-Learning [1]. 
I. e. the Q-Function is learned which represents the discounted expected future reward if the agent acts according to the Q-Function.
The agent receives a reward of -1 if he dies otherwise a reward of 0.  

# References
[1] Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." nature 518.7540 (2015): 529-533. 
 