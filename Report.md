# What I learned from Project 1: Navigation

## Overall Description of the Implementation

### Functional Approach

In search of a way to force myself to consider every portion of the algorithm involved in deep Q learning
despite having much of it provided by Udacity's thorough and well explained precursor assignments, I
decided to take the plunge and also attempt to write my version in as functional a style as I could in 
python without getting too bogged down in the transformation from imperative to functional or paddling 
upstream too much with regards to pythonic style. So first, I'm outlining some psuedocode for myself based
on the Deep Q Learning whitepaper, which I go over in the 'Learning Algorithm Used' section below

## Learning Algorithm Used

#### High Level Pseudocode from deep q learning paper

##### Initialization
Initialize the replay buffer
Initialize action-value function stand-in (torch nn) with random weights
Initialize a target action-value function stand in (torch nn) with weights set to the moving nn

##### Procedure
- for all of these steps, do them for every episode in a list of epsiodes M long
1. in the paper they get the first sequence and preprocessed sequence of 4 raster images, but for the version of this where you are getting raycast info just use intial state
- for all of these steps, do them for every time step t in an episode
2. select an action in an epsilon greedy fashion, so it is epsilon chance it is random explore, otherwise greedy
3. execute that action and get back the reward and the next state
4. in the deep q version with images, you'd have to get the image and use that image to preprocess and get the next 4 image state tensor, but in this case you already have the next state fed to you by the environment
5. add the state transition information in the memory buffer. You will uniformly sample from these instead of just learning from transitions as you are creating them. That helps dampen effects caused by the correlation you'd see between states that are only one or two time steps apart. The network is likely to pick up on the relationships between sequential steps and make updates that overemphasize the importance of these relationships and their role in maximizing discounted return. Also, sampling randomly from experience means an experience can be used multiple times to update the parameters and that means more data efficiency than using each experience only once. However, this is one area where optimization is possible because not all experiences have the same inherent value for learning to optimize
6. sample the minibatch transitions. In the deep Q udacity code they are careful to ensure that there are enough samples by this point that it makes sense to do a sample.
7. set the reward for this sample equal to the reward you get from taking the action if you are on the very last step, otherwise set it to the reward plus the discounted action value of taking a follow up action with the target network
8. do gradient descent with respect to the parameters using the loss function, in which y is represented by the target network's action value with more fixed parameters and the loss function you are finding is between the squared error between the y target and the action value function with the current network.  you are using the discounted reward calculated with the target network with fixed parameters for the y term, and this will help reduce the likelihood of oscillations you would get if you used the same network to determine y that you did to determine you action value. If you didn't do this, there'd be an effect where updating the action value (with adjusted weights) would likely also increase the reward, so the gradient descent is more likely to oscillate because it is chasing a moving target. separating updates more gives the optimization time to try an hit the established reward target before moving it, which makes for a more focused movement.
9. by the time you have reached C time steps where C is some hyperparameter used for how often you update the target network weights, do a soft update of the target network. 

## Plot Of Rewards


### Number Of Episodes to Solution


## Ideas For Future Work

