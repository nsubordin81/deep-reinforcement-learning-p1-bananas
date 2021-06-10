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

I used an implementation of the original DeepQLearning algorithm. One modification I had was I opted for a soft update as well eventually
as a way of tuning the correlationa between the target network and the actively learning network, because I wasn't seeing good performance with hard
updates at the time. 

### High Level Pseudocode from deep q learning paper

#### Initialization
Initialize the replay buffer
Initialize action-value function stand-in (torch nn) with random weights
Initialize a target action-value function stand in (torch nn) with weights set to the moving nn

#### Procedure
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

### Notes about what I learned during implementation
1. I also used this project as an opportunity to attempt a more functional approach. I like to apply a more functional paradigm informed approach to programming when I can because I like the simplicity and design flexibility afforded by referential transparency and function composition over class based object  oriented designs. 

However, What I learned from attempting this approach with this project was that pytorch takes a more classical object/oriented approach to setting up its network, so I kind of had to treat that as more or less its own component that I interacted with as an interface but pretty much followed their conventions for setting up the neural network class and instantiating my networks. My second finding in this space is that python's support for functional constructs leaves a little to be desired as compared to Scala, Haskell, etc. That wasn't a big surprise, but the place where I noticed it most was dealing with calling context i.e. how many parameters I needed to pass around between functions. setting up an MDP based training scenario without stateful objects means a lot of global state to hold and manage all of the object that were needed, which means there are lots of parameters to the top level functions for hyperparameters and the like, and they filter down from top to bottom. I was able to take advantage of generators in a few places to achieve stateful control flow and also higher order functions and currying to embed some of the parameters into partially applied functions, but the outermost functions still had quite a few parameters in them just for the sake of inversion of control so I could configure them at the entrypoint to the training script. My approach affords a lot of flexibility while keeping functions more granular and ultimately independent of an object driven structure, but it also means there are parameters that are passed down only to be used in a function more than one layer deep. I'm hoping to build a library of functions that I can use to set up reinforcement learning solutions faster in the future, as a promise of doing things in this functional manner would allow, but I have yet to see if this approach would be more extensible and flexible than doing the same with the more familiar and conventional OO approach in python. I might be fighting a losing battle, also because referential transparency is difficult to achieve when so many of the conventional ways to do this rely on functions that don't return a value and datastructures that are updated in place.

2. Then there is what I learned specific to reinforcement learning and deep learning in the context of DQN. Once I got to the point where the agent was taking actions in the environment and the BananaQNN was being leveraged to hopefully learn on a set interval, I noticed that not only was there not monotonic improvement across my 500 episode training period, there was instead oscillation and barely any change in the scores for each 10 episode increment. The 10 episode experience logs were showing that the agent would be averaging in the -0.01 to 0 range for some tens of episodes and then it would appear to improve and instead stay in the range 0.01 to 0.1. The problem was it would go back to the other range only 100 episodes more after that. I made several changes, some of which were issues with my algorithm, some of which were more like tuning: 
    - I noticed that my epsilon generator function was being redefined on each timestep instead of annealing at the rate of 0.005 like I wanted it was staying at a constant 1, so it was never using the greedy action and always doing uniform random selection. I moved definition of the generator out to fix that and with some debug statements was able to verify that epsilon was annealing properly. 
    - Even though the agent was now acting greedily more often, the oscillating scores were still present, but it looked like there was some attempt to minimize loss going on. But something was preventing the model from converging on a policy that would find the yellow bananas. One thing I had remembered doing intentially was adopt a hard swap of weights between the model that was actively learning and the model providing the target. So I tried extending this period out and then ultimately moved to a soft update and also taking actions for several steps and then learning and updating weights together on the Cth step, like the one used in the DQN exercise on udacity. This did not appreciably increase the scores or make them more linearly increasing as I hoped it would. 
    - I continued to tune hyperparameters like batch size, learning rate, how often I learned and swapped weights, whether I did these two things at the same time or not, and how many entries were allowed in the experience replay deque, but nothing was creating significant improvement. Then I noticed a small but significant problem with my learning function. 
    
    I had correctly calculated the target and assigned it to a variable, but instead of using that in the loss function I had just passed in the target network s` action values directly as the target by mistake. It follows that this would make the network poor at optimizing because the action value of the target network at the s' state has no guarantee of being larger than the action value for the learning network at state s. So not only was this of course going to be a poor approximator of the true target compared with a term tha thad the reward and discount factored in, it also might not even be higher than the current value, so imagine trying to minimize a loss like that! 
    
    Fixing this improved the performance markedly from the -.3-.3 range to at the best times almost averaging 1 banana an episode. But I still needed to go back and try out some of the hyperparameter tuning again now that the network was better able to learn.

## Plot Of Rewards


### Number Of Episodes to Solution


## Ideas For Future Work

