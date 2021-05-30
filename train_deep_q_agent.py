import random
import functools
from collections import deque
from dataclasses import dataclass
from typing import Any

import torch
import numpy as np
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment

from agent import determine_next_action


@dataclass
class UnityParams:
    """ class for holding the mlagents specific unity environment parameters """

    env: UnityEnvironment
    brain_name: str
    brain: Any


@dataclass
class TrainingParams:
    """ class for holding hyperparameters for the deep Q algorithm """

    epsilon_initial: float
    num_episodes: int
    max_timesteps: int


# entry point for training
def main():
    # setting some initial values:
    # need to set up environment first
    scores = []
    scores_window = deque(maxlen=100)
    training_params = TrainingParams(0.99, 20, 1000)
    unity_params = setup_unity_env(file_name="./Banana_Windows_x86_64/Banana.exe")

    # side effecting for now, maybe there is a return value I'm not thinking of yet.
    [
        train_agent(training_params, unity_params, i, scores, scores_window)
        for i in range(training_params.num_episodes + 1)
    ]

    # all this code can go away once I'm done breaking it up into functions
    # keeping it around now for reference

    for i in range(1, num_episodes + 1):
        # return the BrainInfo object from the environment for this episode
        brain_env_info = env.reset(train_mode=True)[brain_name]
        # getting initial state
        state = brain_env_info.vector_observations[0]
        # reset score to zero, new episode
        # loop through timesteps in current episode
        experience_counter = 0
        while True:
            # pick next action from action space. this will be epsilon-greedy
            action = determine_next_action(
                brain, selection_method=epsilon_greedy
            )  # make a selection with local nn

            # get the next state for that action
            brain_env_info = env.step(action)[brain_name]  # this is the actual step
            next_state = brain_env_info.vector_observations[0]  # getting s'
            reward = brain_env_info.rewards[0]  # getting r
            # at this point you have the SARS' tuple, so you can do similar to
            # the deep q learning approach. You need to create two networks
            # and you need one to have the fixed target params and the other
            # to have the moving weights that get updated.
            # TODO  full experience tuple needs to be saved in memory
            #    - need an immutable data structure replay buffer
            #    - need something that adds to the replay buffer
            #    - need a way to check what is in the replay buffer to see if we can sample
            #    - higher level abstraction, need a way to learn dependent on the replay buffer
            #       TODO need a way to sample from experience tuples
            #       TODO logic for knowing when to learn from experience

            # TODO need epsilon-greedy offline policy action selection, get from net and then be epsilon greedy so we explore enough
            # TODO need to interpolate between the target network weights and the ones for the network that is learning
            # TODO create the loss function
            done = brain_env_info.local_done[0]
            state = next_state
            if done:
                break


""" setup_unity_env
just abstracting away these steps that seem standard for  getting the important 
bits out of the mlagents api. Probably not generic enough to be its own function right 
now but it cleaned up the readability of my main function
"""

# TODO: beyond the project requirements, but environments can have more than one brain
def setup_unity_env(file_name):
    env = UnityEnvironment(file_name)
    brain_name = env.brain_names[0]
    # Interesting thing I learned, because there can be multiple agents,
    brain = env.brains[brain_name]
    return UnityParams(env, brain_name, brain)


""" play_episode_and_train
this funciton runs one complete time step's actions
if you consider all actions of the functions composing it, will include the agent taking 
an action as a forward pass through the nn or at random, 
populating the replay buffer, and performing minibatch gradient descent updates of the 
network's parameters against the reward accumulated by the target network. It
will return the total reward accumulated for this time step for use by the scorekeeper
"""


def play_episode_and_train(
    training_params, unity_params,
):
    n = start
    while n < training_params["max_episodes"]:
        if n / 3 == 1:
            return
        else:
            yield n
        n += 1


""" train_agent
this function runs once per epsiode. 
It holds the inner loop control flow for the deep q algorithm, 
the one for each time step, and controls what state is passed 
to each occurrence of play_episode and train, which it calls once per time step 
"""


@scorekeeper
def train_agent(training_params, unity_params, episode_index, scores, scores_window):
    score = 0
    for i in play_episode_and_train(training_params, unity_params):
        score += i
    return score


def deterimine_state(timestep, unity_params, action):

    if timestep == 0:
        return unity_params["env"].reset(train_mode=True)[unity_params["brain_name"]]
    else:
        return unity_params["env"].step(action)[unity_params["brain_name"]]


""" scorekeeper
this function expects the function it wraps to return the aggregate score from
from its internal iterations. The function will then append that score to a running total
being maintained in two score attributes in the training_params dict that was passed into it
from a higher lexical scope
"""


def scorekeeper(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        score = func(*args, **kwargs)
        # saving scores to 100 len window and overall tracker
        kwargs["scores_window"].append(score)
        kwargs["scores"].append(score)
        report_score(episode_index, scores_window)

    return wrapper()


def report_score(episode_index, scores_window):
    if episode_index % 10 == 0:
        print(f"\rEpisode: {i}\tAverage Score: {np.mean(scores_window)}")


if __name__ == "__main__":
    main()
