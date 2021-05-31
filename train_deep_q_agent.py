import random
from dataclasses import dataclass
from typing import Any, Deque, List

import torch
import numpy as np
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment

from scoring import ScoreTrackers, scorekeeper
from agent import act, epsilon_greedy, anneal_epsilon
from environment import setup_unity_env, environment


@dataclass
class UnityParams:
    """ class for holding the mlagents specific unity environment parameters """

    env: UnityEnvironment
    brain_name: str
    brain: Any


# entry point for training
def main():
    # setting some initial values:
    # need to set up environment first
    num_episodes = 20
    max_timesteps = 1000
    score_trackers = ScoreTrackers()
    unity_params = setup_unity_env(file_name="./Banana_Windows_x86_64/Banana.exe")

    # side effecting for now, maybe there is a return value I'm not thinking of yet.
    [
        train_agent(unity_params, i, max_timesteps, score_trackers)
        for i in range(num_episodes + 1)
    ]


""" train_agent
this function runs once per episode
pre: receives all infor necessary to conduct training and what is neede by the scorekeeper decorator
action: iterates over the play_episode_and_train generator, accumulating scores yielded by it.
returns: the accumulated score for all timestepcs in the episode
"""


@scorekeeper
def train_agent(unity_params, episode_index, max_timesteps, score_trackers):
    score = 0
    epsilon_gen = anneal_epsilon()
    epsilon = next(epsilon_gen)
    for i in play_episode_and_train(epsilon, max_timesteps, unity_params):
        score += i
    return score


""" play_episode_and_train
pre:
action: this generator funciton is responsible for the control flow of the looping through one time step in 
training an agent. future enhancement, if this is to be a framework for mor than deep q, make it
higher order and turn the behaviors of taking actions into pluggable functions.
returns: will yield the score for the current episode until the episode is done, at which point it
will break and return 0

"""
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


def play_episode_and_train(
    epsilon, max_timesteps, unity_params,
):
    n = 0
    state = environment(n, unity_params).vector_observations[0]  # getting s
    while n < max_timesteps:
        action = act(unity_params)  # getting a
        train_env = environment(n, unity_params, action)
        next_state = train_env.vector_observations[0]  # getting s'
        reward = train_env.rewards[0]  # getting r

        # saving sars'
        # learning from experience samples
        # updating target network

        state = next_state  # getting s for next iter
        yield reward
        if train_env.local_done[0]:
            return 0
        n += 1


if __name__ == "__main__":
    main()
