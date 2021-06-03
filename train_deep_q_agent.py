import random
from dataclasses import dataclass
from typing import Any, Deque, List

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment

from utils.scoring import ScoreTrackers, scorekeeper
from reinforcement_learning.policy import policy_function, learn, update_target_weights
from reinforcement_learning.environment import setup_unity_env, environment
from reinforcement_learning.experience_replay import ExperienceDataset

BUFFER_SIZE = int(1e5)  # how many experiences to hold in dataset at a time
BATCH_SIZE = 64  # how many examples per mini batch
ACTION_SIZE = 4
STATE_SIZE = 37
SEED = 0  # a way to seed the randomness for uniform selection so we can have repeatable results
UPDATE_EVERY = 4  # how often to update the weights of the target network to match the active network
GAMMA = 0.99  # discount factor
LEARNING_RATE = 5e-4  # learning rate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

""" Instantiating a dataset for experience replay with the constants from above.
    Not really happy to be putting this in the __main__ namespece, considering memoizing it
    and doing this within the function, but for now this is fine """
experience_dataset = ExperienceDataset(
    ACTION_SIZE, BUFFER_SIZE, BATCH_SIZE, random.seed(SEED)
)

""" having these global to the module and passing them around because I don't have implicits. better
way to do this in python without classes? Not sure so doing this for now """
learning_network = QNN(state_size, action_size, seed).to(device)
target_network = QNN(state_size, action_size, seed).to(device)

optimizer = optim.Adam(learning_network.parameters(), lr=LEARNING_RATE)


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
returns: the accumulated score for all timesteps in the episode
"""


@scorekeeper
def train_agent(unity_params, episode_index, max_timesteps, score_trackers):
    score = 0
    for i in play_episode_and_train(max_timesteps, unity_params):
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
#    - higher level abstraction, need a way to learn dependent on the replay buffer
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
        """ all the code in here could use some refactoring, would like to 
        find ways to make it less imperative, it is a collection of functions called in order
        rather than a series of functional compositions """
        action = policy_function(
            unity_params, learning_network, state, ACTION_SIZE
        )  # getting a
        train_env = environment(n, unity_params, action)
        next_state = train_env.vector_observations[0]  # getting s'
        reward = train_env.rewards[0]  # getting r
        # experience dataset is global, passing it down through would feel better but the context would grow
        # internal deque is mutable so we just side effect here
        experience_dataset.save(
            state, action, reward, next_state, train_env.local_done[0]
        )
        if len(experience_dataset) >= BATCH_SIZE:
            learn(
                learning_network,
                target_network,
                optimizer,
                experience_dataset.sample(),
                GAMMA,
            )

        # updating target network
        if n % UPDATE_EVERY == 0:
            update_target_weights(learning_network, target_network)

        state = next_state  # getting s for next iter
        yield reward  # return the reward for this episode
        if train_env.local_done[0]:
            return 0
        n += 1


if __name__ == "__main__":
    main()
