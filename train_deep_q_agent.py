import random
from collections import deque

import torch
import numpy as np
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment

# entry point for training
def main():
    epsilon_start = 0.99
    # need to set up environment first
    file_name = "./Banana_Windows_x86_64/Banana.exe"
    env, brain_name, brain = setup_unity_env(file_name)
    action_size = brain.vector_action_space_size
    num_episodes = 20
    epsilon = epsilon_start
    for i in range(1, num_episodes + 1):
        scores = []
        scores_window = deque(maxlen=100)
        # return the BrainInfo object from the environment for this episode
        brain_env_info = env.reset(train_mode=True)[brain_name]
        # getting initial state
        state = brain_env_info.vector_observations[0]
        # reset score to zero, new episode
        # loop through timesteps in current episode
        score = 0
        while True:
            # pick next action from action space. in this first iteration
            # we still don't have a neural net telling what the best action is, so
            # just pick a random one
            action = np.random.randint(
                action_size
            )  # we will get this from the learning network
            # get the next state for that action
            brain_env_info = env.step(action)[brain_name]  # this is the actual step
            next_state = brain_env_info.vector_observations[0]  # getting s'
            reward = brain_env_info.rewards[0]  # getting r
            # at this point you have the SARS' tuple, so you can do similar to
            # the deep q learning approach. You need to create two networks
            # and you need one to have the fixed target params and the other
            # to have the moving weights that get updated.
            # TODO  full experience tuple needs to be saved in memory
            # TODO need a way to sample from experience tuples
            # TODO logic for knowing when to learn from experience

            # TODO need epsilon-greedy offline policy action selection, get from net and then be epsilon greedy so we explore enough
            # TODO need to interpolate between the target network weights and the ones for the network that is learning
            # TODO create the loss function
            done = brain_env_info.local_done[0]
            score += reward
            state = next_state
            if done:
                break
        scores_window.append(score)  # saving scores to 100 len window
        scores.append(score)  # and overall tracker
        if i % 10 == 0:
            print(f"\rEpisode: {i}\tAverage Score: {np.mean(scores_window)}")


# TODO: this should eventually support an arbitrary number of brains
def setup_unity_env(file_name):
    env = UnityEnvironment(file_name)
    brain_name = env.brain_names[0]
    # this will hold the actions
    brain = env.brains[brain_name]
    return env, brain_name, brain


if __name__ == "__main__":
    main()
