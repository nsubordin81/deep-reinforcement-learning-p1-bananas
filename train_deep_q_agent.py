import random
from toolz import curry

from unityagents import UnityEnvironment
import torch.optim as optim

from utils.scoring import ScoreTrackers, scorekeeper
from reinforcement_learning.policy import (
    policy_function,
    learn,
    soft_update_target_weights,
    anneal_epsilon,
)
from reinforcement_learning.environment import setup_unity_env, environment
from reinforcement_learning.experience_replay import ExperienceDataset
from deep_learning.deep_q_network import BananaQNN
from utils.shared import device

BUFFER_SIZE = int(1e6)  # how many experiences to hold in dataset at a time
BATCH_SIZE = 128  # how many examples per mini batch
ACTION_SIZE = 4
STATE_SIZE = 37
SEED = 0.0  # a way to seed the randomness for uniform selection so we can have repeatable results
UPDATE_EVERY = 4  # how often to update the weights of the target network to match the active network but also how often you learn
GAMMA = 0.99  # discount factor
TAU = 1e-4  # starting with a very small tau, so the proportion of learning network weight will be small
LEARNING_RATE = 5e-4  # learning rate

NUM_EPISODES = 5000
MAX_TIMESTEPS = 1000

""" Instantiating a dataset for experience replay with the constants from above.
    Not really happy to be putting this in the __main__ namespece, considering memoizing it
    and doing this within the function, but for now this is fine """
experience_dataset = ExperienceDataset(
    ACTION_SIZE, BUFFER_SIZE, BATCH_SIZE, random.seed(SEED)
)

""" Script Entrypoint
sets up all of the global context used in the agent's training, hopefully a one stop shop for
most of the things one would want to tweak when tuning the algorithm """


def main():
    # setting some initial values:
    # need to set up environment first
    score_trackers = ScoreTrackers()
    unity_params = setup_unity_env(file_name="./Banana_Windows_x86_64/Banana.exe")
    # state and funcs to be used by policy
    epsilon_generator = anneal_epsilon()
    learning_network = BananaQNN(STATE_SIZE, ACTION_SIZE, SEED).to(device)
    target_network = BananaQNN(STATE_SIZE, ACTION_SIZE, SEED).to(device)
    banana_optimizer = optim.Adam(learning_network.parameters(), lr=LEARNING_RATE)
    training_counter = 0
    # currying all the global scope, so later I just change vals here
    train_banana_agent = train_agent(
        unity_params=unity_params,
        score_trackers=score_trackers,
        max_timesteps=MAX_TIMESTEPS,
        epsilon_generator=epsilon_generator,
        learning_network=learning_network,
        target_network=target_network,
        optimizer=banana_optimizer,
        training_counter=training_counter,
    )

    # highest level iteration, the episode loop
    [train_banana_agent(episode_index=i) for i in range(1, NUM_EPISODES + 1)]


""" train_agent
this function runs once per episode
pre: receives all infor necessary to conduct training and what is neede by the scorekeeper decorator
action: iterates over the play_episode_and_train generator, accumulating scores yielded by it.

Parameters:
unity_params -- a dataclass object with the unity mlagents objects necessary for training
max_timesteps -- passing the total number of timesteps to keep it in context, TODO curry these first two?
expisode_index --  the index of the episode we are on now, added for scorekeeper
score_trackers -- the deque and array being used to keep the scores, added for scorekeeper
returns: the accumulated score for all timesteps in the episode
"""


@curry
@scorekeeper  # episode_index, score_trackers used by scorekeeper
def train_agent(
    unity_params,
    max_timesteps,
    epsilon_generator,
    episode_index,
    score_trackers,
    learning_network,
    target_network,
    optimizer,
    training_counter,
):
    score = 0
    epsilon = next(epsilon_generator)
    # print(f"starting a new episode, epsilon: {epsilon}")
    # iterate over the play_episode_and_train generator
    for i in play_episode_and_train(
        epsilon,
        max_timesteps,
        unity_params,
        learning_network,
        target_network,
        optimizer,
        training_counter,
    ):
        # if i != 0.0:
        #     print(f"reward for timestep is {i}")
        score += i
    # print(f"overall reward is: {score}")
    return score


""" play_episode_and_train
pre:
action: this generator funciton is responsible for the control flow of the looping through one time step in 
training an agent. future enhancement, if this is to be a framework for mor than deep q, make it
higher order and turn the behaviors of taking actions into pluggable functions.
returns: will yield the score for the current episode until the episode is done, at which point it
will break and return 0

"""


def play_episode_and_train(
    epsilon,
    max_timesteps,
    unity_params,
    learning_network,
    target_network,
    optimizer,
    training_counter,
):
    # Everything before the while loop starts is only executed once per episode, as this is a
    # generator function
    n = 0
    banana_environment = environment(unity_params.brain_name)
    state = banana_environment(n, unity_params.env.reset).vector_observations[
        0
    ]  # getting initial s
    while n < max_timesteps:
        """ all the code in here could use some refactoring, would like to 
        find ways to make it less imperative, it is a collection of functions called in order
        rather than a series of functional compositions """
        action = policy_function(
            epsilon, learning_network, state, ACTION_SIZE,
        )  # getting a

        train_env = banana_environment(n, unity_params.env.step, action)  # step
        next_state = train_env.vector_observations[0]  # getting s'
        reward = train_env.rewards[0]  # getting r
        # global so I don't have to pass it, seeing the advantage of objects with all these function params
        experience_dataset.save(
            state, action, reward, next_state, train_env.local_done[0]
        )

        # updating target network
        training_counter = (training_counter + 1) % UPDATE_EVERY
        if training_counter == 0:
            # had this outside so we could learn every time, but experience pool must need to
            # gather more items, and I guess this way we are always changing the target when we learn
            # might need soft update here also
            if len(experience_dataset) > BATCH_SIZE:
                learn(
                    learning_network,
                    target_network,
                    optimizer,
                    experience_dataset.uniform_random_sample(),
                    GAMMA,
                )
                soft_update_target_weights(
                    learning_network=learning_network,
                    target_network=target_network,
                    tau=TAU,
                )

        state = next_state  # getting s for next iter
        yield reward  # return the reward for this episode
        if train_env.local_done[0]:
            return 0
        n += 1


if __name__ == "__main__":
    main()
