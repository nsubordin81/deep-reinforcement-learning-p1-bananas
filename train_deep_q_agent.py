import random
from toolz import curry

from unityagents import UnityEnvironment
import torch.optim as optim

from utils.scoring import ScoreTrackers, scorekeeper
from reinforcement_learning.policy import policy_function, learn, update_target_weights
from reinforcement_learning.environment import setup_unity_env, environment
from reinforcement_learning.experience_replay import ExperienceDataset
from deep_learning.deep_q_network import BananaQNN
from utils.shared import device

BUFFER_SIZE = int(1e5)  # how many experiences to hold in dataset at a time
BATCH_SIZE = 64  # how many examples per mini batch
ACTION_SIZE = 4
STATE_SIZE = 37
SEED = 0.0  # a way to seed the randomness for uniform selection so we can have repeatable results
UPDATE_EVERY = 4  # how often to update the weights of the target network to match the active network
GAMMA = 0.99  # discount factor
LEARNING_RATE = 5e-4  # learning rate

NUM_EPISODES = 500
MAX_TIMESTEPS = 1000

""" Instantiating a dataset for experience replay with the constants from above.
    Not really happy to be putting this in the __main__ namespece, considering memoizing it
    and doing this within the function, but for now this is fine """
experience_dataset = ExperienceDataset(
    ACTION_SIZE, BUFFER_SIZE, BATCH_SIZE, random.seed(SEED)
)

""" having these global to the module and passing them around because I don't have implicits. better
way to do this in python without classes? Not sure so doing this for now """
learning_network = BananaQNN(STATE_SIZE, ACTION_SIZE, SEED).to(device)
target_network = BananaQNN(STATE_SIZE, ACTION_SIZE, SEED).to(device)
banana_optimizer = optim.Adam(learning_network.parameters(), lr=LEARNING_RATE)


# entry point for training
def main():
    # setting some initial values:
    # need to set up environment first
    score_trackers = ScoreTrackers()
    unity_params = setup_unity_env(file_name="./Banana_Windows_x86_64/Banana.exe")
    # currying all the global scope, so later I just change vals here
    train_banana_agent = train_agent(
        unity_params=unity_params,
        score_trackers=score_trackers,
        max_timesteps=MAX_TIMESTEPS,
    )

    # side effecting for now, maybe there is a return value I'm not thinking of yet.
    [train_banana_agent(episode_index=i) for i in range(NUM_EPISODES + 1)]


""" train_agent
this function runs once per episode
pre: receives all infor necessary to conduct training and what is neede by the scorekeeper decorator
action: iterates over the play_episode_and_train generator, accumulating scores yielded by it.

Parameters:
unity_params -- a dataclass object with the unity mlagents objects necessary for training
max_timesteps -- passing the total number of timesteps to keep it in context, TODO curry these first two?
expisode_index -- 
returns: the accumulated score for all timesteps in the episode
"""


@curry
@scorekeeper  # episode_index, score_trackers used by scorekeeper
def train_agent(unity_params, max_timesteps, episode_index, score_trackers):
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


def play_episode_and_train(
    max_timesteps, unity_params,
):
    n = 0
    banana_environment = environment(unity_params.brain_name)
    state = banana_environment(n, unity_params.env.reset).vector_observations[
        0
    ]  # getting s
    while n < MAX_TIMESTEPS:
        """ all the code in here could use some refactoring, would like to 
        find ways to make it less imperative, it is a collection of functions called in order
        rather than a series of functional compositions """
        action = policy_function(
            unity_params, learning_network, state, ACTION_SIZE
        )  # getting a

        train_env = banana_environment(n, unity_params.env.step, action)  # step
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
                banana_optimizer,
                experience_dataset.uniform_random_sample(),
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
