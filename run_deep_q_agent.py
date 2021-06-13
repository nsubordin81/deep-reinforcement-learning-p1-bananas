from train_deep_q_agent import MAX_TIMESTEPS
import torch

from reinforcement_learning.policy import apply_policy, epsilon_greedy
from reinforcement_learning.environment import environment, setup_unity_env
from deep_learning.deep_q_network import BananaQNN
from utils.shared import device

""" Run Deep Q Agent
This script will load the learnable parameters from the file ./model.pth
and demonstrate an agent taking actions using a network with those weights 
and biases loaded on to it to navigate the unity environment
"""


def main():
    ACTION_SIZE = 4
    STATE_SIZE = 37
    SEED = 0.0  # a way to seed the randomness for uniform selection so we can have repeatable results
    MAX_TIMESTEPS = 1000
    unity_params = setup_unity_env(file_name="./Banana_Windows_x86_64/Banana.exe")

    trained_network = BananaQNN(STATE_SIZE, ACTION_SIZE, SEED).to(device)
    trained_network.load_state_dict(torch.load("model.pth"))
    banana_counter = ([], [])
    tabulate_bananas = (
        lambda x: banana_counter[0].append(x) if x < 0 else banana_counter[1].append(x)
    )
    score = 0
    for i in loop_env(unity_params, trained_network):
        tabulate_bananas(i)
    print(
        f"Score For Epsiode Was {sum(banana_counter[0]) + sum(banana_counter[1])}. Collected {abs(sum(banana_counter[0]))} Blue Bananas and {sum(banana_counter[1])} Yellow Bananas"
    )


# TODO The control flow and much of the environment interaction is the same for this and training. Should just abstract that into
# a function, might as well take advantage of the functional approach. It could live in the environment module.
def loop_env(unity_params, trained_network):
    n = 0
    b_environment = environment(unity_params.brain_name, training=False)
    state = b_environment(n, unity_params.env.reset).vector_observations[0]
    while n < MAX_TIMESTEPS:
        trained_network.eval()
        # epsilon is 0.0 so we are always greedy, TODO, maybe with currying can make this more explicitly online greedy
        # also, this will keep putting the model in train mode, should do something about that.
        action = apply_policy(0.0, trained_network, state, epsilon_greedy,)  # getting a

        train_env = b_environment(n, unity_params.env.step, action)  # step
        next_state = train_env.vector_observations[0]  # getting s'
        reward = train_env.rewards[0]  # getting r

        state = next_state  # getting s for next iter
        yield reward  # return the reward for this episode
        if train_env.local_done[0]:
            return 0
        n += 1


if __name__ == "__main__":
    main()
