import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import numpy as np

from utils.shared import device

""" Maps the environment information received into the next action to take"""


def policy_function(epsilon, q_network, state, action_size):
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    """ not using dropout now but we might in the future, 
    so might as well turn regularization off for inference """
    q_network.eval()
    with torch.no_grad():
        action_values = q_network(state)
    q_network.train()

    greater_than_epsilon = lambda epsilon: random.random() > epsilon
    # have to cast the argmax values as numpy int32, otherwise unity mlagents code breaks environment.py line 322 looking for 'keys'

    max_action = lambda *args: np.argmax(action_values.cpu().data.numpy()).astype(
        np.int32
    )

    random_action = lambda *args: random.choice(np.arange(action_size)).astype(np.int32)

    return (greater_than_epsilon(epsilon) and max_action()) or random_action()


""" Learning Method, Using Deep RL """


def learn(learning_network, target_network, optimizer, experience_batch, gamma):
    states, actions, rewards, next_states, dones = experience_batch

    # for the whole batch, get a' = Q(s', a, r, w)
    # being laborious for debugging internal typing but also to teach myself pytorch api
    action_values = target_network(next_states)
    removed_values = action_values.detach()
    max_action_values = removed_values.max(1)[0]
    reshaped_target_action_value_max = max_action_values.unsqueeze(1)

    # prepare the y approximation, current reward plus the discounted target action values
    # this vector operation will zero out the action value for next state when we got the local_done signal,
    # as per the suggestion in the whitepaper pseudocode and udacity example
    y = rewards + (gamma * reshaped_target_action_value_max * (1 - dones))

    # getting the action values from the model that is learning
    learning_action_value_estimates = learning_network(states).gather(1, actions.long())

    # loss function, mean squared error
    loss = F.mse_loss(learning_action_value_estimates, reshaped_target_action_value_max)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# started with this hard update. after seeing oscillating scores over several episodes,
# came to the conclusion that there might still be a correlation caused by
# the similarity of the learning state of the two networks, so moving to soft update
def update_target_weights(learning_network, target_network):
    for target_param, local_param in zip(
        target_network.parameters(), learning_network.parameters()
    ):
        target_param.data.copy_(local_param.data)


def soft_update_target_weights(learning_network, target_network, tau=None):
    """ interpolate the weights of the target network to stay tau percent near
    their current value and move 1-tau percent towards the newer, learned weights """
    for target_weight, learning_weight in zip(
        target_network.parameters(), learning_network.parameters()
    ):
        target_weight.data.copy_(
            tau * learning_weight.data + (1.0 - tau) * target_weight.data
        )


""" anneal_epsilon
This is a generator that will yield progressively more annealed epsilons 
until it reaches the minimum value and then generates infinite minimums
"""


@dataclass
class EpsilonParams:
    epsilon_initial: float = 1.00
    epsilon_final: float = 0.01
    epsilon_decay_rate: float = 0.995


def anneal_epsilon():
    e_p = EpsilonParams()
    epsilon = e_p.epsilon_initial
    yield epsilon
    while True:
        if epsilon > e_p.epsilon_final:
            epsilon = epsilon * e_p.epsilon_decay_rate
            yield epsilon
        else:
            yield e_p.epsilon_final
