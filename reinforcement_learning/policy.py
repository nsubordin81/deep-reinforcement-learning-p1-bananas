import random
from functools import reduce
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import numpy as np

from utils.shared import device

""" Maps the environment information received into the next action to take"""


def policy_function(unity_params, q_network, state, action_size):
    # get the action from the
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    """ not using dropout now but we might in the future, 
    so might as well turn regularization off for inference """
    q_network.eval()
    with torch.no_grad():
        action_values = q_network(state)
    q_network.train()

    epsilon_gen = anneal_epsilon()
    epsilon = next(epsilon_gen)

    greater_than_epsilon = lambda epsilon: random.random() > epsilon
    max_action = lambda *args: np.argmax(action_values.cpu().data.numpy())
    random_action = lambda *args: random.choice(np.arange(action_size))

    return (greater_than_epsilon(epsilon) and max_action()) or random_action()


""" Learning Method, Using Deep RL """


def learn(learning_network, target_network, optimizer, experience_batch, gamma):
    states, actions, rewards, next_states, dones = experience_batch

    #     import pdb
    #
    #     pdb.set_trace()

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


def update_target_weights(learning_network, target_network):
    for target_param, local_param in zip(
        target_network.parameters(), learning_network.parameters()
    ):
        target_param.data.copy_(local_param.data)


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
