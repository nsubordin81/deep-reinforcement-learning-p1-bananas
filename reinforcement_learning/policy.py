from dataclasses import dataclass

""" Maps the environment information received into the next action to take"""


def policy_function(unity_params):
    # I know I'll need these in here, to decide how to act when on policy
    epsilon_gen = anneal_epsilon()
    epsilon = next(epsilon_gen)
    pass


""" Learning Method, Using Deep RL """


def learn():
    pass


def update_target_weights(learning_network, target_network):
    pass


""" Epsilon Greedy Exploration """


def epsilon_greedy(epsilon, action_values, action_size):
    pass


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
