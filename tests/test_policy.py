from unittest.mock import patch

import pytest
import torch

from reinforcement_learning.policy import EpsilonParams, anneal_epsilon, epsilon_greedy


@pytest.mark.parametrize(
    "input,times,expected",
    [
        (EpsilonParams(10, 0, 0.5), 1, 10),
        (EpsilonParams(10, 1, 0.5), 5, 1),
        (EpsilonParams(10, 1, 0.5), 10, 1),
        (EpsilonParams(10, 0, 0.5), 2, 5),
    ],
)
def test_anneal_epsilon_decreases_normally(input, times, expected):
    epsilon = anneal_epsilon(input)
    result = None
    for t in range(times):
        result = next(epsilon)
    assert result == expected


@pytest.mark.parametrize(
    "epsilon, random, choice, actions, expected_index",
    [
        (0.5, 0.99, 4, torch.tensor([8, 2, 3, 4]), 0),
        (0.99, 0.5, 2, torch.tensor([8, 2, 3, 4]), 2),
        (0.5, 0.99, 0, torch.tensor([4, 2, 3, 8]), 3),
    ],
)
@patch("reinforcement_learning.policy.random.choice")
@patch("reinforcement_learning.policy.random.random")
def test_epsilon_greedy_picks_max_and_random_according_to_strategy(
    test_random, test_choice, epsilon, random, choice, actions, expected_index
):
    test_random.return_value = random
    test_choice.return_value = choice
    result = epsilon_greedy(epsilon, actions)
    assert result == expected_index

