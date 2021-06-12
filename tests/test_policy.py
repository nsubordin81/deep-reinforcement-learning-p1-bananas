from reinforcement_learning.policy import EpsilonParams
from reinforcement_learning.policy import anneal_epsilon
import pytest


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
