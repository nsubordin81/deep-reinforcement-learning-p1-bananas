from reinforcement_learning.policy import EpsilonParams
from reinforcement_learning.policy import anneal_epsilon
import pytest


@pytest.mark.parametrize(
    "input,times,expected"[
        (EpsilonParams(10, 0, 5), 1, 5), (EpsilonParams(10, 0, 5), 2, 0)
    ]
)
def test_anneal_epsilon_decreases_normally(input, times, expected):
    new_e = None
    for t in range(times):
        new_e = anneal_epsilon()
    assert new_e == expected
