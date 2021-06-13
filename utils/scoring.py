import functools
from collections import deque
from dataclasses import dataclass, field

import torch
import numpy as np

# this is the goal according to the assignment
GOAL_SCORE = 13.5

""" scorekeeper
pre: expects the wrapped function to have scores_window deque, 
scores list and an episode_index defined.
expects: wrapped function is iterating over an episode and accumulating score totals, and its
return value is the accumulated total over the episode.
post: will add the score totals returned from the wrapped func to an array and a windowed deque, 
and it will report to std out at set intervals (currently not configurable) what the average score was
returns: None, since the calling function of func doesn't expect a return, this is side-effecting
"""


def scorekeeper(func):
    """ save the scores into provided scores and scores_window data structures, print at intervals """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        score = func(*args, **kwargs)
        # saving scores, requires that the wrapped func has a ScoreTracker class in its args
        score_trackers = kwargs["score_trackers"]
        score_trackers.scores_window.append(score)
        score_trackers.scores.append(score)
        # reporting scores, requires taht the wrapped func has a parameter called episode_index
        mean_score = np.mean(score_trackers.scores_window)
        episode_number = kwargs["episode_index"]
        _report_score(episode_number, mean_score)
        learning_network = kwargs["learning_network"]
        if mean_score > GOAL_SCORE:
            print(
                f"\nEnvironment Solved!! it took {episode_number-100} episodes and average score per episode was {mean_score} "
            )
            torch.save(learning_network.state_dict(), "model.pth")
        return score

    return wrapper


def _report_score(episode_index, mean_score):
    if episode_index % 10 == 0:
        print(f"\rEpisode: {episode_index}\tAverage Score: {mean_score}")


@dataclass
class ScoreTrackers:
    scores: list = field(default_factory=list)
    scores_window: deque = deque(maxlen=100)
