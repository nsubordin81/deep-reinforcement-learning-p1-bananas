import functools
from collections import deque
from functools import dataclass

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
        _report_score(kwargs["episode_index"], score_trackers.scores_window)

    return wrapper


def _report_score(episode_index, scores_window):
    if episode_index % 10 == 0:
        print(f"\rEpisode: {i}\tAverage Score: {np.mean(scores_window)}")


@dataclass
class ScoreTrackers:
    scores: list = []
    scores_window: deque = deque(maxlen=100)
