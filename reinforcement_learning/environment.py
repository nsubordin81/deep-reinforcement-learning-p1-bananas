from toolz import curry
from dataclasses import dataclass
from typing import Any

from unityagents import UnityEnvironment


@dataclass
class UnityParams:
    """ class for holding the mlagents specific unity environment parameters """

    env: UnityEnvironment
    brain_name: str
    brain: Any


""" setup_unity_env
abstracting away steps that seem standard for setup unity environment.
Probably not generic enough to be its own function right now, but it 
cleaned up the readability of my main function
"""

# TODO: beyond the project requirements, but environments can have more than one brain
def setup_unity_env(file_name):
    env = UnityEnvironment(file_name)
    brain_name = env.brain_names[0]
    # Interesting thing I learned: Unity MLAgents there can be multiple agents, and also different brains as the unity
    # abstraction for the type of agent doing the task, Player, Heuristic, Internal and External
    brain = env.brains[brain_name]
    return UnityParams(env, brain_name, brain)


""" environment
Just a personal preference, I like to only have one way to get the unity env info and have 
pre: recieves the current timestep as well as the action taken in the time step and the environment info
action: either restarts the unity environment or moves it forward depending on the time step and action
returns: the new environment state corresponding to the action taken
"""


@curry
def environment(brain_name, timestep, step_func, action=None, training=True):
    # side effecting here for some defensive coding
    if timestep == 0 and action is None:
        return step_func(train_mode=training)[brain_name]
    else:
        return step_func(int(action))[brain_name]

