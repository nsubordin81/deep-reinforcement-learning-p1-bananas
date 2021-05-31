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


def environment(timestep, unity_params, action=None):
    # side effecting here for some defensive coding
    assert (timestep != 0 and not action) or (
        timestep == 0 and action
    ), "Either there was a value passed for action at time step 0 or an action \
    was missing at a later time step"

    if timestep == 0 and not action:
        return unity_params.env.reset(train_mode=True)[unity_params.brain_name]
    else:
        return unity_params.env.step(action)[unity_params.brain_name]

