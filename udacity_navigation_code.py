from unityagents import UnityEnvironment
import numpy as np


def main():

    env = UnityEnvironment(file_name="./Banana_Windows_x86_64/Banana.exe")

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    print(f"environment info: {env_info}")

    # number of agents in the environment
    print('number of agents:', len(env_info.agents))

    # number of actions (aside I find it interesting the action space is on the brain not env_info)
    # this is because more or less the brains are entities that can take actions in the environment
    # going to how you would code a unity game they have controller objects and you can either
    # have brains that are player controlled or  agent controlled. So we only have one agent, 
    # and no players, so we just want to have the one brain. But as you can see below, 
    # you have to specify which brain is taking the action in the environment because there
    # could be more than one. 
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # example the state space
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    env_info = env.reset(train_mode=False)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    score = 0                                          # initilize the score

    # this code (this whole file is provided by Udactiy) 
    # has the agent go through a single episode in the environment taking uniform random actions
    while True:
        action = np.random.randint(action_size)        # select an action at random
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if the episode is over
        score += reward                                # update the score
        state = next_state                             # roll over the state to the next time step
        if done:                                       # exit the loop if the episode is done
            break

    print("Score: {}".format(score))



if __name__ == "__main__":
    main()
