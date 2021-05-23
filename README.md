# Deep Reinforcement Learning Nanodegree Program Project One: Navigation

## Environment Details

The environment for this task consists of observations with  a state space vector of 37 dimensions which provides the agent with information such as their velocity as well was a ray trace that shoots out and helps them to detect and identify objects along their forward movement vector in 3d space. 

There is an action space consisting of only 4 possible actions, numbered 0 to 3. these correspond to movements the agent can take in the 3d world, either forward, backward, left or right. 

The task is episodic rather than continuous, so within an episode for an optimal policy the agent select the appropriate action in each time step to maximize the number of yellow bananas it collects based on its observations, while also minimizing the number of blue bananas it collects. 

Specifically, the task will be considered 'solved' when the agent is able to maintain an average score of +13 over 100 consecutive episodes. 

## Instructions For Installing Dependencies Or Downloading Files necessary To Run My Project

To run the Unity and train the agent there are a few dependencies to be installed. Most of the setup leverages the python/ directory 
of the Udacity Deep Reinforcement Learning course repository, Only the pytorch version used was no longer listed in Pypi so I 
downloaded it as a local wheel and saved it into what is mostly a copy of that directory. 
First of all, this was tested on a 64 bit Windows machine, the platform shouldn't matter but I haven't tested on OSX or Linux. 

The runtime environment for the project requires python version 3.6 and the unityagent package depends on an old version of pytorch. The way I ensured I could install these particular versions of python and the packages was using conda, though I imagine I could have done it
with a local python distribution or a virtual environment. I recommend conda. 

download it for your appropriate platform from here https://www.anaconda.com/products/individual, you probably only need the conda package manager for this, especially if you are OSX or Linux but I ran my examples from within their powershell client, so I'm pointing to that here. Now you are ready to install your dependencies. . . 

create a conda environment `conda create -n <name it something> python==3.6`
now install the dependencies, you need to run this from within the python/ directory because the local wheel's path is relative to that: 
`<from repo root> cd python`
`pip install -r requirements.txt # this will download the old pytorch version as well as unity agents, removed the setup.py as we don't need to package anything anymore`

a good way to test that you have been able to successfully install dependencies os to go back to the root of the project and run:
`udacity_navigation_code.py`
you should see the udacity Bananas environment fire up inside a new unity mlagent application window and an agent will randomly take actions for an episode.


## I Have Instructions On How To Run The Project, How To Train The Agent In Other Words

<TBD>
