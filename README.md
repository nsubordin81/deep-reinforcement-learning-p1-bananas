# Deep Reinforcement Learning Nanodegree Program Project One: Navigation

## Attribution

This project is based on a Udacity course, the Deep Reinforcement Learning Nanodegree, and my source code was bootstrapped off of the basis of the Lunar Lander exercise which contained a sample solution I went through before starting. There are sections that directly make use of concepts presented there. Additionally, Udacity used Unity MLAgents environments and I leveraged their example and the ml agents set up code for this Banana example.

## Environment Details

The environment for this task consists of observations with  a state space vector of 37 dimensions which provides the agent with information such as their velocity as well was a ray trace that shoots out and helps them to detect and identify objects along their forward movement vector in 3d space. 

There is an action space consisting of only 4 possible actions, numbered 0 to 3. these correspond to movements the agent can take in the 3d world, either forward, backward, left or right. 

The task is episodic rather than continuous, so within an episode for an optimal policy the agent select the appropriate action in each time step to maximize the number of yellow bananas it collects based on its observations, while also minimizing the number of blue bananas it collects. 

Specifically, the task will be considered 'solved' when the agent is able to maintain an average score of +13 over 100 consecutive episodes. 

## Instructions For Installing Dependencies Or Downloading Files necessary To Run My Project

**Notes:** To run the Unity and train the agent there are a few dependencies to be installed. Most of the setup leverages the python/ directory 
of the Udacity Deep Reinforcement Learning course repository which is basically the mlagents one with a few additional pip dependencies. 

The pytorch version used was no longer listed in Pypi and the wheel is over 500Mb
so I've included instructions to downloa it as a local wheel and save it to a wheels/ dir within the python directory that holds the project configuration

First of all, this was tested on a 64 bit Windows machine, the platform shouldn't matter but I haven't tested on OSX or Linux. 

The runtime environment for the project requires python version 3.6 and the unityagent package depends on an old version of pytorch. The way I ensured I could install these particular versions of python and the packages was using conda, though I imagine I could have done it
with a local python distribution or a virtual environment. I recommend using conda, and for all commands I've provided what works in conda. If you were using a python interpreter you installed directly onto your host machine for example, you might need to use more specific commands like `python3.6 -m pip` instead of just python or pip in the below commands

1. download conda for your appropriate platform from here https://www.anaconda.com/products/individual, you probably only need the conda package manager for this, especially if you are OSX or Linux but I ran my examples from within their powershell client, so I'm pointing to their full distribution here. Now you are ready to install your dependencies. . . 

2. clone this repository to your machine: `git clone git@github.com:nsubordin81/deep-reinforcement-learning-p1-bananas.git`
and `cd deep-reinforcement-learning-p1-bananas.git`

3. create a conda environment `conda create -n <name it something> python=3.6`
`conda activate <whatever you chose to call it>`

5. This next step is important. Since the project was first created, it seems Pypi has delisted the 0.4.0 version of torch from their index for pip. As a result, I am including steps here to install the wheel from conda directly which will be separate from your requirements.txt install step. This workedf for me on a Windows x64 architecture with an Nvidia GTX 1070 family graphics card so I used CUDA 9.2. You may want to do some research depending on your machine's specifications, pytorch has a guide on their page here https://pytorch.org/get-started/previous-versions/
`conda install pytorch=0.4.0 cuda92 -c pytorch`

6. now that you have pytorch installed, you should be able to use the typical pip install -r requirements.txt to install all of the other dependencies with the project root as your working directory: 
`pip install -r requirements.txt`

7. a good way to test that you have been able to successfully install dependencies os to go back to the root of the project and run:
`python udacity_navigation_code.py`
you should see the udacity Bananas environment fire up inside a new unity mlagent application window and an agent will randomly take actions for an episode.

8. The deep q code has more dependencies than that example, however, and it still won't run if, like me, you were on Windows 10 and don't have CUDA support for your nvidia GPU. If you don't have an nvidia gpu, you will need to change your pytorch conda install command to use the cpu only version of pytorch, so replace torch with `pytorch-cpu`, but if you have a GPU and want to use it you may need to install CUDA support if you haven't already. I used this method to get the cuda libraries (thank you to stack overflow post accepted answer here: https://stackoverflow.com/questions/49395819/import-torch-giving-error-from-torch-c-import-dll-load-failed-the-specif?noredirect=1&lq=1): (Again, this is just something you might run up against if you are running a similar Windows OS situation to what I have and you have an Nvidia GPU that doesn't have these DLLs already loaded)
    1. download the tar file containing the DLLs, https://anaconda.org/anaconda/intel-openmp/files
    2. extract them using the method of your choice,
    3. copy the contants of Library/bin in the extracted folder into this path on your OS: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin`
    4. add that path to your Path Environment Variable. 

9. when you installed the requirements, you also installed the project to site-packages with pip, so you should be able to run the few unit tests I've written with `pytest .` in the root of the project. The tests are for the policy portion of the code, as that is where the bulk of the reinforcement learning logic belongs and most of the rest is either just standard pytorch or code to integrate with the unity env. If I go further with this and turn it into a framework, then I'll likely add tests and try to make the framework more agnostic to the source of environment. 

## Instructions On How To Run The Project, How To Train The Agent In Other Words

### Training the agent
Once you've completed the steps in setup above, you can train the agent with `python train_deep_q_agent.py`. This will print out an experience log, every 10 episodes it will print the average score per episode which is just net banana collection both positive and negative across the 300 timesteps. It should also display a plot of the scores in a separate window at the end of the training run assuming it is not interrupted by an error or keyboard interrupt. If the agent reaches the goal of 13 bananas average over 100 episodes, then the weights of the learning network will automatically be saved into a pytorch checkpoint.

### Running the DQN agent doing inference from loaded pre-trainied weights and biases
run `python run_deep_q_agent.py` from project root
