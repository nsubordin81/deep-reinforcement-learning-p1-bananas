# Deep Reinforcement Learning Nanodegree Program Project One: Navigation

## Environment Details

The environment for this task consists of observations with  a state space vector of 37 dimensions which provides the agent with information such as their velocity as well was a ray trace that shoots out and helps them to detect and identify objects along their forward movement vector in 3d space. 

There is an action space consisting of only 4 possible actions, numbered 0 to 3. these correspond to movements the agent can take in the 3d world, either forward, backward, left or right. 

The task is episodic rather than continuous, so within an episode for an optimal policy the agent select the appropriate action in each time step to maximize the number of yellow bananas it collects based on its observations, while also minimizing the number of blue bananas it collects. 

Specifically, the task will be considered 'solved' when the agent is able to maintain an average score of +13 over 100 consecutive episodes. 

## Instructions For Installing Dependencies Or Downloading Files necessary To Run My Project

**Notes:** To run the Unity and train the agent there are a few dependencies to be installed. Most of the setup leverages the python/ directory 
of the Udacity Deep Reinforcement Learning course repository, Only the pytorch version used was no longer listed in Pypi and the wheel is over 500Mb
so I've included instructions to downloa it as a local wheel and save it to a wheels/ dir within the python directory that holds the project configuration
First of all, this was tested on a 64 bit Windows machine, the platform shouldn't matter but I haven't tested on OSX or Linux. 

The runtime environment for the project requires python version 3.6 and the unityagent package depends on an old version of pytorch. The way I ensured I could install these particular versions of python and the packages was using conda, though I imagine I could have done it
with a local python distribution or a virtual environment. I recommend using conda, and for all commands I've provided what works in conda. If you were using a python interpreter you installed directly onto your host machine for example, you might need to use more specific commands like `python3.6 -m pip` instead of just python or pip in the below commands

1. download conda for your appropriate platform from here https://www.anaconda.com/products/individual, you probably only need the conda package manager for this, especially if you are OSX or Linux but I ran my examples from within their powershell client, so I'm pointing to their full distribution here. Now you are ready to install your dependencies. . . 

2. clone this repository to your machine: `git clone git@github.com:nsubordin81/deep-reinforcement-learning-p1-bananas.git`

3. create a conda environment `conda create -n <name it something> python==3.6`
`conda activate <whatever you chose to call it>`
all the rest of the install steps will be performed from within python/, a directory whose contents are based on the setup configuration python/ dir Udacity course companion repo https://github.com/udacity/deep-reinforcement-learning/tree/master/python
 
4. `<from repo root> cd python`

5. This next step is important. Since the project was first created, it seems Pypi has delisted the 0.4.0 version of torch from their index. As a result, I am including steps here to download the wheel directly and have it locally install along with your requirements. first, get the wheel from pypi's archives with the following command: 
`python -m pip download torch==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html`
then the file should be in your python/ directory. The requirements.txt though expects it to be under wheels/ so you'll want to copy it there: 
`cp <name of file downloaded> wheels/`

6. now that you have the wheel in the right location, you should be able to use the requirements.txt to install all of the dependencies, you need to run this from within the python/ directory because the local wheel's path is relative to that location and expects python/ as the working directory: 
`pip install -r requirements.txt # this will download the old pytorch version as well as unity agents, removed the setup.py as we don't need to package anything anymore`

7. a good way to test that you have been able to successfully install dependencies os to go back to the root of the project and run:
`python udacity_navigation_code.py`
you should see the udacity Bananas environment fire up inside a new unity mlagent application window and an agent will randomly take actions for an episode.

8. The deep q code has more dependencies than that example, however, and it still won't run if, like me, you were on Windows 10 and don't have CUDA support for your nvidia GPU. If you don't have an nvidia gpu, you will need to change the requirements.txt to use the cpu only version of pytorch, so replace torch with `pytorch-cpu`, but if you have a GPU and want to use it you may need to install CUDA support if you haven't already. I used this method to get the cuda libraries (thank you to stack overflow post accepted answer here: https://stackoverflow.com/questions/49395819/import-torch-giving-error-from-torch-c-import-dll-load-failed-the-specif?noredirect=1&lq=1): (Again, this is just something you might run up against if you are running a similar Windows OS situation to what I have and you have an Nvidia GPU that doesn't have these DLLs already loaded)
    1. download the tar file containing the DLLs, https://anaconda.org/anaconda/intel-openmp/files
    2. extract them using the method of your choice,
    3. copy the contants of Library/bin in the extracted folder into this path on your OS: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin`
    4. add that path to your Path Environment Variable. 
`

## Instructions On How To Run The Project, How To Train The Agent In Other Words

<TBD>
