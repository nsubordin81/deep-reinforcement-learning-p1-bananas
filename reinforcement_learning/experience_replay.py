import random
from toolz import curry
from collections import namedtuple, deque

import torch
import numpy as np
from utils.shared import device

""" Experience Replay
This will be a stateful, mutable data structure, because we know exactly the scope in which it
is being used, there won't be concurrent updates or accesses, so it is pretty safe and 
more efficient to just have it change. 

the whitepaper discusses a strategy to have a dataset Dt which would hold the most recent 
N experiences in memory, pooled from the episodes (set of non terminal states) into the 
memory. These would be sampled from in a uniformly random fashion as part of an offline
learning strategy where the policy is improved not while it is acting but instead after
it acts enough to build up a pool of experience and then uses that pool to improve with
non sequential experiences that might be repeated.
"""


experience_tuple = namedtuple(
    "Experience", field_names=["state", "action", "reward", "next_state", "done"]
)


@curry
def extract_attribute_as_tensor(
    attribute, sample, device, np_type=None, tensor_type=None
):
    # some hoop jumping here to get implicit casting to work
    vertical_array = np.vstack([attribute(s) for s in sample if s is not None]).astype(
        np_type
    )
    tensor = tensor_type(torch.from_numpy(vertical_array))
    return tensor.to(device)


class ExperienceDataset:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """ right now this is looking pretty much like the ReplayBuffer used in the course
        but with changing some names to remind me more 
        of the whitepaper. can't think of that many changes I would
        make since deque is a great data structure to represent a limited pool. If I want to improve
        the solution by selectively saving and replaying the more useful experiences then 
        I would need modifications.

         Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.action_size = action_size
        self.pool = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)

    def save(self, state, action, reward, next_state, done):
        """ overloading because I'm not sure yet if I want to pass these separately or expect them already combined """
        e = experience_tuple(state, action, reward, next_state, done)
        self.pool.append(e)

    def uniform_random_sample(self):
        """ take a uniform random sample from the pool of experiences """
        experiences = random.sample(self.pool, k=self.batch_size)

        extract = extract_attribute_as_tensor(sample=experiences, device=self.device)
        float_tensor = lambda tensor: tensor.float()
        long_tensor = lambda tensor: tensor.long()
        extract_float_tensor = extract(tensor_type=float_tensor)
        extract_long_tensor = extract(tensor_type=long_tensor)
        extract_unsigned_int_numpy_float_tensor = extract_float_tensor(np_type=np.uint8)

        states = extract_float_tensor(lambda x: x.state)
        actions = extract_long_tensor(lambda x: x.action)
        rewards = extract_float_tensor(lambda x: x.reward)
        next_states = extract_float_tensor(lambda x: x.next_state)
        dones = extract_unsigned_int_numpy_float_tensor(lambda x: x.done)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """ how many experiences are in memory """
        return len(self.pool)

