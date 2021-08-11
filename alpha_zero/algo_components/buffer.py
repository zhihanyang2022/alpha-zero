import numpy as np
import torch

from algo_components.utils import get_device


def maybe_fliplr(array, do_the_flip):
    if do_the_flip:
        return np.flip(array, axis=2)
    else:
        return array


class Buffer:

    def __init__(self, board_width, board_height, buffer_size, batch_size):
        # storage
        self.states = np.zeros((buffer_size, board_width, board_height))
        self.pi_vecs = np.zeros((buffer_size, board_width * board_height))
        self.zs = np.zeros((buffer_size, board_width * board_height))
        # hyper-parameters
        self.board_width = board_width
        self.board_height = board_height
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        # pointers
        self.ptr = 0
        self.cnt = 0  # number of items added so far

    def push(self, states_org, pi_vecs_org, zs_org):

        game_duration = states_org.shape[0]
        pi_vecs_org = pi_vecs_org.reshape(states_org.shape)

        for num_rots in [0, 1, 2, 3]:
            for do_the_flip in [True, False]:
                states = maybe_fliplr(np.rot90(states_org, k=num_rots, axes=(1, 2)), do_the_flip)
                pi_vecs = maybe_fliplr(np.rot90(pi_vecs_org, k=num_rots, axes=(1, 2)), do_the_flip)\
                    .reshape(game_duration, -1)
                for i in range(game_duration):
                    self.states[self.ptr] = states[i]
                    self.pi_vecs[self.ptr] = pi_vecs[i]
                    self.zs[self.ptr] = zs_org[i]
                    self.ptr = (self.ptr + 1) % self.buffer_size
                    self.cnt = min(self.cnt + 1, self.buffer_size)

    def is_ready(self):
        return self.batch_size <= self.cnt

    def sample(self):
        indices = np.random.randint(self.cnt, size=self.batch_size)
        states = torch.tensor(self.states[indices]).view(self.batch_size, 1, self.board_width, self.board_height)\
            .to(get_device()).float()
        pi_vecs = torch.tensor(self.pi_vecs[indices]).view(self.batch_size, self.board_width * self.board_height)\
            .to(get_device()).float()
        zs = torch.tensor(self.zs[indices]).view(self.batch_size, self.board_width * self.board_height)\
            .to(get_device()).float()
        return states, pi_vecs, zs
