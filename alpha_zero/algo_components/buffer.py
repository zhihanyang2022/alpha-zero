import numpy as np
import torch

from algo_components.utils import get_device


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

    def push(self, states, pi_vecs, zs):
        game_duration = states.shape[0]
        for i in range(game_duration):
            self.states[self.ptr] = states[i]
            self.pi_vecs[self.ptr] = pi_vecs[i]
            self.zs[self.ptr] = zs[i]
            self.ptr = (self.ptr + 1) % self.buffer_size
            self.cnt = min(self.cnt + 1, self.buffer_size)

    def is_ready(self):
        return self.batch_size <= self.cnt

    def sample(self):
        indices = np.random.randint(self.cnt, size=self.batch_size)
        states = torch.tensor(self.states[indices]).view(self.batch_size, 1, self.board_width, self.board_height).to(get_device()).float()
        pi_vecs = torch.tensor(self.pi_vecs[indices]).view(self.batch_size, self.board_width * self.board_height).to(get_device()).float()
        zs = torch.tensor(self.zs[indices]).view(self.batch_size, self.board_width * self.board_height).to(get_device()).float()
        return states, pi_vecs, zs
