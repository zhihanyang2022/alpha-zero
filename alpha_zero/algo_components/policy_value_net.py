import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from algo_components.utils import get_device


class PolicyValueNet(nn.Module):
    """policy-value network module"""
    def __init__(self, board_width, board_height):
        super().__init__()

        self.board_width = board_width
        self.board_height = board_height
        # common layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4*board_width*board_height,
                                 board_width*board_height)
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*board_width*board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4*self.board_width*self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2*self.board_width*self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_act, x_val

    def policy_value_fn(self, first_person_view: np.array, valid_moves: list, return_pi_vec: bool = False) -> tuple:

        fpv_torch = torch.from_numpy(first_person_view).unsqueeze(0).unsqueeze(0).to(get_device())
        assert fpv_torch.shape == (1, 1, self.board_height, self.board_width)

        with torch.no_grad():
            x_act, x_val = self(fpv_torch.float())

        p_vec = np.exp(x_act.squeeze().cpu().numpy())

        probs = []
        for move in valid_moves:
            index = move[0] * self.board_width + move[1]
            probs.append(p_vec[index])

        if return_pi_vec is False:  # as guiding policy
            return {"moves": valid_moves, "probs": probs}, float(x_val)
        else:  # for evaluation
            return p_vec, float(x_val)
