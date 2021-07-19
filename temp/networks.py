import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):

    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class PolicyValueNetwork(nn.Module):

    def __init__(self, board_width, board_height):
        super().__init__()

        self.board_width = board_width
        self.board_height = board_height

        self.shared_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.policy_layers = nn.Sequential(
            nn.Conv2d(128, 4, kernel_size=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(4 * board_width * board_height, board_width * board_height)
        )

        self.value_layers = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(2 * board_width * board_height, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, board_repr, mode):
        assert board_repr.shape == (-1, self.board_width, self.board_height)
        temp = self.layers(board_repr)
        logits, value = self.policy_layers(temp), self.value_layers(temp)
        if mode == "train":
            log_prob = F.log_softmax(logits, dim=1)  # more numerically stable for optimization
            return log_prob.view(-1, self.board_width * self.board_height), value.view(-1, 1)
        elif mode == "act":
            prob = F.softmax(logits, dim=1)  # for taking actions
            return prob.view(self.board_width * self.board_height)
        else:
            raise ValueError(f"Mode {mode} is not recognized.")
