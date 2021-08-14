from algo_components import play_one_game_against_pure_mcts
from games import Connect4
import torch
from algo_components.policy_value_net import PolicyValueNet


def func(first_hand):
    policy_value_net = PolicyValueNet(*Connect4().board.shape)
    policy_value_net.load_state_dict(torch.load("trained_models/pvnet_2000.pth", map_location=torch.device('cpu')))
    outcome = play_one_game_against_pure_mcts(
        Connect4,
        num_mcts_iters_pure=10000,
        num_mcts_iters_alphazero=500,
        policy_value_fn=policy_value_net.policy_value_fn,
        first_hand=first_hand,
        has_audience=True
    )
    return outcome


if __name__ == '__main__':

    for first_hand in ["alphazero"]:
        outcomes = []
        for i in range(20):
            outcome = func(first_hand); print(outcome)
            outcomes.append(outcome)
        print(first_hand, outcomes)
