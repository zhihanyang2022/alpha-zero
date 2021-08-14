from algo_components import play_one_game_against_pure_mcts
from games import Connect4
from multiprocessing import Pool
import torch
from algo_components.policy_value_net import PolicyValueNet


def func(first_hand):
    policy_value_net = PolicyValueNet(*Connect4().board.shape)
    policy_value_net.load_state_dict(torch.load("trained_models/pvnet_2000.pth", map_location=torch.device('cpu')))
    outcome = play_one_game_against_pure_mcts(
        Connect4,
        num_mcts_iters_pure=10000,
        policy_value_fn=policy_value_net.policy_value_fn,
        first_hand=first_hand,
    )
    return outcome


if __name__ == '__main__':

    num_games = 20

    for first_hand in ["pure_mcts", "alphazero"]:
        with Pool(num_games) as p:
            outcomes = p.map(func, [first_hand] * num_games)
        print(first_hand, outcomes)
