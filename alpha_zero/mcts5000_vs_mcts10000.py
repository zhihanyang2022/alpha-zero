import numpy as np
from algo_components import play_one_game_against_pure_mcts
from games import Connect4
from multiprocessing import Pool


def func(first_hand):
    outcome = play_one_game_against_pure_mcts(
        Connect4,
        num_mcts_iters_pure=1000,
        num_mcts_iters_alphazero=1000,
        policy_value_fn=None,
        first_hand=first_hand,
    )
    return outcome


if __name__ == '__main__':

    num_games = 10

    for first_hand in ["pure_mcts", "alphazero"]:
        with Pool(num_games) as p:
            outcomes = p.map(func, [first_hand] * num_games)
        print(first_hand, np.mean(outcomes))
