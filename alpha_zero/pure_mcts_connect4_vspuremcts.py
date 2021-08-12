import numpy as np
from algo_components import play_one_game_against_pure_mcts
from games import Connect4
from multiprocessing import Pool


def func(num_iter, first_hand):
    outcome = play_one_game_against_pure_mcts(
        Connect4,
        num_mcts_iters_pure=5000,
        num_mcts_iters_alphazero=num_iter,
        policy_value_fn=None,
        first_hand=first_hand,
    )
    if outcome == 1:
        return 1
    else:
        return 0


if __name__ == '__main__':

    num_games = 10

    for num_iter in [500, 5000, 10000]:
        for first_hand in ["pure_mcts", "alphazero"]:
            with Pool(num_games) as p:
                wins = p.starmap(func, [(num_iter, first_hand)] * num_games)
            print(num_iter, first_hand, np.mean(wins))
