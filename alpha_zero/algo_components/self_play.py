from typing import Callable, Tuple
import numpy as np

from algo_components.node import Node
from algo_components.mcts import mcts_one_iter


def generate_self_play_data(
        game_klass: Callable,
        num_mcts_iter: int,
        high_temp_for_first_n: int,
        policy_value_fn: Callable = None,
        has_audience: bool = False,
) -> Tuple[np.array, np.array, np.array]:

    """Let guided MCTS play against itself"""

    game = game_klass()

    states, pi_vecs = [], []

    num_actions_take = 0

    while True:

        state = game.board * game.get_current_player()  # for nn
        if has_audience:
            print(game)

        root = Node(parent=None, prior_prob=1.0)
        for _ in range(num_mcts_iter):
            mcts_one_iter(game, root, policy_value_fn=policy_value_fn)

        if num_actions_take < high_temp_for_first_n:
            move, pi_vec = root.get_move_and_pi_vec(game.board.shape[0], game.board.shape[1], temp=1, alpha=None)
        else:
            move, pi_vec = root.get_move_and_pi_vec(game.board.shape[0], game.board.shape[1], temp=0, alpha=0.3)

        states.append(state)  # for nn
        pi_vecs.append(pi_vec)  # for nn

        done, winner = game.evolve(move)
        num_actions_take += 1
        if done:
            if has_audience:
                print(game)
            break

    game_duration = len(states)

    if winner == 0:
        zs = [0] * game_duration
    else:  # there exists a winner
        zs = []
        player_of_final_move = game.get_previous_player()
        z = 1 if player_of_final_move == winner else -1
        for _ in range(game_duration):
            zs.append(z)
            z *= -1
        zs = list(reversed(zs))  # reverse to correct temporal order

    states, pi_vecs, zs = map(np.array, [states, pi_vecs, zs])

    # states will have shape (game_duration, board_width, board_height)
    # pi_vecs and zs will have shape (game_duration, board_width * board_height)

    return states, pi_vecs, zs
