from typing import Callable, Tuple
import numpy as np
from algo_components.node import Node
from algo_components.mcts import mcts_one_iter


def generate_self_play_data(
        game_klass: Callable,
        num_mcts_iter: int,
        policy_value_fn: Callable
) -> Tuple[np.array, np.array, np.array]:

    """Generate self-play data using guided MCTS"""

    game = game_klass()

    states, pi_vecs = [], []

    while True:

        state = game.get_first_person_view()  # for nn

        root = Node(parent=None, prior_prob=1.0)
        for _ in range(num_mcts_iter):
            mcts_one_iter(game, root,
                          policy_fn=None, policy_value_fn=policy_value_fn)

        move, pi_vec = root.get_move_and_pi_vec(game.board.shape[0], game.board.shape[1], temp=1)  # for nn

        states.append(state)  # for nn
        pi_vecs.append(pi_vec)  # for nn

        done, winner = game.evolve(move)
        if done:
            break

    game_duration = len(states)

    if winner == 0:
        zs = [0] * game_duration
    else:  # there exists a winner
        zs = []
        player_of_final_move = game.get_current_player() * -1
        z = 1 if player_of_final_move == winner else -1
        for _ in range(game_duration):
            zs.append(z)
            z *= -1
        zs = list(reversed(zs))  # reverse to correct temporal order

    return states, pi_vecs, zs
