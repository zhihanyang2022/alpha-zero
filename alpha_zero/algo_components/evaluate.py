from numpy import unravel_index

from algo_components.node import Node
from algo_components.mcts import mcts_one_iter


def play_one_game_against_pure_mcts(
        game_klass,
        num_mcts_iters_pure,
        policy_value_fn,
        first_hand,
        has_audience=False
):

    game = game_klass()  # at the start, current player is always 1

    if first_hand == "pure_mcts":
        pure_mcts = game.get_current_player()
        alphazero = game.get_current_player() * -1
    elif first_hand == "alphazero":
        pure_mcts = game.get_current_player() * -1
        alphazero = game.get_current_player()
    else:
        raise NotImplementedError

    while True:

        if has_audience:
            print(game)

        if game.get_current_player() == pure_mcts:

            root = Node(parent=None, prior_prob=1.0)

            for _ in range(num_mcts_iters_pure):
                mcts_one_iter(game, root)

            move = root.get_move(temp=0)

        else:

            pi_vec, _ = policy_value_fn(game.board * game.get_current_player(), game.get_valid_moves(), True)
            pi_vec[pi_vec < 0.01] = 0

            pi_vec_square = pi_vec.reshape(game.board.shape)

            move = unravel_index(pi_vec_square.argmax(), pi_vec_square.shape)

        done, winner = game.evolve(move)
        if done:
            if has_audience:
                print(game)
            break

    # return the outcome of the game from perspective of alphazero
    if winner == pure_mcts:
        return -1
    elif winner == alphazero:
        return 1
    elif winner == 0:
        return 0
    else:
        raise NotImplementedError
