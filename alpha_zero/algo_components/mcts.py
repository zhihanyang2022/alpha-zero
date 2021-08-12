from copy import deepcopy
import random

from games import Game
from algo_components.node import Node


def random_policy(valid_moves) -> dict:
    num_valid_moves = len(valid_moves)
    probs = [1 / num_valid_moves] * num_valid_moves
    return {
        "moves": valid_moves,
        "probs": probs
    }


def sample(policy):
    return random.choices(policy["moves"], weights=policy["probs"])[0]


def rollout(game: Game):
    current_player = game.get_current_player()
    while True:
        move = sample(random_policy(game.get_valid_moves()))
        done, winner = game.evolve(move)
        if done:
            break
    if winner == 0:  # no winner
        return 0
    else:
        return 1 if winner == current_player else -1


def mcts_one_iter(game: Game, root: Node, policy_value_fn=None):

    # each node represent a game state; each edge represents a move

    game, node = deepcopy(game), root

    while True:
        if node.is_leaf():
            break
        else:
            move, node = node.select()  # move is the edge
            done, winner = game.evolve(move)

    if node.is_root() or not done:  # when node.is_root, obviously can't be done yet; "not done" is not even checked
        if policy_value_fn is None:
            node.expand(guide=random_policy(game.get_valid_moves()))
            neg_leaf_value = rollout(game)
            node.backup(-neg_leaf_value)
        else:
            guide, leaf_value = policy_value_fn(game.board * game.get_previous_player(), game.get_valid_moves())
            node.expand(guide=guide)
            node.backup(leaf_value)
    else:
        # no expansion
        # degenerate rollout
        if winner == 0:  # no winner
            neg_leaf_value = 0
        else:
            neg_leaf_value = 1 if winner == game.get_current_player() else -1
        node.backup(-neg_leaf_value)
