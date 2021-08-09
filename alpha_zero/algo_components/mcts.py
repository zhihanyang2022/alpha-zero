from copy import deepcopy
import random
from typing import Callable

from games.abstract_game import Game
from algo_components.node import Node


def random_policy(first_person_view, valid_moves) -> dict:
    num_valid_moves = len(valid_moves)
    probs = [1 / num_valid_moves] * num_valid_moves
    return {
        "moves": valid_moves,
        "probs": probs
    }


def sample(policy):
    return random.choices(policy["moves"], weights=policy["probs"])[0]


def rollout(game: Game, rollout_policy: Callable):
    current_player = game.get_current_player()
    while True:
        move = sample(rollout_policy(game.get_first_person_view(), game.get_valid_moves()))
        done, winner = game.evolve(move)
        if done:
            break
    if winner == 0:  # no winner
        return 0
    else:
        return 1 if winner == current_player else -1


def mcts_one_iter(game: Game, root: Node, policy_fn=random_policy, policy_value_fn=None):

    assert policy_fn is None or policy_value_fn is None

    # each node represent a game state; each edge represents a move

    game, node = deepcopy(game), root

    while True:
        if node.is_leaf():
            break
        else:
            move, node = node.select()  # move is the edge
            done, winner = game.evolve(move)

    if node.is_leaf() or not done:  # when node.is_leaf, obviously can't be done yet; "not done" is not even checked
        if policy_fn is not None:
            node.expand(guide=policy_fn(game.get_first_person_view(), game.get_valid_moves()))
            leaf_value = rollout(game, policy_fn)
        else:
            guide, leaf_value = policy_value_fn(game.get_first_person_view(), game.get_valid_moves())
            node.expand(guide=guide)
    else:
        # no expansion
        # degenerate rollout
        if winner == 0:  # no winner
            leaf_value = 0
        else:
            leaf_value = 1 if winner == game.get_current_player() else -1

    node.backup(-leaf_value)  # although it's current player's round, the action led to this node was taken by opponent
