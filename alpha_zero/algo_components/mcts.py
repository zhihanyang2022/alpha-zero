from copy import deepcopy
import random

from games.abstract_game import Game
from algo_components.node import Node


def random_policy(game: Game) -> dict:
    valid_moves = game.get_valid_moves()
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
        move = sample(random_policy(game))
        done, winner = game.evolve(move)
        if done:
            break
    if winner == 0:  # no winner
        return 0
    else:
        return 1 if winner == current_player else -1


def mcts_one_iter(game: Game, root: Node):

    # each node represent a game state; each edge represents a move

    game, node = deepcopy(game), root

    while True:
        if node.is_leaf():
            break
        else:
            move, node = node.select()  # move is the edge
            done, winner = game.evolve(move)

    if node.is_root():  # definitely not done
        node.expand(guide=random_policy(game))
        leaf_value = rollout(game)
    else:
        if not done:
            node.expand(guide=random_policy(game))  # expansion
            leaf_value = rollout(game)  # rollout
        else:
            # no expansion
            # degenerate rollout
            if winner == 0:  # no winner
                leaf_value = 0
            else:
                leaf_value = 1 if winner == game.get_current_player() else -1

    node.backup(-leaf_value)  # although it's current player's round, the action led to this node was taken by opponent
