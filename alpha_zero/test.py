from ast import literal_eval
import time

from games.tic_tac_toe import TicTacToe
from algo_components.node import Node
from algo_components.mcts import mcts_one_iter


game = TicTacToe()

while True:

    print(game)

    if game.current_player == -1:

        root = Node(parent=None, prior_prob=1.0)

        start = time.perf_counter()
        for _ in range(2000):
            mcts_one_iter(game, root)
        end = time.perf_counter()
        print(end - start)

        move = root.get_move()

    else:

        move = literal_eval(input("What's your move: "))

    done, winner = game.evolve(move)
    if done:
        print(game)
        print(f"Winner is player {winner}.")
        break
