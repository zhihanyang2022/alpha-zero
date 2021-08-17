from ast import literal_eval
import time

from games.connect4 import Connect4
from algo_components.node import Node
from algo_components.mcts import mcts_one_iter


game = Connect4()

while True:

    print(game)

    if game.current_player == -1:

        root = Node(parent=None, prior_prob=1.0)

        start = time.perf_counter()
        for _ in range(10000):
            mcts_one_iter(game, root)
        end = time.perf_counter()

        print('Decision time:', end - start)

        move = root.get_move(temp=0)

    else:

        move = literal_eval(input("What's your move: "))
        move = (move[0] - 1, move[1] - 1)

    done, winner = game.evolve(move)
    if done:
        print(game)
        print(f"Winner is player {winner}.")
        break
