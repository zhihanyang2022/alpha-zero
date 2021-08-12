from ast import literal_eval
import time
import torch

# from games.tic_tac_toe import TicTacToe
from games.connect4 import Connect4
from algo_components.node import Node
from algo_components.mcts import mcts_one_iter
from algo_components.policy_value_net import PolicyValueNet


game = Connect4()

policy_value_net = PolicyValueNet(*game.board.shape)
policy_value_net.load_state_dict(torch.load("trained_models/pvnet_1200.pth", map_location=torch.device('cpu')))

while True:

    print(game)

    if game.current_player == -1:

        root = Node(parent=None, prior_prob=1.0)

        start = time.perf_counter()
        for _ in range(500):
            mcts_one_iter(game, root, policy_value_fn=policy_value_net.policy_value_fn)
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
