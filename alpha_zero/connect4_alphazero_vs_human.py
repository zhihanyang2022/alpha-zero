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

print(game)

while True:

    if game.current_player == -1:

        pi_vec, val = policy_value_net.policy_value_fn(game.board * game.get_current_player(), game.get_valid_moves(), True)
        pi_vec[pi_vec < 0.01] = 0
        print(pi_vec.reshape(game.board.shape))
        print(val)

        root = Node(parent=None, prior_prob=1.0)

        start = time.perf_counter()
        for _ in range(2):
            mcts_one_iter(game, root, policy_value_fn=policy_value_net.policy_value_fn)
        end = time.perf_counter()

        print('Decision time:', end - start)

        move = root.get_move(temp=0)

    else:

        move = literal_eval(input("What's your move: "))
        move = (move[0] - 1, move[1] - 1)

    done, winner = game.evolve(move)
    # if game.get_previous_player() == 1:
    #     print("Predicted score:", policy_value_net.policy_value_fn(game.board * game.get_previous_player(), game.get_valid_moves())[1])
    print(game)

    if done:
        print(game)
        print(f"Winner is player {winner}.")
        break
