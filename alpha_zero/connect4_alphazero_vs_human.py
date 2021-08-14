from ast import literal_eval
import time
import torch
from numpy import unravel_index

# from games.tic_tac_toe import TicTacToe
from games.connect4 import Connect4
from algo_components.node import Node
from algo_components.mcts import mcts_one_iter
from algo_components.policy_value_net import PolicyValueNet


game = Connect4()

policy_value_net = PolicyValueNet(*game.board.shape)
policy_value_net.load_state_dict(torch.load("trained_models/pvnet_2000.pth", map_location=torch.device('cpu')))

print(game)

while True:

    if game.current_player == 1:

        pi_vec, _ = policy_value_net.policy_value_fn(game.board * game.get_current_player(), game.get_valid_moves(), True)
        pi_vec[pi_vec < 0.01] = 0
        
        pi_vec_square = pi_vec.reshape(game.board.shape)

        move = unravel_index(pi_vec_square.argmax(), pi_vec_square.shape)

    else:

        move = literal_eval(input("What's your move: "))
        move = (move[0] - 1, move[1] - 1)

    done, winner = game.evolve(move)
    print(game)

    if game.get_previous_player() == 1:
        print("Predicted score:", -policy_value_net.policy_value_fn(game.board * game.get_current_player(), game.get_valid_moves())[1])

    if done:
        print(game)
        print(f"Winner is player {winner}.")
        break
