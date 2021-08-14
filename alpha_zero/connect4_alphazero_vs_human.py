from ast import literal_eval
import torch
import numpy as np

from games.connect4 import Connect4
from algo_components.policy_value_net import PolicyValueNet
from algo_components import Node, mcts_one_iter


game = Connect4()

policy_value_net = PolicyValueNet(*game.board.shape)
policy_value_net.load_state_dict(torch.load("trained_models/pvnet_2000.pth", map_location=torch.device('cpu')))

print(game)

while True:

    if game.current_player == -1:

        pi_vec, _ = policy_value_net.policy_value_fn(game.board * game.get_current_player(), game.get_valid_moves(), True)
        pi_vec[pi_vec < 0.01] = 0
        print(pi_vec.reshape(game.board.shape))

        root = Node(parent=None, prior_prob=1.0)

        for _ in range(np.random.randint(300, 1000)):  # introduce some stochasticity here
            mcts_one_iter(game, root, policy_value_fn=policy_value_net.policy_value_fn)

        move = root.get_move(temp=0)

    else:

        move = literal_eval(input("What's your move: "))
        move = (move[0] - 1, move[1] - 1)

    done, winner = game.evolve(move)
    print(game)

    if done:
        print(game)
        print(f"Winner is player {winner}.")
        break
