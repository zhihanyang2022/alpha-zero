import numpy as np
import torch
import torch.optim as optim

from games import Connect4
from algo_components import PolicyValueNet, Buffer, generate_self_play_data, play_one_game_against_pure_mcts, get_device

# @@@@@@@@@@ hyper-parameters @@@@@@@@@@

game_klass = Connect4
num_games_for_training = 2000
num_grad_steps = 5
eval_freq = 1
buffer_size = 10000
batch_size = 5
num_mcts_iter_alphazero = 500
num_mcts_iter_pure_mcts = 500

# @@@@@@@@@@ important objects @@@@@@@@@@

board_width, board_height = game_klass().board.shape
policy_value_net = PolicyValueNet(board_width, board_height).float().to(get_device())
optimizer = optim.Adam(policy_value_net.parameters(), lr=1e-3, weight_decay=1e-4)  # l2 norm
buffer = Buffer(board_width, board_height, buffer_size, batch_size)

# @@@@@@@@@@ training loop @@@@@@@@@@

for game_idx in range(num_games_for_training):

    print(f'@@@@@ Iteration {game_idx+1} @@@@@')

    states, mcts_probs, zs = generate_self_play_data(
        game_klass=game_klass,
        num_mcts_iter=num_mcts_iter_alphazero,
        policy_value_fn=policy_value_net.policy_value_fn
    )
    buffer.push(states, mcts_probs, zs)

    print('finished self-play')

    print(buffer.cnt)

    if buffer.is_ready():
        for n in range(num_grad_steps):

            states_b, mcts_probs_b, zs_b = buffer.sample()
            predicted_probs, predicted_zs = policy_value_net(states_b)

            loss_part1 = torch.mean((zs_b - predicted_zs) ** 2)
            loss_part2 = torch.mean(torch.sum(mcts_probs_b * torch.log(predicted_probs), dim=1))
            loss = loss_part1 + loss_part2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print('finished training nn')

    if (game_idx + 1) % eval_freq == 0:

        first_hand_scores = []
        for i in range(1):
            score = play_one_game_against_pure_mcts(
                game_klass=game_klass,
                num_mcts_iters_pure=num_mcts_iter_pure_mcts,
                num_mcts_iters_alphazero=num_mcts_iter_alphazero,
                policy_value_fn=policy_value_net.policy_value_fn,
                first_hand="alphazero"
            )
            first_hand_scores.append(score)

        second_hand_scores = []
        for i in range(1):
            score = play_one_game_against_pure_mcts(
                game_klass=game_klass,
                num_mcts_iters_pure=num_mcts_iter_pure_mcts,
                num_mcts_iters_alphazero=num_mcts_iter_alphazero,
                policy_value_fn=policy_value_net.policy_value_fn,
                first_hand="pure_mcts"
            )
            second_hand_scores.append(score)

        mean_first_hand_score = float(np.mean(first_hand_scores))
        mean_second_hand_score = float(np.mean(second_hand_scores))

        if mean_first_hand_score > 0.99 and num_mcts_iter_pure_mcts < 5000:  # basically perfect
            num_mcts_iter_pure_mcts += 500

        print(f"@@@@@ Eval after {game_idx}/{num_games_for_training} "
              f"games against pure-mcts {num_mcts_iter_pure_mcts} @@@@@")
        print(f"First-hand score: {round(mean_first_hand_score, 2)}")
        print(f"Second-hand score: {round(mean_second_hand_score, 2)}")
