from games import Connect4
from algo_components import PolicyValueNet, Buffer


num_games_for_training = 2000
num_grad_steps = 5
eval_freq = 50
buffer_size = 10000
batch_size = 512
num_mcts_iter = 500

game = Connect4()
policy_value_net = PolicyValueNet(game.board_width, game.board_height)
buffer = Buffer(buffer_size, batch_size, geometric_augment=True)

for game_idx in range(num_games_for_training):

    states, mcts_probs, zs = generate_self_play_data(
        game_klass=Connect4,
        num_mcts_iter=num_mcts_iter,
        guiding_policy=policy_value_net.policy_value_fn
    )
    buffer.push(states, mcts_probs, winner)

    if buffer.is_ready():
        for n in range(num_grad_steps):
            states_b, mcts_probs_b, zs_b = buffer.sample()
            probs, value = nn(states_b)

            # do we set probs for invalid moves to zero
            # compute loss for (probs, mcts_probs) and (value, winner_z)
            # use adam optimizer with weight decay to minimize the loss

    if (game_idx + 1) % eval_freq:
        # use low temperature here
        play(alpha_zero, pure_mcts)
