from algo_components import generate_self_play_data
from games import Connect4

states, pi_vecs, zs = generate_self_play_data(Connect4, num_mcts_iter=5000, has_audience=True)

