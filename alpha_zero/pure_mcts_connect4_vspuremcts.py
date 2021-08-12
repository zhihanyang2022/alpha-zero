from algo_components import play_one_game_against_pure_mcts
from games import Connect4

for i in range(10):
    outcome = play_one_game_against_pure_mcts(Connect4, 10000, 10000, None, first_hand="alphazero", has_audience=True)
    print(outcome)
