import numpy as np
import torch
from networks import PolicyValueNetwork
from copy import deepcopy


device = 'cuda'


class AlphaZeroAgent:

	def __init__(self):

		self.nn_policy = PolicyValueNetwork(board_width=8, board_height=8)
		self.num_rollouts_per_action = 50

	def get_action_from_nn_policy(self, game) -> int:

		board_repr = torch.from_numpy(game.board_repr).unsqueeze(0).unsqueeze(0).to(device)

		with torch.no_grad():
			action_probs = self.nn_policy(board_repr, mode="act").cpu().numpy()

		action_probs[game.invalid_actions] = 0
		action_probs = action_probs / np.sum(action_probs)
		action = np.random.choice(game.all_actions, p=action_probs)

		return action

	def get_action_from_rollout_policy(self, game):

		# do policy evaluation and policy improvement at the same time
		# but improved policy is difficult to improve fruther since it
		# is a rollout policy built from current policy

		# the action values are from the perspective of the current player

		action_values_sum = np.zeros((len(game.valid_actions), ))
		action_values_cnt = np.zeros((len(game.valid_actions), ))

		for i, action in enumerate(game.valid_actions):

			game_copy_for_action = deepcopy(game)
			finished, reward = game_copy_for_action.step(action)

			if finished:
				action_values_sum[i] += reward
				action_values_cnt[i] += 1
			else:
				for j in range(self.num_rollouts_per_action):
					game_copy_for_rollout = deepcopy(game_copy_for_action)
					while True:
						action = self.get_action_from_nn_policy(game_copy_for_rollout)
						finished, reward = game_copy_for_rollout.step(action)
						if finished:
							action_values_sum[i] += reward * (1 if game.current_player == game_copy_for_rollout.current_player else -1)
							action_values_cnt[i] += 1
							break

		action_values_mean = action_values_sum / action_values_cnt
		action_idx = np.argmax(action_values_mean)
		action = game.valid_actions[action_idx]

		pi_vec = np.zeros((len(game.all_actions), ))
		pi_vec[action] = 1.0

		return action, pi_vec

	def project_rollout_policy_to_nn_policy(self, batch):
		# project rollout policy to nn to enable further opt
		pass
		# project to better policy back to neural network space so taht we can use MCTS again
		# each batch contains (s, pi_vec, z)

	def save_nn_policy(self):
		pass


class Buffer:

	def __init__(self):
		pass

	def push(self, board_state, pi_vec):
		pass

	def finalize_episode(self, final_reward):
		pass
		# also does some invariant transformations good

	def sample(self):
		pass
		# sample a batch for AlphaZero.project_rollout_policy_to_nn

