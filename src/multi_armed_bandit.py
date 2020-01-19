import numpy as np
import math

def push(x, y):
    push_len = len(y)
    assert len(x) >= push_len
    x[:-push_len] = x[push_len:]
    x[-push_len:] = y
    return x


class Multi_armed_bandit:
	"""
	UCB multi armed bandit solution, using moving average reward for the non stationary problem
	see https://www.analyticsvidhya.com/blog/2018/09/reinforcement-multi-armed-bandit-scratch-python/
	for a quick explanation and the original code
	"""
	def __init__(self, actions, exploration_weight=1, moving_avg_len=100):
		"""
		initialize with the list of actions to choose from
		:param actions:
		"""
		n_arms = len(actions)
		self.moving_avg_len = moving_avg_len
		self.sums_of_reward = np.array([0.]*n_arms)
		self.rewards = np.zeros((n_arms, self.moving_avg_len), dtype=float)
		self.n_selections = np.array([1e-300]*n_arms)
		self.total_reward = 0
		self.n_trials = 0
		self.n_arms = n_arms
		self.selected_arms = []
		self.actions = actions
		self.exploration_weight = exploration_weight

	def select_action(self):
		"""
		select next action to be performed
		:return:
		"""
		average_reward = self.sums_of_reward / self.n_selections
		delta = self.exploration_weight*np.sqrt(2 * np.log(self.n_trials + 1) / self.n_selections)
		upper_bound = average_reward + delta
		idx = np.argmax(upper_bound)
		self.selected_arms.append(idx)
		self.n_selections[idx] += 1

		return self.actions[idx]

	def update_reward(self, reward):
		"""
		update statistics with the reward of the last action
		:param reward:
		:return:
		"""
		idx = self.selected_arms[-1]
		self.rewards[idx] = push(self.rewards[idx], [reward])
		self.sums_of_reward[idx] = np.sum(self.rewards[idx])
		self.total_reward += reward
		self.n_trials +=1
