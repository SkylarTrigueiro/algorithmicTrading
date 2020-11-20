from QLmodel.model import Agent, Env, play_one_episode, StateMapper
from QLmodel.processing.data_management import load_data
from QLmodel.config import config

import numpy as np
import pandas as pd


def train_model():

	train_data, test_data = load_data(config.STOCKS+config.TARGET)

	train_env = Env(train_data)
	test_env = Env(test_data)

	action_size = len(train_env.action_space)
	state_mapper = StateMapper(train_env)
	agent = Agent(action_size, state_mapper)

	train_rewards = np.empty(config.NUM_EPISODES)
	test_rewards = np.empty(config.NUM_EPISODES)

	for e in range(config.NUM_EPISODES):
		r = play_one_episode(agent, train_env, is_train=True)
		train_rewards[e] = r

		# test on the test set
		tmp_epsilon = agent.epsilon
		agent.epsilon = 0.
		tr = play_one_episode(agent, test_env, is_train=False)
		agent.epsilon = tmp_epsilon
		test_rewards[e] = tr

		print(f"eps: {e + 1}/{config.NUM_EPISODES}, train: {r:.5f}, test: {tr:.5f}")

	print('')
	print(f'Train data algorithmic trading growth: {r:.5f}, Training data buy and hold growth: {train_env.total_buy_and_hold:.5f}')
	print(f'Test data algorithmic trading growth: {tr:.5f}, Training data buy and hold growth: {test_env.total_buy_and_hold:.5f}')

	return train_rewards, test_rewards