from QLmodel.config import config
import numpy as np
import pandas as pd
import itertools

class Env:
	def __init__(self, df):
		self.df = df
		self.n = len(df)
		self.current_idx = 0
		self.action_space = [0, 1, 2] # BUY, SELL, HOLD
		self.invested = 0

		self.states = self.df[config.STOCKS].to_numpy()
		self.rewards = self.df[config.TARGET].to_numpy()
		self.total_buy_and_hold = 0

	def reset(self):
		self.current_idx = 0
		self.total_buy_and_hold = 0
		return self.states[self.current_idx]

	def step(self, action):
		# need to return (next_state, reward, done)

		self.current_idx += 1
		if self.current_idx >= self.n:
			raise Exception("Episode already done")

		if action == 0: # BUY
			self.invested = 1
		elif action == 1: # SELL
			self.invested = 0
		
		# compute reward
		if self.invested:
			reward = self.rewards[self.current_idx]
		else:
			reward = 0

		# baseline
		self.total_buy_and_hold += float(self.rewards[self.current_idx])

		# state transition
		next_state = self.states[self.current_idx]

		done = (self.current_idx == self.n - 1)
		return next_state, reward, done

class StateMapper:
	def __init__(self, env, n_bins=6, n_samples=10000):
		# first, collect sample states from the environment
		states = []
		done = False
		s = env.reset()
		self.D = len(s) # number of elements we need to bin
		states.append(s)
		for _ in range(n_samples):
			a = np.random.choice(env.action_space)
			s2, _, done = env.step(a)
			states.append(s2)
			if done:
				s = env.reset()
				states.append(s)

		# convert to numpy array for easy indexing
		states = np.array(states)

		# create the bins for each dimension
		self.bins = []
		for d in range(self.D):
			column = np.sort(states[:,d])

			# find the boundaries for each bin
			current_bin = []
			for k in range(n_bins):
				boundary = column[int(n_samples / n_bins * (k + 0.5))]
				current_bin.append(boundary)

			self.bins.append(current_bin)


	def transform(self, state):
		x = np.zeros(self.D)
		for d in range(self.D):
			x[d] = int(np.digitize(state[d], self.bins[d]))
		return tuple(x)


	def all_possible_states(self):
		list_of_bins = []
		for d in range(self.D):
			list_of_bins.append(list(range(len(self.bins[d]) + 1)))
		# print(list_of_bins)
		return itertools.product(*list_of_bins)

class Agent:
	def __init__(self, action_size, state_mapper):
		self.action_size = action_size
		self.gamma = 0.8  # discount rate
		self.epsilon = 0.1
		self.learning_rate = 1e-1
		self.state_mapper = state_mapper

		# initialize Q-table randomly
		self.Q = {}
		for s in self.state_mapper.all_possible_states():
			s = tuple(s)
			for a in range(self.action_size):
				self.Q[(s,a)] = np.random.randn()

	def act(self, state):
		if np.random.rand() <= self.epsilon:
			return np.random.choice(self.action_size)

		s = self.state_mapper.transform(state)
		act_values = [self.Q[(s,a)] for a in range(self.action_size)]
		return np.argmax(act_values)  # returns action

	def train(self, state, action, reward, next_state, done):
		s = self.state_mapper.transform(state)
		s2 = self.state_mapper.transform(next_state)

		if done:
			target = reward
		else:
			act_values = [self.Q[(s2,a)] for a in range(self.action_size)]
			target = reward + self.gamma * np.amax(act_values)

		# Run one training step
		self.Q[(s,action)] += self.learning_rate * (target - self.Q[(s,action)])

def play_one_episode(agent, env, is_train):
	state = env.reset()
	done = False
	total_reward = 0

	while not done:
		action = agent.act(state)
		next_state, reward, done = env.step(action)
		total_reward += reward
		if is_train:
			agent.train(state, action, reward, next_state, done)
		state = next_state

	return float(total_reward)