import numpy as np

import gym
from gym.utils import seeding
from gym import spaces

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

from collections import Counter
import logging



class MastermindEnv(gym.Env):
    values = 6
    size = 4
    guess_max = 12

    def __init__(self):
        self.values = 6
        self.size = 4
        self.guess_max = 15
        self.target = None
        self.guess_count = None
        self.observation = None

        self.observation_space = spaces.Tuple([spaces.Discrete(3) for _ in range(4)])
        self.action_space = spaces.Tuple([spaces.Discrete(self.values) for _ in range(4)])

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_observation(self, action):
        match_idxs = set(idx for idx, ai in enumerate(action) if ai == self.target[idx])
        n_correct = len(match_idxs)
        g_counter = Counter(self.target[idx] for idx in range(self.size) if idx not in match_idxs)
        a_counter = Counter(action[idx] for idx in range(self.size) if idx not in match_idxs)
        n_white = sum(min(g_count, a_counter[k])for k, g_count in g_counter.items())
        return tuple([0] * (self.size - n_correct - n_white) + [1] * n_white + [2] * n_correct)

    def step(self, action):
        assert self.action_space.contains(action)
        self.guess_count += 1
        done = action == self.target or self.guess_count >= self.guess_max
        if done and action == self.target:
            reward = 1
        else:
            reward = 0
        return self.get_observation(action), reward, done, {}

    def reset(self):
        self.target = self.action_space.sample()
        logger.debug("target=%s", self.target)
        self.guess_count = 0
        self.observation = (0,) * self.size
        return self.observation

env = MastermindEnv()
np.random.seed(123)
env.seed(123)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n       

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])      #There is an error here, which is not letting me to import this module both in Colab and local Jupyter

dqn.fit(env, nb_steps=5000, visualize=True, verbose=2)
