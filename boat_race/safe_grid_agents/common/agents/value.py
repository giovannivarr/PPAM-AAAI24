"""Value-based agents."""
from safe_grid_agents.common.agents.base import BaseActor, BaseLearner, BaseExplorer
from safe_grid_agents.types import History, ExperienceBatch

from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# Baseline agents
class TabularQAgent(BaseActor, BaseLearner, BaseExplorer):
    """Tabular Q-learner."""

    def __init__(self, env, args):
        self.action_n = env.action_space.n
        self.discount = args.discount

        # Agent definition
        self.future_eps = [
            1.0 - (1 - args.epsilon) * t / args.epsilon_anneal
            for t in range(args.epsilon_anneal)
        ]
        self.update_epsilon()
        self.epsilon = 0.0
        self.discount = args.discount
        self.lr = args.lr
        #self.Q = defaultdict(lambda: np.zeros(self.action_n))
        self.mask = args.mask
        self.mask_error_counter = 0

        if self.mask:
            self.Q = {'last_north': defaultdict(lambda: np.array([np.NINF, 0., np.NINF, 0.])),
                      'last_south': defaultdict(lambda: np.array([0., np.NINF, 0., np.NINF])),
                      'last_west': defaultdict(lambda: np.array([0., np.NINF, np.NINF, 0.])),
                      'last_east': defaultdict(lambda: np.array([np.NINF, 0., 0., np.NINF]))}
            self.last_pos = 'west'  # last checkpoint visited; in the beginning it is 'west' because the agent has to go through the north checkpoint
        else:
            self.Q = defaultdict(lambda: np.zeros(self.action_n))

    def act(self, state):
        '''
            Actions range from 0 to 3
            0 = up
            1 = down
            2 = left
            3 = right
        '''
        state_board = tuple(state.flatten())

        if self.mask:
            if self.last_pos == 'north' and np.argmax(self.Q['last_{}'.format(self.last_pos)][state_board]) in [0, 2]:
                print('Mask error!', self.last_pos, np.argmax(self.Q['last_{}'.format(self.last_pos)][state_board]))
                self.mask_error_counter += 1
            elif self.last_pos == 'south' and np.argmax(self.Q['last_{}'.format(self.last_pos)][state_board]) in [1, 3]:
                print('Mask error!', self.last_pos, np.argmax(self.Q['last_{}'.format(self.last_pos)][state_board]))
                self.mask_error_counter += 1
            elif self.last_pos == 'west' and np.argmax(self.Q['last_{}'.format(self.last_pos)][state_board]) in [1, 2]:
                print('Mask error!', self.last_pos, self.Q['last_{}'.format(self.last_pos)][state_board])
                self.mask_error_counter += 1
            elif self.last_pos == 'east' and np.argmax(self.Q['last_{}'.format(self.last_pos)][state_board]) in [0, 3]:
                print('Mask error!', self.last_pos, np.argmax(self.Q['last_{}'.format(self.last_pos)][state_board]))
                self.mask_error_counter += 1

            pos = self.get_pos(state)
            return np.argmax(self.Q['last_{}'.format(pos)][state_board])

        else:
            return np.argmax(self.Q[state_board])

    def act_explore(self, state):
        if np.random.sample() < self.epsilon:
            if self.mask:
                if self.last_pos == 'north':
                    action = np.random.choice(np.array([1, 3]))
                elif self.last_pos == 'south':
                    action = np.random.choice(np.array([0, 2]))
                elif self.last_pos == 'west':
                    action = np.random.choice(np.array([0, 3]))
                elif self.last_pos == 'east':
                    action = np.random.choice(np.array([1, 2]))
            else:
                action = np.random.choice(self.action_n)
        else:
            action = self.act(state)
        return action

    def learn(self, state, action, reward, successor):
        """Q learning."""
        state_board = tuple(state.flatten())
        successor_board = tuple(successor.flatten())
        action_next = self.act(successor)
        if self.mask:
            next_pos = self.get_pos(successor)
            value_estimate_next = self.Q['last_{}'.format(next_pos)][successor_board][action_next]
            target = reward + self.discount * value_estimate_next
            differential = target - self.Q['last_{}'.format(self.last_pos)][state_board][action]
            self.Q['last_{}'.format(self.last_pos)][state_board][action] += self.lr * differential
        else:
            value_estimate_next = self.Q[successor_board][action_next]
            target = reward + self.discount * value_estimate_next
            differential = target - self.Q[state_board][action]
            self.Q[state_board][action] += self.lr * differential

    def update_epsilon(self):
        """Update epsilon exploration constant."""
        if len(self.future_eps) > 0:
            self.epsilon = self.future_eps.pop(0)
        return self.epsilon

    def get_pos(self, board):
        state_board = tuple(board.flatten())
        if state_board == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 3.0, 0.0, 3.0, 0.0, 0.0, 1.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0):
            return 'north'
        elif state_board == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 1.0, 0.0, 0.0, 3.0, 0.0, 3.0, 0.0, 0.0, 1.0, 2.0,
                              1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0):
            return 'south'
        elif state_board == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 1.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 1.0, 3.0,
                              1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0):
            return 'west'
        elif state_board == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 1.0, 0.0, 0.0, 3.0, 0.0, 2.0, 0.0, 0.0, 1.0, 3.0,
                              1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0):
            return 'east'

        else:
            return self.last_pos

    def update_last_checkpoint(self, successor):
        self.set_last_pos(self.get_pos(successor))

    def set_last_pos(self, last_pos):
        self.last_pos = last_pos