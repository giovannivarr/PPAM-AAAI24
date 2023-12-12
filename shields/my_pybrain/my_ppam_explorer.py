from pybrain.rl.explorers.explorer import Explorer
from copy import deepcopy
from scipy import where
from random import choice, random
from math import sqrt, log
import numpy as np


class MyPPAMGreedyExplorer(Explorer):
    def __init__(self, exploration):
        Explorer.__init__(self, 2, 1)
        self.exploration = exploration

    def _setModule(self, module):
        """ Tell the explorer the module. """
        self._module = module

    def activate(self, state, action):
        """ Save the current state for state-dependent exploration. """
        self.state = state
        return Explorer.activate(self, state, action)

    def _forwardImplementation(self, inbuf, outbuf):
        """ Draws a random number between 0 and 1. If the number is less
            than epsilon, a random action is chosen. If it is equal or
            larger than epsilon, the greedy action is returned.
            The PPAM also masks an action if it cannot be performed, so that exploration is safe.
        """
        assert self.module

        values = self.module.getActionValues(self.state)

        state = self.state[0]

        '''
        Get the amount of litres currently in the water tank and the state of the switch. The switch encodes which 
        subformulas of the PPAM are true (Y is the "yesterday" modality of PPLTLf): 
        switch = 0 --> Y Y open, Y close
        switch = 1 --> Y Y close, Y close
        switch = 2 --> Y Y Y close, Y Y close, Y close (now the valve can be opened again)
        switch = 4 --> Y Y Y open, Y Y open, Y open (now the valve can be closed again)
        switch = 5 --> Y Y open, Y close
        switch = 6 --> Y Y close, Y open
        State 3 is never reached
        '''
        litres, switch = (int(state) // 7) + 1, int(state) % 7

        '''
        If there is too much water in the water tank or not enough timesteps have elapsed since the valve was closed,
        disable the open action
        '''
        if (litres > 93 and switch in [2, 4]) or switch in [0, 1]:
            values[1] = np.iinfo(np.int64).min

        '''
        If there is not enough water in the water tank or not enough timesteps have elapsed since the valve was opened,
        disable the close action
        '''
        if (litres < 4 and switch in [2, 4]) or switch in [5, 6]:
            values[0] = np.iinfo(np.int64).min

        actions = []
        if random() <= self.exploration:
            new_action = where(values != np.iinfo(np.int64).min)[0]
            new_action = choice(new_action)
            # new_action = choice(range(len(values)))
            np.delete(values, new_action)
            actions.append(new_action)
        else:
            new_action = where(values == max(values))[0]
            new_action = choice(new_action)
            np.delete(values, new_action)
            actions.append(new_action)

        while len(actions) < self.outdim:
            actions.append(-1)

        outbuf[:] = actions
