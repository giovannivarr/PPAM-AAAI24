from pybrain.rl.learners.valuebased.interface import ActionValueInterface
from pybrain.structure.modules import Table, Module
from pybrain.structure.parametercontainer import ParameterContainer
from scipy import where
from random import choice
import numpy as np

class MyPPAMActionValueTable(Table, ActionValueInterface):
    def __init__(self, numStates, numActions, name=None):
        Module.__init__(self, 1, numActions, name)
        ParameterContainer.__init__(self, numStates * numActions)
        self.numRows = numStates
        self.numColumns = numActions

    @property
    def numActions(self):
        return self.numColumns

    def _forwardImplementation(self, inbuf, outbuf):
        """ Take a vector of length 1 (the state coordinate) and return
            the action with the maximum value over all actions for this state.
            The PPAM also masks an action if it cannot be performed, so that no unsafe action is taken.
        """
        outbuf[:] = self.getMaxAction(inbuf[0])

    def getMaxAction(self, state):
        """ Return the action with the maximal value for the given state. """
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

        values = self.params.reshape(self.numRows, self.numColumns)[int(state), :].flatten()

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
        for i in range(self.outdim):
            action = where(values == max(values))[0]
            action = choice(action)
            np.delete(values, action)
            actions.append(action)
        return actions

    def getActionValues(self, state):
        if isinstance(state,list):
            return self.params.reshape(self.numRows, self.numColumns)[list(map(int,state)), :].flatten()
        else:
            return self.params.reshape(self.numRows, self.numColumns)[state, :].flatten()


    def initialize(self, value=0.0):
        """ Initialize the whole table with the given value. """
        self._params[:] = value