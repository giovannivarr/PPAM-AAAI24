import pygame, sys
import numpy as np
import atexit
import random
import time
import math
from math import fabs
import task_executor
from task_executor import *

ACTION_NAMES = ['<-', '->', '^', 'v', 'g', 'd']
# 0: left, 1: right, 2: up, 3: down, 4: get, 5: deliver

LOCATIONS = [('coke', red, 1, 1), ('beer', gold, 2, 3),
             ('chips', yellow, 3, 1), ('biscuits', brown, 0, 3),
             ('john', blue, 4, 2), ('mary', pink, 1, 4)]

TASKS = {
    'serve_drink_john': [['get_coke', 'deliver_john']],
    'serve_drink_mary': [['get_beer', 'deliver_mary']],
    'serve_snack_john': [['get_biscuits', 'deliver_john']],
    'serve_snack_mary': [['get_chips', 'deliver_mary']]
}

TASKRESTRAIN = {
    'serve_drink_john': [['get_coke', 'deliver_john']],
    'serve_drink_mary': [['get_beer', 'deliver_mary']],
    'serve_snack_john': [['get_biscuits', 'deliver_john']],
    'serve_snack_mary': [['get_chips', 'deliver_mary']]
}

REWARD_STATES = {
    'Init': 0,
    'Alive': 0,
    'Dead': 0,
    'Score': 1000,
    'Hit': 0,
    'Forward': 0,
    'Turn': 0,
    'BadGet': 0,
    'BadDeliver': 0,
    'TaskProgress': 100,
    'TaskComplete': 1000
}

DELIVERABLES = ['coke', 'beer', 'chips', 'biscuits']

PEOPLE = ['john', 'mary']


class CocktailParty(TaskExecutor):

    def __init__(self, rows=5, cols=5, trainsessionname='test'):
        global ACTION_NAMES, LOCATIONS, TASKS, REWARD_STATES, DELIVERABLES, PEOPLE
        TaskExecutor.__init__(self, rows, cols, trainsessionname)
        self.locations = LOCATIONS
        self.action_names = ACTION_NAMES
        self.tasks = TASKS
        self.reward_states = REWARD_STATES
        self.maxitemsheld = 1
        self.map_actionfns = {4: self.doget, 5: self.dodeliver}
        self.rb_states = True
        self.restraintask = False
        self.fluents = {'served_john_food': False, 'served_john_drink': False,
                        'served_mary_food': False, 'served_mary_drink': False}
        self.beer_violations = 0    # counter for serving beers to minors
        self.serving_violations = 0 # counter for serving a guest multiple times

        if (cols > 5):
            self.move('john', cols - 1, 2)
            self.move('mary', 1, rows - 1)
            self.move('coke', cols / 2, rows / 2)
            self.move('chips', 3, rows / 2)

    def move(self, what, xnew, ynew):
        f = None
        n = None
        for t in self.locations:
            if (t[0] == what):
                f = t
                n = (t[0], t[1], xnew, ynew)
        self.locations.remove(f)
        self.locations.append(n)

    def setRestrainTask(self):
        self.restraintask = True
        self.tasks = TASKRESTRAIN

    def setPPAMStates(self):
        self.rb_states = False

    def reset(self):
        TaskExecutor.reset(self)
        self.fluents = {'served_john_food': False, 'served_john_drink': False,
                        'served_mary_food': False, 'served_mary_drink': False}

    def doget(self):
        """
        Called when the agent performs the "get" action, checks that the agent can really grab something and performs
        the appropriate transition
        """
        what = self.itemat(self.pos_x, self.pos_y)
        if what != None and not self.isAuto:
            print("get: ", what)
        if (what == None):
            r = self.reward_states['BadGet']
        elif (len(self.has) == self.maxitemsheld):
            r = self.reward_states['BadGet']
        else:
            self.has.append(what)
            r = self.check_action_task('get', what)
        return r

    def dodeliver(self):
        """
        Called when the agent performs the "deliver" action, checks that the agent can really deliver something and
        performs the appropriate transition
        """
        what = self.itemat(self.pos_x, self.pos_y)
        if what != None and not self.isAuto:
            print("deliver %r to %s " % (self.has, what))
        if (what == None):
            r = self.reward_states['BadDeliver']
        elif (len(self.has) == 0):
            r = self.reward_states['BadDeliver']
        else:
            if 'beer' in self.has and what == 'john':
                self.beer_violations += 1
            if 'served_%s_food' % (what) in self.fluents.keys():
                if (('chips' in self.has or 'biscuits' in self.has) and self.fluents['served_%s_food' % (what)]) or \
                        (('beer' in self.has or 'coke' in self.has) and self.fluents['served_%s_drink' % (what)]):
                    self.serving_violations += 1

                if ('chips' in self.has and what == 'mary') or ('biscuits' in self.has and what == 'john'):
                    self.fluents['served_%s_food' % (what)] = True
                elif ('beer' in self.has and what == 'mary') or ('coke' in self.has and what == 'john'):
                    self.fluents['served_%s_drink' % (what)] = True

            self.has = []
            r = self.check_action_task('deliver', what)
        return r

    def get_normative_actions(self):
        """
        Computes the set of actions that comply with the normative specifications given the current state and truth
        values of the fluents.

        :return: a tuple of integers, representing the indexes of actions that the agent can perform given their norms.
        """
        # initialize actions list
        valid_a = list(range(5))

        what = self.itemat(self.pos_x, self.pos_y)

        if what == 'john':
            if ((not self.fluents['served_john_drink'] and 'coke' in self.has) or
                (not self.fluents['served_john_food'] and ('biscuits' in self.has or 'chips' in self.has))):
                valid_a.append(5)

        elif what == 'mary':
            if (not self.fluents['served_mary_drink'] and ('coke' in self.has or 'beer' in self.has)) or \
                    (not self.fluents['served_mary_food'] and ('biscuits' in self.has or 'chips' in self.has)):
                valid_a.append(5)

        return valid_a

    def get_executable_actions(self, valid_a):
        """
        Computes, from the set of normative actions, the set of actions that can be executed
        in the current environment state.

        :param valid_a: the tuple of integers representing the indexes of actions that can be executed
        according to their norms.
        :return: a tuple of integers, representing the indexes of normative actions that the agent can perform
        given the environment state.
        """
        set_a = valid_a
        what = self.itemat(self.pos_x, self.pos_y)
        if 4 in valid_a:
            # check that the agent can get an item and that it doesn't hold the maximum number of items it is allowed to
            if what not in DELIVERABLES or len(self.has) >= self.maxitemsheld:
                set_a.remove(4)
        if 5 in valid_a:
            # check that the agent can serve someone
            if what not in PEOPLE:
                set_a.remove(5)

        return tuple(set_a)

    def get_valid_actions(self):
        """
        Gets the set of actions that are valid, according to the current MDP state and the PPAM state (i.e., the set of
        fluents that are true).
        """
        if not self.restraintask:
            return tuple(range(6))
        else:
            valid_a = self.get_normative_actions()
            set_a = self.get_executable_actions(valid_a)

            return set_a

    def evaluate_agent_fluents(self):
        """
        Evaluate, for the current timestep, the set of fluents that are true. These fluents are the set of subformulas
        used to determine which actions the agent can perform.

        :return: a tuple containing the truth values (1 for True, 0 for False) of the fluents.
        """
        fluents = {'served_john_food': self.fluents['served_john_food'],
                   'served_john_drink': self.fluents['served_john_drink'],
                   'served_mary_food': self.fluents['served_mary_food'],
                   'served_mary_drink': self.fluents['served_mary_drink'],
                   'has_biscuits': False, 'has_chips': False,
                   'has_beer': False, 'has_coke': False,
                   'at_john': False, 'at_mary': False}

        what = self.itemat(self.pos_x, self.pos_y)

        if what in PEOPLE:
            fluents['at_%s' % what] = True

        if self.has:
            fluents['has_%s' % self.has[0]] = True

        return tuple(1 if v else 0 for v in fluents.values())
