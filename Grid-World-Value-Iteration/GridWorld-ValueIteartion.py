# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 11:53:34 2021

@author: prakh
"""

import numpy as np

# global variables
BOARD_ROWS = 3
BOARD_COLS = 4
WIN_STATE = (0, 3)
START = (2, 0)

class State:
    def __init__(self, state=START):        
        self.state = state
        self.isEnd = False        

    def getReward(self):
        if self.state == WIN_STATE:
            return 1        
        else:
            return 0

    def isEndFunc(self):
        if (self.state == WIN_STATE):
            self.isEnd = True

    def nxtPosition(self, action):        
        if action == "up":                
            nxtState = (self.state[0] - 1, self.state[1])                
        elif action == "down":
            nxtState = (self.state[0] + 1, self.state[1])
        elif action == "left":
            nxtState = (self.state[0], self.state[1] - 1)
        else:
            nxtState = (self.state[0], self.state[1] + 1)

            
        if (nxtState[0] >= 0) and (nxtState[0] <= 2):
            if (nxtState[1] >= 0) and (nxtState[1] <= 3):                    
                    return nxtState # if next state legal
        return self.state # Any move off the grid leaves state unchanged

class Agent:

    def __init__(self):
        self.states = []
        self.actions = ["up", "down", "left", "right"]
        self.State = State()
        self.discount = 0.9
        self.theta = 0.0001
        
        # initialise state values
        self.state_values = {}        
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.state_values[(i, j)] = 0  # set initial value to 0

        self.new_state_values = {}
            
    def value_iteration(self):
        # Value iteration implementation                        
        e = 0
        # Loop through all the states rows * cols until delta < theta       
        while True:
            delta = 0
            i = 0
            j = 0
            while i < BOARD_ROWS:            
                while j < BOARD_COLS:                                                                        
                    mx_nxt_value = 0
                    self.State = State(state=(i, j))
                    for a in self.actions:
                        nxt_value = self.State.getReward() + (self.discount * self.state_values[self.State.nxtPosition(a)])
                        if nxt_value >= mx_nxt_value:                    
                            mx_nxt_value = nxt_value
                    
                    # Update the state values
                    self.new_state_values[(i,j)] = mx_nxt_value
                    print("Delta", delta)
                    delta = max(delta, abs(self.state_values[self.State.state] - self.new_state_values[(i,j)]))
                    j += 1                    
                i += 1
                j = 0                                        
            self.state_values = self.new_state_values.copy()
            if(delta < self.theta):
                break
            e += 1
                
    def showValues(self):
        for i in range(0, BOARD_ROWS):
            print('----------------------------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                out += str(self.state_values[(i, j)]).ljust(6) + ' | '
            print(out)
        print('----------------------------------')


if __name__ == "__main__":
    ag = Agent()
    ag.value_iteration()
    print(ag.showValues())
