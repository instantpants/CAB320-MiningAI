#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    GROUP MEMBERS

    Thomas Fabian   - n10582835
    Celine Lindeque - n10481478
    Daniel Edwards  - n5538815
"""
"""
Created on Wed Feb  3 17:56:47 2021

@author: frederic

    
class problem with     

An open-pit mine is a grid represented with a 2D or 3D numpy array. 

The first coordinates are surface locations.

In the 2D case, the coordinates are (x,z).
In the 3D case, the coordinates are (x,y,z).
The last coordinate 'z' points down.

    
A state indicates for each surface location  how many cells 
have been dug in this pit column.

For a 3D mine, a surface location is represented with a tuple (x,y).

For a 2D mine, a surface location is represented with a tuple (x,).


Two surface cells are neighbours if they share a common border point.
That is, for a 3D mine, a surface cell has 8 surface neighbours.


An action is represented by the surface location where the dig takes place.


"""
import numpy as np
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import itertools

import functools # @lru_cache(maxsize=32)

from numbers import Number

import search

def my_team():
    '''
    Return list of the team members of this assignment submission as a list
    of triplet of t he form (student_number, first_name, last_name)
    '''
    return [
        (10582835, 'Thomas', 'Fabian'),
        (10481478, 'Celine', 'Lindeque'),
        (5538815, 'Daniel', 'Edwards')
    ]
    
def convert_to_tuple(a):
    '''
    Convert the parameter 'a' into a nested tuple of the same shape as 'a'.
    
    The parameter 'a' must be array-like. That is, its elements are indexed.

    Parameters
    ----------
    a : flat array or an array of arrays

    Returns
    -------
    the conversion of 'a' into a tuple or a tuple of tuples

    '''
    if isinstance(a, Number):
        return a
    if len(a)==0:
        return ()
    # 'a' is non empty tuple
    if isinstance(a[0], Number):
        # 'a' is a flat list
        return tuple(a)
    else:
        # 'a' must be a nested list with 2 levels (a matrix)
        return tuple(tuple(r) for r in a)
    
    
def convert_to_list(a):
    '''
    Convert the array-like parameter 'a' into a nested list of the same 
    shape as 'a'.

    Parameters
    ----------
    a : flat array or array of arrays

    Returns
    -------
    the conversion of 'a' into a list or a list of lists

    '''
    if isinstance(a, Number):
        return a
    if len(a)==0:
        return []
    # 'a' is non empty tuple
    if isinstance(a[0], Number):
        # 'a' is a flat list
        return list(a)
    else:
        # 'a' must be a nested list with 2 levels (a matrix)
        return [list(r) for r in a]    




class Mine(search.Problem):
    '''
    
    Mine represent an open mine problem defined by a grid of cells 
    of various values. The grid is called 'underground'. It can be
    a 2D or 3D array.
    
    The z direction is pointing down, the x and y directions are surface
    directions.
    
    An instance of a Mine is characterized by 
    - self.underground : the ndarray that contains the values of the grid cells
    - self.dig_tolerance : the maximum depth difference allowed between 
                           adjacent columns 
    
    Other attributes:
        self.len_x, self.len_y, self.len_z : int : underground.shape
        self.cumsum_mine : float array : cumulative sums of the columns of the 
                                         mine
    
    A state has the same dimension as the surface of the mine.
    If the mine is 2D, the state is 1D.
    If the mine is 3D, the state is 2D.
    
    state[loc] is zero if digging has not started at location loc.
    More generally, state[loc] is the z-index of the first cell that has
    not been dug in column loc. This number is also the number of cells that
    have been dugged in the column.
    
    States must be tuple-based.
    
    '''    
    
    def __init__(self, underground, dig_tolerance = 1):
        '''
        Constructor
        
        Initialize the attributes
        self.underground, self.dig_tolerance, self.len_x, self.len_y, self.len_z,
        self.cumsum_mine, and self.initial
        
        The state self.initial is a filled with zeros.

        Parameters
        ----------
        underground : np.array
            2D or 3D. Each element of the array contains 
            the profit value of the corresponding cell.
        dig_tolerance : int
             Mine attribute (see class header comment)
        Returns
        -------
        None.
        '''
        # super().__init__() # call to parent class constructor not needed
    
        self.underground = underground # should be considered as a 'read-only' variable!
        self.dig_tolerance = dig_tolerance
        assert underground.ndim in (2,3)

        if underground.ndim == 2:
            # 2D Mine Setup
            self.len_x = np.size(underground, axis=0)
            self.len_y = 0 # we're 2D mine atm so we won't have a y axis
            self.len_z = np.size(underground, axis=1)

            self.cumsum_mine = np.cumsum(underground, axis=1) # use Z axis

            self.initial = np.zeros((self.len_x,), dtype=int)
        else:
            # 3D Mine Setup
            self.len_x = np.size(underground, axis=1)
            self.len_y = np.size(underground, axis=2)
            self.len_z = np.size(underground, axis=0)

            self.cumsum_mine = np.cumsum(underground, axis=0) # use Z axis 

            self.initial = np.zeros((self.len_x, self.len_y,), dtype=int)

    # REMOVE THIS BEFORE SUBMITTING
    def DEBUG_PRINTING(self, state):
        '''
        Prints some mine debug info:

        - Underground shape and contents
        - State shape and contents
        - Payoff for supplied state
        - Actions for supplied state
        - Whether the state is dangeous

        Parameters
        ----------
        state : A state to be debugged.
            typically just use the initial state.
        '''
        state = self.initial

        print('-------------- DEBUG INFO -------------- ')
        print("Underground", self.underground.shape, ":\n", self.underground)
        # print("Initial State", state.shape, ":\n", state)
        # print("Payoff:", self.payoff(state))
        # print("Actions:", self.actions(state))
        # print("Is Dangerous?:", self.is_dangerous(state))

    def surface_neigbhours(self, loc):
        '''
        Return the list of neighbours of loc

        Parameters
        ----------
        loc : surface coordinates of a cell.
            a singleton (x,) in case of a 2D mine
            a pair (x,y) in case of a 3D mine

        Returns
        -------
        A list of tuples representing the surface coordinates of the
        neighbouring surface cells.

        '''
        L=[]
        assert len(loc) in (1,2)
        if len(loc)==1:
            if loc[0]-1>=0:
                L.append((loc[0]-1,))
            if loc[0]+1<self.len_x:
                L.append((loc[0]+1,))
        else:
            # len(loc) == 2
            for dx,dy in ((-1,-1),(-1,0),(-1,+1),
                          (0,-1),(0,+1),
                          (+1,-1),(+1,0),(+1,+1)):
                if  (0 <= loc[0]+dx < self.len_x) and (0 <= loc[1]+dy < self.len_y):
                    L.append((loc[0]+dx, loc[1]+dy))
        return L
     
    def actions(self, state):
        '''
        Return a generator of valid actions in the give state 'state'
        An action is represented as a location. An action is deemed valid if
        it doesn't  break the dig_tolerance constraint.

        Parameters
        ----------
        state : 
            represented with nested lists, tuples or a ndarray
            state of the partially dug mine

        Returns
        -------
        a generator of valid actions

        '''        
        state = np.array(state)
        shape = state.shape
        actions = []

        # Flatten the state to be used in a single loop
        for i, val in enumerate(i for i in state.flat):
            next_val = val + 1

            # Can't dig a cell if we're already at the bottom
            if next_val > self.len_z:
                continue

            # Get 1D or 2D index of cell, 2D index must be modulated from i
            idx = (i,) if state.ndim == 1 else (i // shape[1], i % shape[1],) 
            # All neighbouring indexes as a tuple, can't index without it!
            nind = tuple(zip(*self.surface_neigbhours(idx)))
            # If all neighbours are within tolerance, lets add the index as an action
            if np.all(abs(next_val - state[nind]) <= self.dig_tolerance):
                actions.append(idx)
    
        return tuple(actions)
  
    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must a valid actions.
        That is, one of those generated by  self.actions(state)."""
        action = tuple(action)
        new_state = np.array(state) # Make a copy
        new_state[action] += 1
        return convert_to_tuple(new_state)
                
    def console_display(self):
        '''
        Display the mine on the console

        Returns
        -------
        None.

        '''
        print('Mine of depth {}'.format(self.len_z))
        if self.underground.ndim == 2:
            # 2D mine
            print('Plane x,z view')
        else:
            # 3D mine
            print('Level by level x,y slices')
        #
        print(self.__str__())
        
    def __str__(self):
        if self.underground.ndim == 2:
            # 2D mine
            return str(self.underground.T)
        else:
            # 3D mine
            # level by level representation
            return '\n'.join('level {}\n'.format(z)
                   +str(self.underground[...,z]) for z in range(self.len_z))
                    
            #return self.underground[loc[0], loc[1],:]
          
    @staticmethod   
    def plot_state(state):
        if state.ndim==1:
            fig, ax = plt.subplots()
            ax.bar(np.arange(state.shape[0]) ,
                    state
                    )
            ax.set_xlabel('x')
            ax.set_ylabel('z')
        else:
            assert state.ndim==2
            # bar3d(x, y, z, dx, dy, dz,
            # fake data
            _x = np.arange(state.shape[0])
            _y = np.arange(state.shape[1])
            _yy, _xx = np.meshgrid(_y, _x) # cols, rows
            x, y = _xx.ravel(), _yy.ravel()            
            top = state.ravel()
            bottom = np.zeros_like(top)
            width = depth = 1
            fig = plt.figure(figsize=(3,3))
            ax1 = fig.add_subplot(111,projection='3d')
            ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_zlabel('z')
            ax1.set_title('State')
        #
        plt.show()

    def payoff(self, state):
        '''
        Compute and return the payoff for the given state.
        That is, the sum of the values of all the digged cells.
        
        No loops needed in the implementation!        
        '''
        state = np.array(state)
        c = np.nonzero(state) # 2D X Axis or 3D XY Axis (columns)
        z = state[c] - 1      # Z Axis

        # Now get the payoff for each column, depending on the shape
        if state.ndim == 2:
            return sum(self.cumsum_mine[z, c[0], c[1]])
        else:
            return sum(self.cumsum_mine[c[0], z])

    def is_dangerous(self, state):
        '''
        Return True if the given state breaches the dig_tolerance constraints.
        
        No loops needed in the implementation!
        '''
        state = np.array(state)
        tol = self.dig_tolerance # Just for convenience/readability

        # Use slicing in each direction to compare, then check for any
        # values that are greater than the tolerance.
        if state.ndim == 2:
            # For a 2D state we have 8 neighbours, and as we can only 
            # slice left to right, we have to rotate the array by 
            # 90 degrees to until we have all directions
            SE = np.any(abs(state[:-1,:-1] - state[1:,1:]) > tol)
            S = np.any(abs(state[:-1,:] - state[1:,:]) > tol)
            E = np.any(abs(state[:,:-1] - state[:,1:]) > tol)
            state = np.rot90(state)
            SW = np.any(abs(state[:-1,:-1] - state[1:,1:]) > tol)
            W = np.any(abs(state[:-1,:] - state[1:,:]) > tol)
            N = np.any(abs(state[:,:-1] - state[:,1:]) > tol)
            state = np.rot90(state)
            NW = np.any(abs(state[:-1,:-1] - state[1:,1:]) > tol)
            state = np.rot90(state)
            NE = np.any(abs(state[:-1,:-1] - state[1:,1:]) > tol)
            
            return N or S or E or W or NE or SE or SW or NW
        else:
            # For a 1D state we just flip it horizontally once
            E = np.any(abs(state[:-1] - state[1:]) > tol)
            state = np.flip(state)
            W = np.any(abs(state[:-1] - state[1:]) > tol)

            return E or W
        

 # ========================  Class Mine  ==================================
    
    
    
def search_dp_dig_plan(mine):
    '''
    Search using Dynamic Programming the most profitable sequence of 
    digging actions from the initial state of the mine.
    
    Return the sequence of actions, the final state and the payoff
    

    Parameters
    ----------
    mine : a Mine instance

    Returns
    -------
    best_payoff, best_action_list, best_final_state

    '''
    # TODO: psuedocode of this section before implementation

    best_action_list = None

    best_payoff = None

    best_final_state = None

    return best_payoff, best_action_list, best_final_state


    
    
def search_bb_dig_plan(mine):
    '''
    Compute, using Branch and Bound, the most profitable sequence of 
    digging actions from the initial state of the mine.
        

    Parameters
    ----------
    mine : Mine
        An instance of a Mine problem.

    Returns
    -------
    best_payoff, best_action_list, best_final_state

    '''
    # TODO: psuedocode of this section before implementation

    best_action_list = None

    best_payoff = None

    best_final_state = None

    return best_payoff, best_action_list, best_final_state



def find_action_sequence(s0, s1):
    '''
    Compute a sequence of actions to go from state s0 to state s1.
    There may be several possible sequences.
    
    Preconditions: 
        s0 and s1 are legal states, s0<=s1 and 
    
    Parameters
    ----------
    s0 : tuple based mine state
    s1 : tuple based mine state 

    Returns
    -------
    A sequence of actions to go from state s0 to state s1

    '''    
    s0 = np.array(s0)
    s1 = np.array(s1)
    sequence = []

    while(np.any(s0 != s1)):
        cost = s1 - s0
        # We use where so that we can collect every possible 
        # action at this time, alternative methods will have 
        # us dig the same cell multiple times in a row which 
        # will breach a tolerance of 1.
        actions = np.where(cost != 0) 
        s0[actions] += 1

        # Append each action to the sequence, though as
        # np.where produces X,Y values as separate arrays
        # we have to zip the 2D ones before appending
        if s0.ndim == 2:
            for a in zip(actions[0], actions[1]):
                sequence.append(a)
        else:
            for a in actions[0]:
                sequence.append((a,))
    
    return convert_to_tuple(sequence)
        
if __name__ == '__main__':
    import time
    """
    Links:
        GoogleDocs:     https://docs.google.com/document/d/1SZjn7aqxmaZgs2Ei4RpKSgsj8sX6Tu7364aL4TOCtQs/edit?usp=sharing
        GitHub Repo:    https://github.com/CelineLind/MiningAI/
    """

    # ## INSTANTIATE MINE ##
    underground = np.random.rand(5, 3) # 3 columns, 5 rows
    m = Mine(underground, dig_tolerance=1)

    # ## BEGIN SEARCHES ##

    # Dynamic Programming search
    t0 = time.time()
    best_payoff, best_action_list, best_final_state = search_dp_dig_plan(m)
    t1 = time.time()

    print ("DP solution -> ", best_final_state)
    print ("DP Solver took ",t1-t0, ' seconds')
    
    # Best Branch search
    # t0 = time.time()
    best_payoff, best_action_list, best_final_state = search_bb_dig_plan(m)
    t1 = time.time()

    print ("BB solution -> ", best_final_state)
    print ("BB Solver took ",t1-t0, ' seconds')
