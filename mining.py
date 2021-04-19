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
        
        underground = np.array([
        [-0.814,  0.637, 1.824, -0.563],
        [ 0.559, -0.234, -0.366,  0.07 ],
        [ 0.175, -0.284,  0.026, -0.316],
        [ 0.212,  0.088,  0.304,  0.604],
        [-1.231, 1.558, -0.467, -0.371]])

        # underground = np.array([[[ 0.455,  0.579, -0.54 , -0.995, -0.771],
        #                         [ 0.049,  1.311, -0.061,  0.185, -1.959],
        #                         [ 2.38 , -1.404,  1.518, -0.856,  0.658],
        #                         [ 0.515, -0.236, -0.466, -1.241, -0.354]],
        #                         [[ 0.801,  0.072, -2.183,  0.858, -1.504],
        #                         [-0.09 , -1.191, -1.083,  0.78 , -0.763],
        #                         [-1.815, -0.839,  0.457, -1.029,  0.915],
        #                         [ 0.708, -0.227,  0.874,  1.563, -2.284]],
        #                         [[ -0.857,  0.309, -1.623,  0.364,  0.097],
        #                         [-0.876,  1.188, -0.16 ,  0.888, -0.546],
        #                         [-1.936, -3.055, -0.535, -1.561, -1.992],
        #                         [ 0.316,  0.97 ,  1.097,  0.234, -0.296]]])

        self.underground = underground 
        # self.underground  should be considered as a 'read-only' variable!
        self.dig_tolerance = dig_tolerance
        assert underground.ndim in (2,3)

        if underground.ndim == 2:
            # 2D Mine Setup
            self.len_x = np.size(underground, axis=0)
            self.len_y = 0 # we're 2D mine atm so we won't have a y axis
            self.len_z = np.size(underground, axis=1)

            self.cumsum_mine = np.cumsum(underground, axis=1) # use Z axis

            self.initial = np.zeros(self.len_x)
        else:
            # 3D Mine Setup
            self.len_x = np.size(underground, axis=1)
            self.len_y = np.size(underground, axis=2)
            self.len_z = np.size(underground, axis=0)

            self.cumsum_mine = np.cumsum(underground, axis=0) # use Z axis 

            self.initial = np.zeros((self.len_x, self.len_y))

        # 2D Underground State
        state = np.array([0, 1, 2, 1, 0])

        # 3D Underground State
        # state = np.array([
        #     [ 3, 2, 1, 0, 0],
        #     [ 2, 1, 1, 0, 0],
        #     [ 1, 1, 0, 0, 0],
        #     [ 0, 0, 0, 0, 0]])

        print("Underground:", self.underground.shape, ":\n", self.underground)
        print("State:", state.shape, ":\n", state)
        print("Cumsum:", self.cumsum_mine.shape, ":\n", self.cumsum_mine)
        print("Payoff:", self.payoff(state))

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

        # # 2D Underground State
        # state = np.array([ 1, 1, 2, 1, 1])

        # # 3D Underground State
        # state = np.array([
        #     [ 1, 1, 0, 0, 0],
        #     [ 1, 1, 0, 0, 0],
        #     [ 0, 0, 0, 0, 0],
        #     [ 0, 0, 0, 0, 0]])

        state = np.array(state)
        actions = []

        # Flatten array to be used in a single loop 
        for i, v in enumerate(i for i in state.flat):
            # Get 1D or 2D index of cell, 2D index must be modulated from i
            idx = (i,) if state.ndim == 1 else (i % np.size(state, 0), i % np.size(state, 1),) 
            
            # If all neighbours are within tolerance, lets add the index as an action
            if np.all(abs(v+1 - [state[n] for n in self.surface_neigbhours(idx)]) <= self.dig_tolerance):
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
        c = np.nonzero(state) # 2D X Axis or 3D XY Axis
        z = state[c] - 1      # Z Axis

        # Now get the payoff for each column, depending on the dimension
        if state.ndim == 2:
            return sum(self.cumsum_mine[z, c[0], c[1]])
        else:
            return sum(self.cumsum_mine[c[0], z])


    def is_dangerous(self, state):
        '''
        Return True if the given state breaches the dig_tolerance constraints.
        
        No loops needed in the implementation!
        '''
        # convert to np.array in order to use numpy operators
        state = np.array(state)
        ax = 1 if state.ndim==2 else 0
        col_sum = np.sum(state, axis=ax)

        for x, y in enumerate(col_sum):
            print(state[x,y])
        
        #return abs(state[a]-state[b]) < self.dig_tolerance
           


    
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

    #return best_payoff, best_action_list, best_final_state 
    # ^^^^ this line is correct (and correct order) but obviously variables aren't yet
    raise NotImplementedError


    
    
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
    
    raise NotImplementedError



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
    # approach: among all columns for which s0 < s1, pick the column loc
    # with the smallest s0[loc]
    raise NotImplementedError
        
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

    # # Dynamic Programming search
    # t0 = time.time()
    # best_payoff, best_action_list, best_final_state = search_dp_dig_plan(m)
    # t1 = time.time()

    # print ("DP solution -> ", best_final_state)
    # print ("DP Solver took ",t1-t0, ' seconds')
    
    # # Best Branch search
    # # t0 = time.time()
    # best_payoff, best_action_list, best_final_state = search_bb_dig_plan(m)
    # t1 = time.time()

    # print ("BB solution -> ", best_final_state)
    # print ("BB Solver took ",t1-t0, ' seconds')
