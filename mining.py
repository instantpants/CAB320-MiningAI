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
from heapq import *

import search

def my_team():
    '''
    Return list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
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

        self.len_x = np.size(underground, axis=0)
        self.len_z = np.size(underground, axis=-1)

        # Setup dimension specific values
        if underground.ndim == 2:
            # 2D Mine Setup
            self.len_y = 0 # we're 2D mine atm so we won't have a y axis
            self.initial = np.zeros((self.len_x,), dtype=int)
        else:
            # 3D Mine Setup
            self.len_y = np.size(underground, axis=1)
            self.initial = np.zeros((self.len_x, self.len_y,), dtype=int)

        # Convert it to a tuple
        self.initial = convert_to_tuple(self.initial)
        self.cumsum_mine = np.cumsum(underground, axis=-1) # use Z axis

    @functools.lru_cache(maxsize=None)
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
     
    @functools.lru_cache(maxsize=None)
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

        e.g.
        For a state of (0, 1, 2, 1, 0) and a tolerance of 1. 
        Return value would be ((0,), (4,))
        '''        
        state = np.array(state) # Copy to np.array for indexing and other methods
        actions = set() # List to contain valid actions for supplied state

        # Flatten state so we can iterate both 2D and 3D states
        for i, z in enumerate(state.flat):
            next_z = z + 1 # Next dug value at this column
            if next_z > self.len_z: # If next_z is deeper than allowed, skip it
                continue

            # Get dimensional coordinate
            if state.ndim == 2:
                coord = (i // self.len_y, i % self.len_y)
            else:
                coord = (i,)

            # Get neighbouring cells and values
            neighbours = self.surface_neigbhours(coord)
            neighbour_vals = [state[n] for n in neighbours]

            # If all neighbour values adhere to slope constraint, add the coord to actions
            if np.all(abs(next_z - neighbour_vals) <= self.dig_tolerance):
                actions.add(coord)

        return tuple(actions)
  
    @functools.lru_cache(maxsize=None)
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

    @functools.lru_cache(maxsize=None)
    def payoff(self, state):
        '''
        Compute and return the payoff for the given state.
        That is, the sum of the values of all the dug cells.
        
        No loops needed in the implementation!        
        '''
        state = np.array(state)        
        if np.any(state > 0):
            columns = np.nonzero(state) # Get the indexes of all dug columns (X or XY)
            depth = state[columns] - 1  # Get depth vals - 1 as arrays are 0 terminated
            coords = columns + (depth,) # Add the depth dimension so indexing is easy

            # Sum the cumsum_mine values and return
            return sum(self.cumsum_mine[coords])
        else:
            return 0

    @functools.lru_cache(maxsize=None)
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
    @functools.lru_cache(maxsize=None) # This line will 'memoize' every call to function with some state
    def search_rec(node):
        best_payoff = mine.payoff(node.state)
        best_node = node
        
        ''' Recursive (DFS) search function that will dynamically locate the best payoff possible before returning. '''
        for child in node.expand(mine):
            # Perform DFS on this branch
            check_payoff, check_node = search_rec(child)

            # Update return values if branch was better
            if check_payoff > best_payoff:
                best_payoff = check_payoff
                best_node = check_node
        
        # Return best this branch has to offer
        return best_payoff, best_node

    # Initial recursive function call
    root = search.Node(mine.initial)
    best_payoff, best_node = search_rec(root)
    best_final_state = best_node.state
    best_action_list = find_action_sequence(mine.initial, best_final_state)

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
    root = search.Node(mine.initial)
    frontier = [root] 
    best_payoff = float('-inf')
    best_node = root

    @functools.lru_cache(maxsize=None)
    def b(state):
        """ Returns best possible sum from some state when ignoring slope constraint """
        state = np.array(state) - 1
        state[state < 0] = 0
        payoff = np.zeros(state.shape)

        # Enumerate flat array to get coordinates suitable for both 2D and 3D arrays
        for i, z in enumerate(state.flat):
            if state.ndim == 2:
                c = (i // mine.len_y, i % mine.len_y,)
            else:
                c = (i,)
            
            # Find best possible value after current z value in column c
            payoff[c] = np.amax(mine.cumsum_mine[c][z:])

        # Return sum of best possible values
        return np.sum(payoff)

    @functools.lru_cache(maxsize=None)
    def expand(node):
        """ Returns a list of children of node, where b(s) > best_payoff """
        children = []
        # Flatten state so we can iterate both 2D and 3D states
        for child in node.expand(mine):
            if child not in frontier and b(child.state) >= best_payoff:
                children.append(child)
        return children

    # Main loop
    while len(frontier) > 0:
        node = frontier.pop(0) # FIFO queue
        payoff = mine.payoff(node.state)

        if payoff >= best_payoff:
            # New best node has been found!
            best_payoff = payoff
            best_node = node

            # Prune all other trees and re-expand from best node
            frontier = [] # We do this first or else we might run into 'in frontier' conflicts
            frontier = expand(node) 
        else:
            # We haven't found the best yet, so keep branching
            frontier.extend(expand(node)) 

    best_final_state = best_node.state
    best_action_list = find_action_sequence(mine.initial, best_final_state)

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
        # Get all possible actions at this state where diff is != 0
        diff = s1 - s0
        actions = np.where(diff != 0) 

        # Add 1 (dig cost) to each position in s0 so we can re-compute 
        # diff next iteration
        s0[actions] += 1 

        # Loop through actions and add them to the sequence array as a 
        # properly formatted tuple.
        if s0.ndim == 2:
            for a in zip(*actions):
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
    some_2D_underground = np.array([
        [-0.814,  0.637, 1.824, -0.563],
        [ 0.559, -0.234,-0.366,  0.074],
        [ 0.175, -0.284,  0.026,-0.316],
        [ 0.212,  0.088,  0.304, 0.604],
        [-1.231,  1.558, -0.467,-0.371]
    ])
    some_2D_state = np.array([
        0, 1, 2, 1, 0
    ])
    
    some_3D_underground = np.array([
        [
            [ 0.455,  0.579,  -0.54, -0.995, -0.771],
            [ 0.049,  1.311, -0.061,  0.185, -1.959],
            [ 2.38 , -1.404,  1.518, -0.856,  0.658],
            [ 0.515, -0.236, -0.466, -1.241, -0.354]
        ],
        [
            [ 0.801,  0.072, -2.183,  0.858, -1.504],
            [-0.09 , -1.191, -1.083,  0.78 , -0.763],
            [-1.815, -0.839,  0.457, -1.029,  0.915],
            [ 0.708, -0.227,  0.874,  1.563, -2.284]
        ],
        [
            [-0.857,  0.309, -1.623,  0.364,  0.097],
            [-0.876,  1.188, -0.16 ,  0.888, -0.546],
            [-1.936, -3.055, -0.535, -1.561, -1.992],
            [ 0.316,  0.97 ,  1.097,  0.234, -0.296]
        ]
    ])
    some_3D_state = np.array([
        [ 3, 2, 1, 0],
        [ 2, 2, 1, 0],
        [ 1, 1, 1, 0],  
    ])

    underground = some_2D_underground
    state = some_2D_state
    
    # ## INSTANTIATE MINE ##
    # underground = np.random.randn(3, 4, 5)
    # underground = np.random.randn(5, 4)
    mine = Mine(underground, dig_tolerance=1)

    print(underground.T)   

    print('-------------- DP computations -------------- ')
    tic = time.time()
    best_payoff, best_a_list, best_final_state = search_dp_dig_plan(mine)
    toc = time.time() 
    print('DP Best payoff:',best_payoff)
    print('DP Best final state:', best_final_state)  
    print('DP action list:', best_a_list)
    print('DP Computation took {} seconds\n'.format(toc-tic))   

    print('-------------- BB computations -------------- ')
    tic = time.time()
    best_payoff, best_a_list, best_final_state = search_bb_dig_plan(mine)
    toc = time.time() 
    print('BB Best payoff:',best_payoff)
    print('BB Best final state:', best_final_state)  
    print('BB action list:', best_a_list)
    print('BB Computation took {} seconds\n'.format(toc-tic))
