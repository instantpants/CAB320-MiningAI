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

        if underground.ndim == 2:
            # 2D Mine Setup
            self.len_x = np.size(underground, axis=0)
            self.len_y = 0 # we're 2D mine atm so we won't have a y axis
            self.len_z = np.size(underground, axis=1)

            self.cumsum_mine = np.cumsum(underground, axis=1) # use Z axis

            self.initial = np.zeros((self.len_x,), dtype=int)
        else:
            # 3D Mine Setup
            self.len_x = np.size(underground, axis=0)
            self.len_y = np.size(underground, axis=1)
            self.len_z = np.size(underground, axis=2)

            self.cumsum_mine = np.cumsum(underground, axis=2) # use Z axis 

            self.initial = np.zeros((self.len_x, self.len_y,), dtype=int)

        self.initial = convert_to_tuple(self.initial)

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
     
    @functools.lru_cache(maxsize = 128)
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
        actions = [] # List to contain valid actions for supplied state

        # Flatten the state to be used in a single loop this allows us to work 
        # with both 1D and 2D states in a single loop.
        for i, v in enumerate(state.flat):
            # The value of the cell if it were dug (for neighbour comparisons)
            next_val = v + 1

            # Can't dig a cell if we're already at the bottom
            if next_val > self.len_z:
                continue
            
            # Get column index as tuple for usage in neighbours
            if state.ndim == 1:
                idx = (i,) # (x,)
            else:                
                # Convert 1D flat index to 2D index
                idx = (i // state.shape[1], i % state.shape[1],) # (x, y,)
            
            # Convert neighbour indexes to a zipped tuple for indexing.
            neighbour_indices = tuple(zip(*self.surface_neigbhours(idx)))
            neighbour_vals = state[neighbour_indices] # Neighbour values

            # If all neighbours are within tolerance, lets add the index to actions
            if np.all(abs(next_val - neighbour_vals) <= self.dig_tolerance):
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
        That is, the sum of the values of all the dug cells.
        
        No loops needed in the implementation!        
        '''
        state = np.array(state)
        #print(f"UG {self.underground.shape}:\n {self.underground}\n")
        #print(f"CS {self.cumsum_mine.shape}:\n {self.cumsum_mine}\n")
        #print(f"State {state.shape}:\n {state}")
        
        # Get the indexes of all non-zero columns in the state, these are 
        # columns that have been dug.
        c = np.nonzero(state)

        # Make a copy of the non-zero values and take 1 from each of them 
        # as arrays are 0 terminated. (This is to access the correct depth)
        depth = state[c] - 1

        # Calculate the payoff by summing the cumulative sum at each index
        # defined above.
        if state.ndim == 2:
            print(self.cumsum_mine[c[0], c[1], depth])
            return sum(self.cumsum_mine[c[0], c[1], depth]) # cumsum_mine[x, y, z]
        else:
            return sum(self.cumsum_mine[c[0], depth])       # cumsum_mine[x, z]

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

@functools.lru_cache(maxsize = 256)
def get_best_child(mine, node):
    '''
    Returns the child node which has the largest payoff.

    Parameters
    ----------
    mine : a Mine instance

    node : a Node whose children to check

    Returns
    -------
    best_child_node
        
    '''
    # Get all children states
    children = node.expand(mine)

    # Generate a list of the childrens payoff values
    children_payoffs = [mine.payoff(child.state) for child in children]

    # Find the index of the child whose payoff is the largest
    best_child_index = np.argmax(children_payoffs)

    # Use that above index to get the best child node
    best_child_node = children[best_child_index]

    return best_child_node
    
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
    #### THOMAS PUT THIS HERE TO HELP YOU GET FAMILIAR WITH THE NODE STUFF ####

    # Initialize Start node to mines initial state
    startNode = search.Node(mine.initial)
    print("Start: ", startNode) # Print startNode state

    # Find the child node with the best payoff
    bestChildNode = get_best_child(mine, startNode)
    print(f"Best Child Node: {bestChildNode}, Payoff = {mine.payoff(bestChildNode.state)}")

    # Loop through each child and print their information to verify the above stuff
    for i, childNode in enumerate(startNode.expand(mine)):   
        childPayoff = mine.payoff(childNode.state)
        print(f"Child{i}: {childNode}, Payoff = {childPayoff}")

        # # Below is an outline of what a Node object can do, though you aren't
        # # obviously expected to use all of them to get things to work.

        # # THESE ARE SOME VARIABLES YOU CAN ACCESS FROM A NODE OBJECT
        # s = childNode.state        # State of the child
        # p = childNode.parent       # Parent node of the child
        # a = childNode.action       # Action taken to get to this node
        # c = childNode.path_cost    # Total cost of path up to this node
        # d = childNode.depth        # Depth in the tree of this node

        # # THESE ARE SOME FUNCTIONS YOU CAN USE IF YOU NEED TO
        # S = childNode.solution()   # Return the sequence of actions to go from the root state to this node state.
        # P = childNode.path()       # Return a list of nodes forming the path from the root to this node.
        # C = childNode.expand(mine) # List the nodes reachable in one step from this node.
        
        # You can print uncomment this if you want to know more about the childs state
        # DEBUG_PRINTING(mine, childNode.state)

    #### END HELP FROM THOMAS ####

    # if payoff is negative, don't dig
    # compare the dig tolerance from the surface to the next one, if it wasn't dug, and see if you can dig it
    # compare to the next columns
    # check the total payoff
    # ... will rethink this afternoon.
    

    #cumulative_sum = mine.cumsum_mine # Just to show a cumulative sum
    #transposed = mine.underground.T
    # print(f"Underground Size: X{mine.len_x}, Y{mine.len_y}, Z{mine.len_z}")
    # print("Underground:\n", mine.underground.T)
    # print("Cumulative Sum:\n", mine.cumsum_mine.T)
    # print("Initial State:\n", mine.initial.T)

    # get all possible actions
    valid_actions = mine.actions(mine.initial)
    #print("valid actions: ",valid_actions)
    #result = Mine.result(mine, mine.initial, valid_actions)
    #print("result: ",result)

    #Mine.DEBUG_PRINTING(mine, mine.initial)

    # get the cumulative sum per column within the bounds given by actions


    # determine how far down to dig per column that doesn't become dangerous

    # determine the actions required to get this tuple
    # find_action_sequence(initial state all zeroes, )

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

    # Branch and Bound Steps
    # Assign an upper bound b (initial bound = 0)
    # to calculate bound, ignore slope constraint.
    # work through each column to find the ideal cumsum per column (ideal dig depth)
    # for each following state, bound = best_payoff
    # need to compare and reassign (if necessary) best_payoff for each state
    # frontier = possible next states from current state s0.
    # if b(s1) <= best_payoff
    # then remove s1 from frontier and move to s2

    # Possible PseudoCode???
    
    # Data: Input Cost matrix M [][]
    # Result: Optimal mine state?
    # Function MinCost(M[][])
    # while true do
    #     E = LeastCost();
    #     if E is a leaf node then
    #         print();
    #         return;
    #     end
    #     for each child S of E do
    #         Add(S);
    #         S --> parent = E;
    #     end
    # end

    # MinCost() = list of active nodes
    # LeastCost() = minimum cost of active node at each level of tree. After finding node with min cost, remove node from list of active and return it
    # Add() = calculates cost of particular node and adds it to list of active nodes




    # TODO: psuedocode of this section before implementation

    # Initialize Start node to mines initial state
    startNode = search.Node(mine.initial)
    print("Start: ", startNode) # Print startNode state

    #Initialise best_payoff
    best_payoff = mine.payoff(mine.initial)
    print ("BB Best payoff for start node =", best_payoff)




    best_action_list = None

    

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

# def TestDPusingNode(mine):
#     explored = set() # set of states

#     @functools.lru_cache(maxsize = 256)
#     def search_rec(node):
#         '''
#         Recursive function that will discover all possible states
#         and add them to a set for later
#         '''
#         explored.add(node)
#         for child in node.expand(mine):
#             if child not in explored:
#                 search_rec(child)

#     # Initialize start node and recursion
#     startNode = search.Node(mine.initial)
#     search_rec(startNode)

#     # Convert our set into a list for indexing
#     explored_list = list(explored)

#     # Look for best possible state within the list
#     # criteria being the one whose payoff is the largest
#     i = np.argmax([mine.payoff(x) for x in explored_list])
#     best_node = explored_list[i]

#     # Finish up by getting the return values
#     best_final_state = best_node.state
#     best_action_list = best_node.solution()
#     best_payoff = mine.payoff(best_final_state)

#     return best_payoff, best_action_list, best_final_state    


def TestDP(mine):
    explored = set()

    @functools.lru_cache(maxsize = 256)
    def search_rec(state):
        '''
        Recursive function that will discover all possible states
        and add them to a set for later
        '''
        # This state has now been explored, so lets remember it
        explored.add(state)

        # Set up variables for this state
        action_list = []
        payoff = mine.payoff(state)
        best_state = state
        
        # Iterate children
        for action in mine.actions(state):
            next_state = mine.result(state, action)
            
            # If we've already checked that state, let's not do it again
            if next_state in explored:
                continue

            # Get the child state values
            next_payoff, next_state = search_rec(next_state)

            # If our next payoff is better, return that value
            if next_payoff > payoff:
                best_state = next_state
                payoff = next_payoff
        
        # If none of the child actions were good, let's return what we have now
        return payoff, best_state
    
    best_payoff, best_final_state = search_rec(mine.initial)
    best_action_list = find_action_sequence(mine.initial, best_final_state)

    return best_payoff, best_action_list, best_final_state

# REMOVE THIS BEFORE SUBMITTING
def DEBUG_PRINTING(mine, state):
    '''
    Prints some mine debug info:

    - Underground shape and contents
    - State shape and contents
    - Payoff for supplied state
    - Actions for supplied state
    - Whether the state is dangeous

    Parameters
    ----------
    mine  : A mine object.

    state : A state to be evaluated.
    '''

    best_payoff = np.argmax(mine.cumsum_mine, axis=1 if state.ndim == 1 else 0)

    print('-------------- DEBUG INFO -------------- ')
    print(f"Underground {mine.underground.shape}\n {mine.underground}")
    print(f"Cumulative Sum {mine.cumsum_mine.shape}\n {mine.cumsum_mine}")
    print("Best paying by column:\n", best_payoff)
    print(f"State {state.shape}:\n {state}")
    print("Payoff:\n", mine.payoff(convert_to_tuple(state)))
    print("Possible Actions:\n", mine.actions(convert_to_tuple(state)))
    print("Action Sequence from empty state:\n", find_action_sequence(np.zeros(state.shape), state))
    print("Is Dangerous?:\n", mine.is_dangerous(state))

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
        [# X0     Z0      Z1      Z2      Z3      Z4
            [ 0.455,  0.579,  -0.54, -0.995, -0.771], # Y0
            [ 0.049,  1.311, -0.061,  0.185, -1.959], # Y1
            [ 2.38 , -1.404,  1.518, -0.856,  0.658], # Y2
            [ 0.515, -0.236, -0.466, -1.241, -0.354]  # Y3
        ],
        [# X1     Z0      Z1      Z2      Z3      Z4
            [ 0.801,  0.072, -2.183,  0.858, -1.504], # Y0
            [-0.09 , -1.191, -1.083,  0.78 , -0.763], # Y1
            [-1.815, -0.839,  0.457, -1.029,  0.915], # Y2
            [ 0.708, -0.227,  0.874,  1.563, -2.284]  # Y3
        ],
        [# X2     Z0      Z1      Z2      Z3      Z4
            [-0.857,  0.309, -1.623,  0.364,  0.097], # Y0
            [-0.876,  1.188, -0.16 ,  0.888, -0.546], # Y1
            [-1.936, -3.055, -0.535, -1.561, -1.992], # Y2
            [ 0.316,  0.97 ,  1.097,  0.234, -0.296]  # Y3
        ]
    ])
    some_3D_state = np.array([
        [ 3, 2, 1, 0],
        [ 2, 2, 1, 0],
        [ 1, 1, 1, 0],  
    ])

    # 3D
    #underground = some_3D_underground
    #state = some_3D_state

    #2D
    underground = some_2D_underground
    state = some_2D_state
    
    # ## INSTANTIATE MINE ##
    # underground = np.random.rand(5, 3) # 3 columns, 5 rows
    m = Mine(underground, dig_tolerance=1)

    # DEBUG_PRINTING(m, state)

    # ## BEGIN SEARCHES ##

    # # Dynamic Programming search
    # t0 = time.time()
    # best_payoff, best_action_list, best_final_state = TestDP(m)
    # t1 = time.time()

    # print ("Test DP solution -> ", best_final_state)
    # print ("Test DP payoff -> ", best_payoff)
    # print ("Test DP action -> ", best_action_list)
    # print ("Test DP Solver took ",t1-t0, ' seconds')

    t0 = time.time()
    best_payoff, best_action_list, best_final_state = search_dp_dig_plan(m)
    t1 = time.time()

    print ("DP solution -> ", best_final_state)
    print ("DP Solver took ",t1-t0, ' seconds')
    
    # # Best Branch search
    # # t0 = time.time()
    # best_payoff, best_action_list, best_final_state = search_bb_dig_plan(m)
    # t1 = time.time()

    # print ("BB solution -> ", best_final_state)
    # print ("BB Solver took ",t1-t0, ' seconds')

    #search_bb_dig_plan(m)

