#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2021/04/02

@author: frederic

Some sanity tests 

Here is the console output

    = = = = = = = = = = = = = = = = = = = = 
    Mine of depth 4
    Plane x,z view
    [[-0.814  0.559  0.175  0.212 -1.231]
     [ 0.637 -0.234 -0.284  0.088  1.558]
     [ 1.824 -0.366  0.026  0.304 -0.467]
     [-0.563  0.07  -0.316  0.604 -0.371]]
    -------------- DP computations -------------- 
    Cache DP function before first call :  CacheInfo(hits=0, misses=0, maxsize=None, currsize=0)
    Cache DP function after all calls :   CacheInfo(hits=432, misses=259, maxsize=None, currsize=259)
    DP Best payoff  2.9570000000000003
    DP Best final state  (3, 2, 3, 4, 3)
    DP action list  ((0,), (1,), (0,), (2,), (1,), (0,), (3,), (2,), (4,), (3,), (2,), (4,), (3,), (4,), (3,))
    DP Computation took 0.015289068222045898 seconds
    
    -------------- BB computations -------------- 
    Cache BB optimistic_value function before first call :  CacheInfo(hits=0, misses=0, maxsize=None, currsize=0)
    Cache BB optimistic_value function after all calls:  CacheInfo(hits=8, misses=215, maxsize=None, currsize=215)
    BB Best payoff  2.9570000000000003
    BB Best final state  (3, 2, 3, 4, 3)
    BB action list  [(0,), (1,), (2,), (3,), (4,), (0,), (1,), (2,), (3,), (4,), (0,), (2,), (3,), (4,), (3,)]
    BB Computation took 0.027152538299560547 seconds
    = = = = = = = = = = = = = = = = = = = = 
    Mine of depth 5
    Level by level x,y slices
    level 0
    [[ 0.455  0.049  2.38   0.515]
     [ 0.801 -0.09  -1.815  0.708]
     [-0.857 -0.876 -1.936  0.316]]
    level 1
    [[ 0.579  1.311 -1.404 -0.236]
     [ 0.072 -1.191 -0.839 -0.227]
     [ 0.309  1.188 -3.055  0.97 ]]
    level 2
    [[-0.54  -0.061  1.518 -0.466]
     [-2.183 -1.083  0.457  0.874]
     [-1.623 -0.16  -0.535  1.097]]
    level 3
    [[-0.995  0.185 -0.856 -1.241]
     [ 0.858  0.78  -1.029  1.563]
     [ 0.364  0.888 -1.561  0.234]]
    level 4
    [[-0.771 -1.959  0.658 -0.354]
     [-1.504 -0.763  0.915 -2.284]
     [ 0.097 -0.546 -1.992 -0.296]]
    -------------- DP computations -------------- 
    Cache DP function before first call :  CacheInfo(hits=0, misses=0, maxsize=None, currsize=0)
    Cache DP function after all calls :   CacheInfo(hits=176863, misses=39098, maxsize=None, currsize=39098)
    DP Best payoff  5.713
    DP Best final state  ((2, 1, 1, 1), (1, 1, 0, 1), (0, 0, 0, 1))
    DP action list  ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (0, 0), (1, 3), (2, 3))
    DP Computation took 6.34877872467041 seconds
    
    -------------- BB computations -------------- 
    Cache BB optimistic_value function before first call :  CacheInfo(hits=0, misses=0, maxsize=None, currsize=0)
    Cache BB optimistic_value function after all calls:  CacheInfo(hits=491, misses=14165, maxsize=None, currsize=14165)
    BB Best payoff  5.713
    BB Best final state  ((2, 1, 1, 1), (1, 1, 0, 1), (0, 0, 0, 1))
    BB action list  [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 3), (2, 3), (0, 0)]
    BB Computation took 3.002690076828003 seconds
"""

import time
import numpy as np
from mining import Mine, search_dp_dig_plan, search_bb_dig_plan, find_action_sequence

np.set_printoptions(3)

some_2D_random = np.random.randn(3, 4)
some_2D_underground = np.array([
        [-0.814,  0.637, 1.824, -0.563],
        [ 0.559, -0.234,-0.366,  0.074],
        [ 0.175, -0.284,  0.026,-0.316],
        [ 0.212,  0.088,  0.304, 0.604],
        [-1.231,  1.558, -0.467,-0.371]
    ])
some_2D_state = np.array([
        0, 1, 2, 3, 2, 1, 0
    ])

some_3D_random = np.random.randn(3, 4, 5)
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

def UndergroundTest(underground):
    '''
    Test function
    
    Performs tests on
        - DP algorithm
        - BB algorithm

    Parameters
    ----------
    underground : np.array
        2D or 3D. Each element of the array contains 
        the profit value of the corresponding cell.
    state : np.array
        2D or 3D state with the same shape as underground
        used for find_action_state() test

    Returns
    -------
    None.
    '''
    mine = Mine(underground)
    mine.console_display()

    print('-------------- TestBB computations -------------- ')
    tic = time.time()
    best_payoff, best_a_list, best_final_state, ci = search_bb_dig_plan(mine)
    toc = time.time() 
    print('DP Best payoff:',best_payoff)
    print('DP Best final state:', best_final_state)  
    print('DP action list:', best_a_list)
    print('DP cache info:', ci)
    print('DP Computation took {} seconds\n'.format(toc-tic))   

    print('-------------- DP computations -------------- ')
    tic = time.time()
    best_payoff, best_a_list, best_final_state, ci = search_dp_dig_plan(mine)
    toc = time.time() 
    print('DP Best payoff:',best_payoff)
    print('DP Best final state:', best_final_state)  
    print('DP action list:', best_a_list)
    print('DP cache info:', ci)
    print('DP Computation took {} seconds\n'.format(toc-tic))   

    # print('-------------- BB computations -------------- ')
    # tic = time.time()
    # best_payoff, best_a_list, best_final_state = search_bb_dig_plan(mine)
    # toc = time.time() 
    # print('BB Best payoff:',best_payoff)
    # print('BB Best final state:', best_final_state)      
    # print('BB action list:', best_a_list)
    # print('BB Computation took {} seconds\n'.format(toc-tic))   

    
    # print('-------------- Find Action Sequence -------------- ')
    # initial_state = np.zeros(state.shape)
    # tic = time.time()
    # sequence = find_action_sequence(initial_state, state)
    # toc = time.time()
    # print('s0:\n', initial_state, '\ns1:\n', state)
    # print('Sequence:', sequence)
    # print('Computation took {} seconds\n'.format(toc-tic))  

    # print('-------------- Other Stuff -------------- ')
    # actions = mine.actions(state)
    # print('Possible Actions:\n', actions)

if __name__=='__main__':
    pass
    print('='*10 + " 2D UNDERGROUND TEST " + '='*10)
    UndergroundTest(some_2D_underground)

    print('='*10 + " 3D UNDERGROUND TEST " + '='*10)
    UndergroundTest(some_3D_underground)

    
