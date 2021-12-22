#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 18:32:18 2021

@author: piyanut

cite: https://www.youtube.com/watch?v=B9LOMA43tPA&list=PLlCuQCjEEYuRnzjPrZh6ErsB-zfMitjTO&index=2
"""

#import library
import numpy as np 
import matplotlib.pyplot as plt 

#input parameters
all_posible_state = [0,1] # 0 Sunny | 1 Rainy
num_time_step = 10
num_sequences = 10


#----------------------------------------------------#
def true_table():
    #true transition probability table

    """
    0 Sunny | 1 Rainy
    -------------------
    row | col
    t-1 | t
    0 | 0 : P(Sunny t | Sunny t-1) = 0.9
    0 | 1 : P(Rainy t | Sunny t-1) = 0.1
    1 | 0 : P(Sunny t | Rainy t-1) = 0.1
    1 | 1 : P(Rainy t | Rainy t-1) = 0.9
    """

    true_trans_prob = np.array([[0.9, 0.1],
                                [0.1, 0.9]])

    #true initial probability table
    """
    row
    t=0
    0 : P(Sunny t=0)
    1 : P(Rainy t=0)
    """
    true_init_prob = np.array([0.5, 0.5])

    return true_init_prob, true_trans_prob

#sequence generator
def sequence_generator(true_init_prob, true_trans_prob, num_time_step, num_sequences):
    
    all_sequences = [] #list collect all sequence
    
    #for each sequence
    for i in range(0, num_sequences):
        one_sequence = []
        
        #for each time step
        for j in range(0, num_time_step):
            random_num = np.random.uniform() # ramdom generate number 0 to 1
            
            # if 1st state
            if j == 0: 
                if random_num < true_init_prob[0]:
                    one_sequence.append(0) #if Sunny
                else:
                    one_sequence.append(1) #if Rainy
                    
            # if not
            else:
                current_state = one_sequence[j-1]
                if random_num < np.amin(true_trans_prob[current_state,:]):
                    one_sequence.append(np.argmin(true_trans_prob[current_state,:]))
                else:
                    one_sequence.append(np.argmax(true_trans_prob[current_state,:]))
                    
        #fin one sequence
        all_sequences.append(one_sequence)
    return all_sequences
    
#----------------------------------------------------#
#----------------------------------------------------#
#testing
true_init_prob, true_trans_prob = true_table()

all_sequences = sequence_generator(true_init_prob, true_trans_prob, num_time_step, num_sequences)



