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



#est trans prob: Makov learning base on counting method

def trans_prob_estimator(all_sequences):
    est_trans_prob = np.zeros((2,2)) #define dimention of transition of probability 2x2
    
    for row in range(est_trans_prob.shape[0]): #previous state
        for col in range(est_trans_prob.shape[1]): #next state
            #target transition
            target_transition = [row, col]
            
            #convert 2 timestep for each sequence
            for each_step in range(1, len(all_sequences[0])):
                this_transition = np.array(all_sequences)[:,[each_step-1, each_step]].tolist()
                
                for each_seq in this_transition:
                    if each_seq == target_transition:
                        est_trans_prob[row, col] +=1
                        
            #sum up to one
            
            est_trans_prob[row, col] /= len(np.argwhere(np.array(all_sequences)[:,:-1] == row))
    return est_trans_prob
                        
#est init prob
def init_prob_estimator(all_sequences):
    init_seq = np.array(all_sequences)[:,0]
    est_init_prob = np.zeros((2))
    
    for i in range(len(all_posible_state)):
        est_init_prob[i] = len(np.argwhere(init_seq == i))
    #sum up to one
    return est_init_prob/est_init_prob.sum()

#predict function
def prediction(est_trans_prob, train_seq):
    next_state_list = []
    for each_seq in range(num_sequences):
        current_state = train_seq[each_seq][-1]
        
        next_state = np.argmax(est_trans_prob[current_state, :])
        next_state_list.append(next_state)
    return np.array(next_state_list)

"""
#----------------------------------------------------#

#testing
true_init_prob, true_trans_prob = true_table()

all_sequences = sequence_generator(true_init_prob, true_trans_prob, num_time_step, num_sequences)

est_trans_prob = trans_prob_estimator(all_sequences)
print("check sum up to 1: ", est_trans_prob.sum(axis = 1))  

est_init_prob = init_prob_estimator(all_sequences)
print("check sum up to 1: ", est_init_prob.sum())
"""
#----------------------------------------------------#
#----------------------------------------------------#

#test1: table : SSE
#input parameters
all_posible_state = [0,1] # 0 Sunny | 1 Rainy
num_time_step = 10
true_init_prob, true_trans_prob = true_table()

SSE_list = []
for num_sequences in range(1, 100):
    #generate data
    all_sequences = sequence_generator(true_init_prob, true_trans_prob, num_time_step, num_sequences)
    
    #estimate transition probability table
    est_trans_prob = trans_prob_estimator(all_sequences)
    
    #measure performance
    SSE = ((true_init_prob -est_trans_prob)**2).sum()
    SSE_list.append(SSE)
    
#visualise
plt.figure()
plt.plot(SSE_list)
plt.show()

#test2: result : accuracy

#input parameters
all_posible_state = [0,1] # 0 Sunny | 1 Rainy
num_time_step = 10
num_sequences = 10
true_init_prob, true_trans_prob = true_table()
all_sequences = sequence_generator(true_init_prob, true_trans_prob, num_time_step, num_sequences)

#train test split
train_seq = np.array(all_sequences)[:,:-1].tolist()
test_seq = np.array(all_sequences)[:,-1]

#training / estimation
est_trans_prob = trans_prob_estimator(all_sequences)

#prediction
pred_seq = prediction(est_trans_prob, train_seq)

#accuracy
print("% acc : ", (pred_seq == test_seq).sum()/len(test_seq) *100)



#------------- END ----------------------------#
