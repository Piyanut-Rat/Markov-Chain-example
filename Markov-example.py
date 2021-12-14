#import library
import numpy as np 
import matplotlib.pyplot as plt 

#input parameters
all_posible_state = [0,1] # 0 Sunny | 1 Rainy
num_time_step = 10


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

#----------------------------------------------------#
#testing
true_init_prob, true_trans_prob = true_table()