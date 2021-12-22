#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 18:13:38 2021

@author: piyanut

cite: https://medium.com/@balamurali_m/markov-chain-simple-example-with-python-985d33b14d19
"""

#example #1: I = [1, 0]

import numpy as np
#Current state
I = np.matrix([[1, 0]])

#Transition Matrix
T = np.matrix([[.7, 0.3],
               [.6, 0.4]])
T1 = I * T
# After 1 hours
print (T1)
T2 = T1 * T
# After 2 hours
print (T2)
T3 = T2 * T
# After 3 hours
print (T3)

#---------------------------------------#

#example #2: I = [0.5, 0.5]

#Current state
I_new = np.matrix([[0.5, 0.5]])

T1_new = I_new * T
# After 1 hours
print (T1_new)
T2_new = T1_new * T
# After 2 hours
print (T2_new)
T3_new = T2_new * T
# After 3 hours
print (T3_new)

#------------- END ----------------------------#
