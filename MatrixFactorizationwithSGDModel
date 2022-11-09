#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 14:23:40 2022

@author: habibirani
"""

import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import random
from copy import copy, deepcopy
import scipy 
from scipy import sparse
from scipy.sparse import csr_matrix


r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('u.data',  sep='\t', names=r_cols,
encoding='latin-1')
ratings.head()

X = ratings.copy()
y = ratings['user_id']


R = X.pivot(index='user_id', columns='movie_id', values='rating')
R.head()



R = R.to_numpy()
R[np.isnan(R)] = 0

Train = deepcopy(R)
Test = deepcopy(R)

SR = csr_matrix(R)
SR.eliminate_zeros()
indptr = SR.indptr
data = SR.data
indices = SR.indices
mylist = list(range(1, 100000))
random.shuffle(mylist)
for m in range(len(indptr) - 1):
            for n in range(indptr[m], indptr[m + 1]):
                i = m
                j = indices[n]
                if mylist[i] <= 20000:
                    Train[i][j] = 0
                else:
                    Test[i][j] = 0
                    
print(Train)
print(Test)


def matrix_factorization(R, P, Q, K, steps=100, alpha, landa):
    # alpha is learning rate
    Q = Q.T       
    for step in range(steps):
         for m in range(len(indptr) - 1):
            for n in range(indptr[m], indptr[m + 1]):
                i = m 
                j = indices[n] 
                eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                
                bu = bu + alpha(eij - landa * bu) 
                bd = bd + alpha(eij - landa * bd)
    
                for k in range(K):
                        P[i][k] = 
                        P[i][k] + alpha * (2 * eij * Q[k][j] - landa * P[i][k])
                        Q[k][j] = 
                        Q[k][j] + alpha * (2 * eij * P[i][k] - landa * Q[k][j])
 
         e = 0
         for m in range(len(indptr) - 1):
                   for n in range(indptr[m], indptr[m + 1]):
                       i = m 
                       j = indices[n] 
                       e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                       for k in range(K):
                           e = 
                           e + (landa/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
         if e < 0.001:
             break
        
            
    return P, Q.T


def matrix_factorization(R, P, Q, K, steps=10000, alpha, alphafr, fr, landa):
    # alpha is GD learning rate
    # alphafr is fractional learning rate
    # fr is fractional parametr
    # landa is regularization term
    Q = Q.T
    # first step transpos Q matrix Q[j][k] to Q[k][j]
    for step in range(steps): 
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    # update P and Q
                    for k in range(K):
                        P[i][k] = 
                        P[i][k] + alpha * (2 * eij * Q[k][j] - landa * P[i][k]) 
                        - alphafr * (2 * math.gamma(2) / math.gamma(2 - fr) 
                        * R[i][j] * Q[k][j] * (pow(P[i][k],1 - fr)) 
                        + math.gamma(3) / math.gamma(3 - fr) * (pow(Q[k][j],2)) 
                        * (pow(P[i][k],2 - fr)) 
                        + 2 * landa * P[i][k] * math.gamma(2) 
                        / math.gamma(2 - fr) * (pow(P[i][k],1 - fr)))
                                                                                                                 
                        Q[k][j] = 
                        Q[k][j] + alpha * (2 * eij * P[i][k] - landa * Q[k][j]) 
                        - alphafr * (2 * math.gamma(2) / math.gamma(2 - fr) 
                        * R[i][j] * P.T[k][i] * (pow(Q[k][j],1 - fr)) 
                        + math.gamma(3) / math.gamma(3 - fr) * (pow(P[i][k],2)) 
                        * (pow(Q[k][j],2 - fr) )
                        + 2 * landa * Q[k][j] * math.gamma(2) 
                        / math.gamma(2 - fr) * (pow(Q[k][j],1 - fr)))
                        
        # evaluate error for stop function
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (landa/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
                        step = step + 1
        if e < 0.001:
            break
    #we have 2 ways to stop: first steps will be maxed & second e < min error
    return P, Q.T


def matrix_factorization(R, P, Q, K, steps=10000, alpha, fr, landa):
    # alpha is fractional learning rate
    # fr is fractional parametr
    # landa is regularization term
    Q = Q.T
    # first step is transpos Q matrix Q[j][k] to Q[k][j]
    for step in range(steps): 
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    # update P and Q
                    for k in range(K):
                        P[i][k] = 
                        P[i][k] - alpha * (2 * math.gamma(2) 
                        / math.gamma(2 - fr) * R[i][j] * Q[k][j] 
                        * (pow(P[i][k],1 - fr)) + math.gamma(3) 
                        / math.gamma(3 - fr) * (pow(Q[k][j],2)) 
                        * (pow(P[i][k],2 - fr)) 
                        + 2 * landa * P[i][k] * math.gamma(2) 
                        / math.gamma(2 - fr) * (pow(P[i][k],1 - fr)))
                                                                                                                 
                        Q[k][j] = 
                        Q[k][j] - alpha * (2 * math.gamma(2) 
                        / math.gamma(2 - fr) * R[i][j] * P.T[k][i] 
                        * (pow(Q[k][j],1 - fr)) + math.gamma(3) 
                        / math.gamma(3 - fr) * (pow(P[i][k],2)) 
                        * (pow(Q[k][j],2 - fr))
                        + 2 * landa * Q[k][j] * math.gamma(2) 
                        / math.gamma(2 - fr) * (pow(Q[k][j],1 - fr)))
        # evaluate error for stop function
        eR = np.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (landa/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
                        step = step + 1
        if e < 0.001:
            break
    # we have 2 ways to stop: first steps will be maxed & second e < min error
    return P, Q.T



N = len(R) 
M = len(R[0])
K = 20

P = np.random.rand(N,K)
Q = np.random.rand(M,K)

def calculate_RMSE(alpha , landa):
    nP , nQ = matrix_factorization(Train, P, Q, K, steps=50, alpha, landa)
    baias = sum(data)/1585183
    nR = np.dot(nP, nQ.T) + baias
    RMSE = np.sqrt(mean_squared_error(Test, nR))
    return RMSE


#evaluate adaptive learning rate
def alpha(alpha = 0.0002, epsilon = 0.001, beta = 0.001, steps = 1000):
    for step in range(steps -1): 
        eij = R[i][j] - np.dot(P[i,:],Q[:,j])
        Eij = pow(eij , 2)
        alphaij = sum((alpha / sqrt(Eij + epsilon)) + beta)
        
