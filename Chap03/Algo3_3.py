#!/usr/bin/env python
# coding: utf-8

# Copyright (c) 2023, Douglas Santry
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, is permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# # This notebook is used to generate Algo3_3.py that Algo3_2.ipynb relies on.
# # It is not run directly.

# In[1]:


import numpy as np

from math import exp as exp


# In[2]:


#
# Sigmoid activation function and its derivative
#

def sigmoid (x):
    return 1/(1+exp(-x))

activation_f = np.vectorize (sigmoid)

def sigmoid_derivative (x):
    return x * (1 - x)

activation_dL = np.vectorize (sigmoid_derivative)


# In[3]:


#
# Given a topology allocate the matrices
#

def ANN_Build_Matrices (layers):
    
    ANN_Weights = []
    last = 1
    
    for layer in layers:
        
        ANN_Weights.append (np.random.random ((layer, last + 1)))
        last = layer

    return ANN_Weights


# In[4]:


#
# Given a topology build a neural network and packaged it in a dictionary
#

def ANN_Factory (topology):

    Weights = ANN_Build_Matrices (topology)

    dL = ANN_Build_Matrices (topology)
    for g in dL:
        g[:] = 0

    responses = []
    for layer in topology:
        responses.append (np.empty ((layer, 1), dtype=float))
        
    ANN = { "Weights" : Weights, "dL" : dL, "topology" : topology, "z" : responses }
    
    return ANN


# In[5]:


#
# Compute an ANN's value at x
#

def ForwardPass (ANN, x):
    
    z = np.array ([[x]])
    last = 1

    responses = ANN["z"]
    
    for idx, K in enumerate (zip (ANN["Weights"], responses)):

        z_ = K[0][:,1:]@z
        z_ += K[0][:, 0:1]
        z = responses[idx] = activation_f (z_)
        
    return z[0][0]


# In[6]:


eta = 0.01 # The learning rate for equation 3.40


# In[7]:


#
# Algorithm 3.3, Demystifying Deep Learning
#
# Compute the loss of an ANN and perform back propagation
#

def ComputeLoss (ANN, x, y):
    
    y_ = ForwardPass (ANN, x)
    
    dz = np.array ([[y_ - y]])
    loss = 0.5 * dz[0]*dz[0]

    dL = ANN["dL"]
    depth = -len (ANN["topology"])
    z = [np.array ([[x]])] + ANN["z"]

    for idx, K in enumerate (zip (reversed (ANN["dL"]), reversed (ANN["Weights"]), reversed (ANN["z"]))):

        # print ("dz", dz)
        
        df = activation_dL (K[2])
        delta = dz * df                                                 # Equation 3.38
        
        dL[-(idx+1)][:,0:1] += delta
                
        dL[-(idx+1)][:,1:] += delta@np.matrix.transpose (z[-(idx+2)])   # Equation 3.39

        dz = np.matrix.transpose (K[1][:,1:])@delta                     # Equation 3.37, propagate the gradient

    return loss[0]


# In[8]:


#
# Update the weights and zero out the accumulated gradient (reset)
#

def UpdateWeights (ANN):

    for W, dL in zip (ANN["Weights"], ANN["dL"]):
        
        W += -eta * dL                                        # Equation 3.40
        dL[:,:] = 0.0

