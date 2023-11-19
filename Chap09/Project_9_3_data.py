#!/usr/bin/env python
# coding: utf-8

# /*
# 
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
# 
# */
# 

# In[1]:


import pandas as pd
import glob
import os
import pathlib

import tensorflow as tf

from datasets import Dataset, DatasetDict


# In[2]:


DataLocation = "/Users/dsantry/Scratch/Data/BBC/bbc"


# In[3]:


folders = glob.glob (f"{DataLocation}/*")


# In[4]:


folders = [*filter (lambda u : True if(os.path.isdir (u)) else False, folders)]


# In[5]:


folders


# In[6]:


labels = [os.path.basename (u) for u in folders]
labels


# In[7]:



TrainingSetX = []
TrainingLabels = []
DocIDList = []
DocID = 0
labelZ = 0

for folder, label in zip (folders, labels):
    
    examples = glob.glob (f"{folder}/*")

    for example in examples:

        DocID += 1 #

        fd = open (example, "r", errors="ignore")
        text = fd.readlines () #.decode(errors='replace')
        fd.close ()

        # *** Strip the \n
        
        Nstrings = len (text)

        index = 0
        
        while index < Nstrings:
            
            if index >= Nstrings:
                break
                
            text[index] = text[index].rstrip ("\n")
            
            if len (text[index]) == 0:
                text.pop (index)
                Nstrings -= 1
            else:
                index += 1

        # *** build the LLM block size segments for the example

        total = sum ([*map (len, text)])
        index = 0
        segmentLen = 0
        segment = ""

        while total > 0:

            u = text[index]
            sentenceLength = len (u)
            
            if sentenceLength + segmentLen < 3072:
                segment += u
                segmentLen += sentenceLength
            else:
                TrainingSetX.append (segment)
                TrainingLabels.append (labelZ)
                DocIDList.append (DocID)
                segment = u
                segmentLen = sentenceLength
        
            total -= sentenceLength
            index += 1

        TrainingSetX.append (segment)
        TrainingLabels.append (labelZ)
        DocIDList.append (DocID)
        
        # per example end
        
    labelZ += 1 # outer loop, per category


# In[8]:


df = pd.DataFrame (list (zip (TrainingSetX, TrainingLabels, DocIDList)), columns =["Text", "Label", "DocID"])


# In[9]:


df


# In[10]:


W = Dataset.from_pandas(df)


# In[11]:


W

