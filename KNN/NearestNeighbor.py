# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# -*- coding: utf-8 -*-

import numpy as np
import operator
import itertools

def CountFrequency(my_list): 
    # Creating an empty dictionary  
        freq = {} 
        for item in my_list: 
            if (item in freq): 
                freq[item] += 1
            else: 
                freq[item] = 1
      
        return freq

class NearestNeighbor(object):
    #http://cs231n.github.io/classification/
    def __init__(self):
        pass

    def train(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y
        
    
  

  
  
 
    def predict(self, X, k= 1, l='L1'):
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]
        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
        # loop over all test rows
        for i in range(num_test):
            # find the nearest training example to the i'th test example
            if l == 'L1':
                # using the L1 distance (sum of absolute value differences)
                distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
            else:
                distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))
            
            #print(distances.shape)
            min_indices = distances.argsort()[:k]
            #print(min_indices)
            min_labels = []
            for index in min_indices:
                min_labels.append(self.ytr[index])
                
            
            
            #print(min_labels)
            #hash_min_labels = list(itertools.chain.from_iterable(min_labels))
            #print(hash_min_labels)
            freq_dict = CountFrequency(min_labels)
            pred_lbl = max(freq_dict.items(), key=operator.itemgetter(1))[0]
            #min_index = np.argmin(distances) # get the index with smallest distance
            Ypred[i] = pred_lbl # predict the label of the nearest example
    
        return Ypred
