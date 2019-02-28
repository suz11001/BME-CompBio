#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 12:41:53 2019

@author: sumairazaman
"""

import numpy as np 
import random 
import scipy.stats as ss 
from sklearn import datasets 
from random import randrange

def distance(p1, p2): 
    return np.sqrt(np.sum(np.power(p2-p1, 2))) #distance between two points  

def majority_vote(votes): 
    vote_counts = {} 
    for vote in votes: 
        if vote in vote_counts: 
           vote_counts[vote]+= 1
        else: 
            vote_counts[vote]= 1
    winners = [] 
    max_count = max(vote_counts.values()) 
    for vote, count in vote_counts.items(): 
        if count == max_count: 
            winners.append(vote) 
    return random.choice(winners) #returns winner randomly if there are more than 1 winner 
  
  
def majority_vote_short(votes): 
    mode, count = ss.mstats.mode(votes) 
    return mode 
  
def find_nearest_neighbours(p, points, k = 5):  #algorithm to find the nearest neighbours 
    distances = np.zeros(points.shape[0]) 
    for i in range(len(distances)): 
        distances[i]= distance(p, points[i]) 
    ind = np.argsort(distances)      #returns index, according to sorted values in array 
    return ind[:k] 
  
def knn_predict(p, points, outcomes, k = 5): 
    ind = find_nearest_neighbours(p, points, k) 
    return majority_vote(outcomes[ind]) 

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def cross_validation_split(dataset, folds=5):
    dataset_split = list()
    pred_ind=[]
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
		fold = list()
		indices=[]
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
			indices.append(index)
		dataset_split.append(fold)
		pred_ind.append(indices)
    #print len(pred_ind)
    return dataset_split,pred_ind


if __name__=="__main__":
    iris = datasets.load_iris() 
    predictors = iris.data[:, 0:2] 
    outcomes = iris.target 
    my_predictions = np.array([knn_predict(p, predictors, outcomes, 5) for p in predictors]) 
    f,ids=cross_validation_split(iris.data[:, 0:2])
    x=getAccuracy(my_predictions,outcomes)
    print "Accuracy of our model is: ", x

