#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 13:45:21 2019

@author: sumairazaman
"""
import numpy as np

# classes/functions go here
def viterbi_alg(A_mat, O_mat, observations):
    # get number of states
    ##num_obs = observations.size
    num_obs=len(observations)
    num_states = A_mat.shape[0]
    #print num_states
    # initialize path costs going into each state
    log_probs = np.array([0.2,0.4,0.4])
    # initialize arrays to store best paths, 1 row for each ending state
    ##paths = np.zeros( (num_states, num_obs+1 ))
    paths = np.zeros( (num_states, num_obs ))
    paths[:, 0] = np.arange(num_states)
    # start looping
    for obs_ind, obs_val in enumerate(observations):
        # for each obs, need to check for best path into each state
        for state_ind in xrange(num_states):
            # given observation, check prob of each path
            temp_probs = log_probs + \
                         np.log(O_mat[state_ind, obs_val]) + \
                         np.log(A_mat[:, state_ind])
            print temp_probs
            # check for largest score
            best_temp_ind = np.argmax(temp_probs)
            print best_temp_ind
            # save the path with a higher prob and score
            paths[state_ind,:] = paths[best_temp_ind,:]
            paths[state_ind,(obs_ind)] = state_ind
            # update the time step i-1
            log_probs = temp_probs
    # we now have the best stuff going into each path, find the best score
    best_path_ind = np.argmax(log_probs)
    # done, get out.
    return (best_path_ind, paths, log_probs)
 
# main script stuff goes here
if __name__ == '__main__':
    # the transition matrix
    A_mat = np.array([[0.1, 0.5, 0.4], [0.9, 0.05, 0.05], [0.7, 0.2, 0.1]])
    #print A_mat
    # the observation matrix
    O_mat = np.array([[0.7, 0.2, 0.1], [0.9, 0.05, 0.05], [0.6, 0.3, 0.1]])
    eat_obs=[2,0,1,2,0]
    # we have what we need, do viterbi
    best_path_ind, paths, log_probs = viterbi_alg(A_mat, O_mat, eat_obs)
    print "observation is " + str(eat_obs)
    print "best path for observation is" + str(paths[best_path_ind,:])
