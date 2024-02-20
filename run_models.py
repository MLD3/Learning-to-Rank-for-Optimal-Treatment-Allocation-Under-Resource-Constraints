import argparse
import numpy as np
import sys 
import pickle
import os 
import subprocess

n_tree = 1000 #Number of trees to use in random forest
n_sample = 1.0 #Size of sample to use in each tree
min_samples_leaf_curr = 2 #Minimum number of samples that a leaf can have
min_samples_split_curr = 20 #Mininum number of samples needed to split a leaf
max_depth_curr = 10 #Maximum depth of tree
min_impurity_curr = -10000.0 #Amount a split must improve over the baseline 
method = 1

flname = 'train_model.py'

to_string = [n_tree, n_sample, min_samples_leaf_curr, min_samples_split_curr, hybrid,
         min_impurity_curr, max_depth_curr]

print(to_string)

to_string = [str(x) for x in to_string]
        
subprocess.run(['python3', '-W', 'ignore', flname] + to_string)