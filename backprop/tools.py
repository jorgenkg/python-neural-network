import numpy as np
import os

class Instance:
    # This is a simple encapsulation of a `input signal : output signal`
    # pair in our training set.
    def __init__(self, features, target):
        self.features = np.array(features)
        self.targets  = np.array(target)
#endclass Instance


def dropout( X, p = 0. ):
    if p:
        retain_p = 1 - p
        X = X * np.random.binomial(1,retain_p,size = X.shape)
        X /= retain_p
    return X
#end  


def add_bias(A):
    return np.hstack(( np.ones((A.shape[0],1)), A )) # Add 1 as bias.
#end addBias


def confirm( promt='Do you want to continue?' ):
	prompt = '%s [%s|%s]: ' % (promt,'y','n')
	while True:
		ans = raw_input(prompt).lower()
		if ans.lower() in ['y','yes']:
			return True
		if ans.lower() in ['n','no']:
			return False
		print "Please enter y or n."
#end

def load_wine_data():
    # UCI Machine Learning Repository http://archive.ics.uci.edu/ml/datasets/Wine
    filename = "wine.data.txt"
    
    with open( filename, "r" ) as f:
        entries = f.readlines()
    
    data = np.array([
        entry[1:]
        for entry in map(lambda x: map(float, x.split(",")), entries)
    ])
    
    data /= np.max(data, axis=0) # naive value normalization
    
    assignments = [
        int(entry[0]) # classification as integer
        for entry in map(lambda x: map(float, x.split(",")), entries)
    ]
    
    
    targets = np.zeros( (data.shape[0], max(assignments)) )
    
    for i, classification in enumerate( assignments ):
        # One-hot encoding
        targets[i][classification-1] = 1
    
    
    
    return [
        Instance( feature, target )
        for feature, target in zip(data, targets)
    ]
#end
    
    