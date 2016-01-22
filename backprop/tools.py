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