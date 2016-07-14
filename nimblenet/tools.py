import numpy as np
import os


def print_test( network, testset, cost_function ):
    assert testset[0].features.shape[0] == network.n_inputs, \
        "ERROR: input size varies from the defined input setting"
    assert testset[0].targets.shape[0]  == network.layers[-1][0], \
        "ERROR: output size varies from the defined output setting"
    
    test_data              = np.array( [instance.features for instance in testset ] )
    test_targets           = np.array( [instance.targets  for instance in testset ] )
    
    input_signals, derivatives = network.update( test_data, trace=True )
    out                        = input_signals[-1]
    error                      = cost_function(out, test_targets )
    
    print "[testing] Network error: %.4g" % error
    print "[testing] Network results:"
    print "[testing]   input\tresult\ttarget"
    for entry, result, target in zip(test_data, out, test_targets):
        print "[testing]   %s\t%s\t%s" % tuple(map(str, [entry, result, target]))
#end
    

def dropout( X, p = 0. ):
    if p != 0:
        retain_p = 1 - p
        X = X * np.random.binomial(1,retain_p,size = X.shape)
        X /= retain_p
    return X
#end  


def add_bias(A):
    # Add a bias value of 1. The value of the bias is adjusted through
    # weights rather than modifying the input signal.
    return np.hstack(( np.ones((A.shape[0],1)), A ))
#end addBias


def confirm( promt='Do you want to continue?' ):
	prompt = '%s [%s|%s]: ' % (promt,'y','n')
	while True:
		ans = raw_input(prompt).lower()
		if ans in ['y','yes']:
			return True
		if ans in ['n','no']:
			return False
		print "Please enter y or n."
#end