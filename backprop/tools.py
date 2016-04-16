import numpy as np
import os


def print_test( network, testset, cost_function ):
    assert testset.features.shape[1] == network.n_inputs, \
            "ERROR: input size varies from the defined input setting"

    assert testset.targets.shape[1]  == network.layers[-1][0], \
            "ERROR: output size varies from the defined output setting"
            
    test_data    = testset.features
    test_targets = testset.targets
    
    input_signals, derivatives = network.update( test_data, trace=True )
    out                        = input_signals[-1]
    error                      = cost_function(out, test_targets )
    
    print "[testing] Network error: %.4g" % error
    #print "[testing] Network results:"
    #print "[testing]   input\tresult\ttarget"
    #for entry, result, target in zip(test_data, out, test_targets):
    #    print "[testing]   %s\t%s\t%s" % tuple(map(str, [entry, result, target]))
#end


def save_network_to_file( network, filename = "network0.pkl" ):
    import cPickle, os, re
    """
    This save method pickles the parameters of the current network into a 
    binary file for persistant storage.
    """
    
    if filename == "network0.pkl":
        while os.path.exists( os.path.join(os.getcwd(), filename )):
            filename = re.sub('\d(?!\d)', lambda x: str(int(x.group(0)) + 1), filename)
    
    with open( filename , 'wb') as file:
        store_dict = {
            "n_inputs"             : network.n_inputs,
            "layers"               : network.layers,
            "n_weights"            : network.n_weights,
            "weights"              : network.weights,
        }
        cPickle.dump( store_dict, file, 2 )
#end


def load_network_from_file( filename ):
    import cPickle
    """
    Load the complete configuration of a previously stored network.
    """
    network = NeuralNet( {"n_inputs":1, "layers":[[0,None]]} )
    
    with open( filename , 'rb') as file:
        store_dict                   = cPickle.load(file)
        
        network.n_inputs             = store_dict["n_inputs"]            
        network.n_weights            = store_dict["n_weights"]           
        network.layers               = store_dict["layers"]
        network.weights              = store_dict["weights"]             
    
    return network
#end
    

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