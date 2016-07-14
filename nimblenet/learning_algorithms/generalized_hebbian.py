from ..tools import dropout, add_bias, confirm
from ..activation_functions import softmax_function
from ..cost_functions import softmax_neg_loss
import numpy as np
import collections
import random
import math


# NOT YET IMPLEMENTED
def generalized_hebbian(network, trainingset, testset, cost_function, learning_rate = 0.001, max_iterations = (), save_trained_network = False ):
    assert softmax_function != network.layers[-1][1] or cost_function == softmax_neg_loss,\
        "When using the `softmax` activation function, the cost function MUST be `softmax_neg_loss`."
    assert cost_function != softmax_neg_loss or softmax_function == network.layers[-1][1],\
        "When using the `softmax_neg_loss` cost function, the activation function in the final layer MUST be `softmax`."
        
    assert trainingset[0].features.shape[0] == network.n_inputs, \
        "ERROR: input size varies from the defined input setting"
    assert trainingset[0].targets.shape[0]  == network.layers[-1][0], \
        "ERROR: output size varies from the defined output setting"
    
    training_data              = np.array( [instance.features for instance in trainingset ] )
    training_targets           = np.array( [instance.targets  for instance in trainingset ] )
                                
    layer_indexes               = range( len(network.layers) )
    epoch                       = 0
                                
    input_signals, derivatives  = network.update( training_data, trace=True )
    error                       = cost_function(input_signals[-1], training_targets )
    input_signals[-1]          -= training_targets
    
    while error > 0.01 and epoch < max_iterations:
        epoch += 1
        
        for i in layer_indexes:
            forgetting_term     = np.dot(network.weights[i], np.tril(np.dot( input_signals[i+1].T, input_signals[i+1] )))
            activation_product  = np.dot(add_bias(input_signals[i]).T, input_signals[i+1])
            dW                  = learning_rate * (activation_product - forgetting_term)
            network.weights[i] += dW
            
            # normalize the weight to prevent the weights from growing unbounded
            #network.weights[i]     /= np.sqrt(np.sum(network.weights[i]**2))
        #end weight adjustment loop
        
        input_signals, derivatives  = network.update( training_data, trace=True )
        error                       = cost_function(input_signals[-1], training_targets )
        input_signals[-1]          -= training_targets
        
        if epoch % 1000 == 0:
            print "[training] Error:", error
    
    print "[training] Finished:"
    print "[training]   Trained for %d epochs." % epoch
    
    if save_trained_network and confirm( promt = "Do you wish to store the trained network?" ):
        network.save_network_to_file()
# end hebbian