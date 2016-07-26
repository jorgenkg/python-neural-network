from ...tools import dropout, add_bias
from ..commons.utils import *
import numpy as np
import collections
import random
import math


["check_network_structure", "verify_dataset_shape_and_modify", "print_training_status", "print_training_results"]


def backpropagation_foundation(network, trainingset, testset, cost_function, calculate_dW, evaluation_function = None, ERROR_LIMIT = 1e-3, max_iterations = (), batch_size = 0, input_layer_dropout = 0.0, hidden_layer_dropout = 0.0, print_rate = 1000, save_trained_network = False, **kwargs):
    check_network_structure( network, cost_function ) # check for special case topology requirements, such as softmax
    
    training_data, training_targets = verify_dataset_shape_and_modify( network, trainingset )
    test_data, test_targets    = verify_dataset_shape_and_modify( network, testset )
    
    
    # Whether to use another function for printing the dataset error than the cost function. 
    # This is useful if you train the network with the MSE cost function, but are going to 
    # classify rather than regress on your data.
    if evaluation_function != None:
        calculate_print_error = evaluation_function
    else:
        calculate_print_error = cost_function
    
    batch_size                 = batch_size if batch_size != 0 else training_data.shape[0] 
    batch_training_data        = np.array_split(training_data, math.ceil(1.0 * training_data.shape[0] / batch_size))
    batch_training_targets     = np.array_split(training_targets, math.ceil(1.0 * training_targets.shape[0] / batch_size))
    batch_indices              = range(len(batch_training_data))       # fast reference to batches
    
    error                      = calculate_print_error(network.update( test_data ), test_targets )
    reversed_layer_indexes     = range( len(network.layers) )[::-1]
    
    epoch                      = 0
    while error > ERROR_LIMIT and epoch < max_iterations:
        epoch += 1
        
        random.shuffle(batch_indices) # Shuffle the order in which the batches are processed between the iterations
        
        for batch_index in batch_indices:
            batch_data                 = batch_training_data[    batch_index ]
            batch_targets              = batch_training_targets[ batch_index ]
            batch_size                 = float( batch_data.shape[0] )
            
            input_signals, derivatives = network.update( batch_data, trace=True )
            out                        = input_signals[-1]
            cost_derivative            = cost_function( out, batch_targets, derivative=True ).T
            delta                      = cost_derivative * derivatives[-1]
            
            for i in reversed_layer_indexes:
                # Loop over the weight layers in reversed order to calculate the deltas
            
                # perform dropout
                dropped = dropout( 
                            input_signals[i], 
                            # dropout probability
                            hidden_layer_dropout if i > 0 else input_layer_dropout
                        )
            
                # calculate the weight change
                dX = (np.dot( delta, add_bias(dropped) )/batch_size).T
                dW = calculate_dW( i, dX )
                
                if i != 0:
                    """Do not calculate the delta unnecessarily."""
                    # Skip the bias weight
                    weight_delta = np.dot( network.weights[ i ][1:,:], delta )
    
                    # Calculate the delta for the subsequent layer
                    delta = weight_delta * derivatives[i-1]
                
                # Update the weights with Nestrov Momentum
                network.weights[ i ] += dW
            #end weight adjustment loop
        
        error = calculate_print_error(network.update( test_data ), test_targets )
        
        if epoch%print_rate==0:
            # Show the current training status
            print "[training] Current error:", error, "\tEpoch:", epoch
    
    print "[training] Finished:"
    print "[training]   Converged to error bound (%.4g) with error %.4g." % ( ERROR_LIMIT, error )
    print "[training]   Measured quality: %.4g" % network.measure_quality( training_data, training_targets, cost_function )
    print "[training]   Trained for %d epochs." % epoch
    
    if save_trained_network and confirm( promt = "Do you wish to store the trained network?" ):
        network.save_network_to_file()
# end backprop