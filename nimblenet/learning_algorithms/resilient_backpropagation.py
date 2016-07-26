from ..tools import add_bias, confirm
from ..activation_functions import softmax_function
from ..cost_functions import softmax_neg_loss
import numpy as np


def resilient_backpropagation(network, trainingset, testset, cost_function, ERROR_LIMIT=1e-3, max_iterations = (), weight_step_max = 50., weight_step_min = 0., start_step = 0.5, learn_max = 1.2, learn_min = 0.5, print_rate = 1000, save_trained_network = False ):
    # Implemented according to iRprop+ 
    # http://sci2s.ugr.es/keel/pdf/algorithm/articulo/2003-Neuro-Igel-IRprop+.pdf
    
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
    test_data                  = np.array( [instance.features  for instance in testset ] )
    test_targets               = np.array( [instance.targets  for instance in testset ] )
    
    # Storing the current / previous weight step size
    weight_step                = [ np.full( weight_layer.shape, start_step ) for weight_layer in network.weights ]
    
    # Storing the current / previous weight update
    dW                         = [  np.ones(shape=weight_layer.shape) for weight_layer in network.weights ]
    
    # Storing the previous derivative
    previous_dEdW              = [ 1 ] * len( network.weights )
    
    # Storing the previous error measurement
    prev_error                 = ( )                             # inf
    
    input_signals, derivatives = network.update( training_data, trace=True )
    out                        = input_signals[-1]
    cost_derivative            = cost_function(out, training_targets, derivative=True).T
    delta                      = cost_derivative * derivatives[-1]
    error                      = cost_function(network.update( test_data ), test_targets )
    
    n_samples                  = float(training_data.shape[0])
    layer_indexes              = range( len(network.layers) )[::-1] # reversed
    epoch                      = 0
    
    while error > ERROR_LIMIT and epoch < max_iterations:
        epoch       += 1
        
        for i in layer_indexes:
            # Loop over the weight layers in reversed order to calculate the deltas
                   
            # Calculate the delta with respect to the weights
            dEdW = (np.dot( delta, add_bias(input_signals[i]) )/n_samples).T
            
            if i != 0:
                """Do not calculate the delta unnecessarily."""
                # Skip the bias weight
                weight_delta = np.dot( network.weights[ i ][1:,:], delta )
    
                # Calculate the delta for the subsequent layer
                delta = weight_delta * derivatives[i-1]
            
            
            # Calculate sign changes and note where they have changed
            diffs            = np.multiply( dEdW, previous_dEdW[i] )
            pos_indexes      = np.where( diffs > 0 )
            neg_indexes      = np.where( diffs < 0 )
            zero_indexes     = np.where( diffs == 0 )
            
            
            # positive
            if np.any(pos_indexes):
                # Calculate the weight step size
                weight_step[i][pos_indexes] = np.minimum( weight_step[i][pos_indexes] * learn_max, weight_step_max )
                
                # Calculate the weight step direction
                dW[i][pos_indexes] = np.multiply( -np.sign( dEdW[pos_indexes] ), weight_step[i][pos_indexes] )
                
                # Apply the weight deltas
                network.weights[i][ pos_indexes ] += dW[i][pos_indexes]
            
            # negative
            if np.any(neg_indexes):
                weight_step[i][neg_indexes] = np.maximum( weight_step[i][neg_indexes] * learn_min, weight_step_min )
                
                if error > prev_error:
                    # iRprop+ version of resilient backpropagation
                    network.weights[i][ neg_indexes ] -= dW[i][neg_indexes] # backtrack
                
                dEdW[ neg_indexes ] = 0
            
            # zeros
            if np.any(zero_indexes):
                dW[i][zero_indexes] = np.multiply( -np.sign( dEdW[zero_indexes] ), weight_step[i][zero_indexes] )
                network.weights[i][ zero_indexes ] += dW[i][zero_indexes]
            
            # Store the previous weight step
            previous_dEdW[i] = dEdW
        #end weight adjustment loop
        
        prev_error                 = error
        
        input_signals, derivatives = network.update( training_data, trace=True )
        out                        = input_signals[-1]
        cost_derivative            = cost_function(out, training_targets, derivative=True).T
        delta                      = cost_derivative * derivatives[-1]
        error                      = cost_function(network.update( test_data ), test_targets )
        
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