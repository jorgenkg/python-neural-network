from tools import dropout, add_bias, confirm
from activation_functions import softmax_function
from cost_functions import softmax_neg_loss
import numpy as np
import collections
import random
import math

all = ["backpropagation", "scaled_conjugate_gradient", "scipyoptimize", "resilient_backpropagation", "generalized_hebbian"]


def backpropagation(network, trainingset, testset, cost_function, evaluation_function = None, ERROR_LIMIT = 1e-3, learning_rate = 0.03, momentum_factor = 0.9, max_iterations = (), batch_size = 0, input_layer_dropout = 0.0, hidden_layer_dropout = 0.0, print_rate = 1000, save_trained_network = False  ):
    assert softmax_function != network.layers[-1][1] or cost_function == softmax_neg_loss,\
        "When using the `softmax` activation function, the cost function MUST be `softmax_neg_loss`."
    assert cost_function != softmax_neg_loss or softmax_function == network.layers[-1][1],\
        "When using the `softmax_neg_loss` cost function, the activation function in the final layer MUST be `softmax`."
        
    assert trainingset[0].features.shape[0] == network.n_inputs, \
        "ERROR: input size varies from the defined input setting"
    assert trainingset[0].targets.shape[0]  == network.layers[-1][0], \
        "ERROR: output size varies from the defined output setting"
    
    # Whether to use another function for printing the dataset error than the cost function. 
    # This is useful if you train the network with the MSE cost function, but are going to 
    # classify rather than regress on your data.
    calculate_print_error      = evaluation_function if evaluation_function != None else cost_function
    
    training_data              = np.array( [instance.features for instance in trainingset ] )
    training_targets           = np.array( [instance.targets  for instance in trainingset ] )
    test_data                  = np.array( [instance.features for instance in testset ] )
    test_targets               = np.array( [instance.targets  for instance in testset ] )
    
    batch_size                 = batch_size if batch_size != 0 else training_data.shape[0] 
    batch_training_data        = np.array_split(training_data, math.ceil(1.0 * training_data.shape[0] / batch_size))
    batch_training_targets     = np.array_split(training_targets, math.ceil(1.0 * training_targets.shape[0] / batch_size))
    batch_indices              = range(len(batch_training_data))       # fast reference to batches
    
    error                      = calculate_print_error(network.update( test_data ), test_targets )
    reversed_layer_indexes     = range( len(network.layers) )[::-1]
    momentum                   = collections.defaultdict( int )
    
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
                dW = -learning_rate * (np.dot( delta, add_bias(dropped) )/batch_size).T + momentum_factor * momentum[i]
            
                if i != 0:
                    """Do not calculate the delta unnecessarily."""
                    # Skip the bias weight
                    weight_delta = np.dot( network.weights[ i ][1:,:], delta )
    
                    # Calculate the delta for the subsequent layer
                    delta = weight_delta * derivatives[i-1]
            
                # Store the momentum
                momentum[i] = dW
                                
                # Update the weights
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


def scipyoptimize(network, trainingset, testset, cost_function, method = "Newton-CG", save_trained_network = False  ):
    from scipy.optimize import minimize
    
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
    
    error_function_wrapper     = lambda weights, training_data, training_targets, test_data, test_targets, cost_function: network.error( weights, test_data, test_targets, cost_function )
    gradient_function_wrapper  = lambda weights, training_data, training_targets, test_data, test_targets, cost_function: network.gradient( weights, training_data, training_targets, cost_function )
        
    results = minimize( 
        error_function_wrapper,                         # The function we are minimizing
        network.get_weights(),                          # The vector (parameters) we are minimizing
        method  = method,                               # The minimization strategy specified by the user
        jac     = gradient_function_wrapper,            # The gradient calculating function
        args    = (training_data, training_targets, test_data, test_targets, cost_function),  # Additional arguments to the error and gradient function
    )
    
    network.set_weights( results.x )
    
    
    if not results.success:
        print "[training] WARNING:", results.message
        print "[training]   Terminated with error %.4g." % results.fun
    else:
        print "[training] Finished:"
        print "[training]   Completed with error %.4g." % results.fun
        print "[training]   Measured quality: %.4g" % network.measure_quality( training_data, training_targets, cost_function )
        
        if save_trained_network and confirm( promt = "Do you wish to store the trained network?" ):
            network.save_network_to_file()
#end


def scaled_conjugate_gradient(network, trainingset, testset, cost_function, ERROR_LIMIT = 1e-6, max_iterations = (), print_rate = 1000, save_trained_network = False ):
    # Implemented according to the paper by Martin F. Moller
    # http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.38.3391
     
    assert softmax_function != network.layers[-1][1] or cost_function == softmax_neg_loss,\
        "When using the `softmax` activation function, the cost function MUST be `softmax_neg_loss`."
    assert cost_function != softmax_neg_loss or softmax_function == network.layers[-1][1],\
        "When using the `softmax_neg_loss` cost function, the activation function in the final layer MUST be `softmax`."
        
    assert trainingset[0].features.shape[0] == network.n_inputs, \
        "ERROR: input size varies from the defined input setting"
    assert trainingset[0].targets.shape[0]  == network.layers[-1][0], \
        "ERROR: output size varies from the defined output setting"
    
    training_data       = np.array( [instance.features for instance in trainingset ] )
    training_targets    = np.array( [instance.targets  for instance in trainingset ] )
    test_data           = np.array( [instance.features  for instance in testset ] )
    test_targets        = np.array( [instance.targets  for instance in testset ] )

    ## Variables
    sigma0              = 1.e-6
    lamb                = 1.e-6
    lamb_               = 0

    vector              = network.get_weights() # The (weight) vector we will use SCG to optimalize
    grad_new            = -network.gradient( vector, training_data, training_targets, cost_function )
    r_new               = grad_new
    # end

    success             = True
    k                   = 0
    while k < max_iterations:
        k               += 1
        r               = np.copy( r_new     )
        grad            = np.copy( grad_new  )
        mu              = np.dot(  grad,grad )
    
        if success:
            success     = False
            sigma       = sigma0 / math.sqrt(mu)
            s           = (network.gradient(vector+sigma*grad, training_data, training_targets, cost_function)-network.gradient(vector,training_data, training_targets,cost_function))/sigma
            delta       = np.dot( grad.T, s )
        #end
    
        # scale s
        zetta           = lamb-lamb_
        s              += zetta*grad
        delta          += zetta*mu
    
        if delta < 0:
            s          += (lamb - 2*delta/mu)*grad
            lamb_       = 2*(lamb - delta/mu)
            delta      -= lamb*mu
            delta      *= -1
            lamb        = lamb_
        #end
    
        phi             = np.dot( grad.T,r )
        alpha           = phi/delta
    
        vector_new      = vector+alpha*grad
        f_old, f_new    = network.error(vector, test_data, test_targets, cost_function), network.error(vector_new, test_data, test_targets, cost_function)
    
        comparison      = 2 * delta * (f_old - f_new)/np.power( phi, 2 )
        
        if comparison >= 0:
            if f_new < ERROR_LIMIT: 
                break # done!
        
            vector      = vector_new
            f_old       = f_new
            r_new       = -network.gradient( vector, training_data, training_targets, cost_function )
        
            success     = True
            lamb_       = 0
        
            if k % network.n_weights == 0:
                grad_new = r_new
            else:
                beta    = (np.dot( r_new, r_new ) - np.dot( r_new, r ))/phi
                grad_new = r_new + beta * grad
        
            if comparison > 0.75:
                lamb    = 0.5 * lamb
        else:
            lamb_       = lamb
        # end 
    
        if comparison < 0.25: 
            lamb        = 4 * lamb
    
        if k%print_rate==0:
            print "[training] Current error:", f_new, "\tEpoch:", k
    #end
    
    network.set_weights( np.array(vector_new) )
    
    print "[training] Finished:"
    print "[training]   Converged to error bound (%.4g) with error %.4g." % ( ERROR_LIMIT, f_new )
    print "[training]   Measured quality: %.4g" % network.measure_quality( training_data, training_targets, cost_function )
    print "[training]   Trained for %d epochs." % k
    
    
    if save_trained_network and confirm( promt = "Do you wish to store the trained network?" ):
        network.save_network_to_file()
#end scg


## NOT YET IMPLEMENTED
#def generalized_hebbian(network, trainingset, testset, cost_function, learning_rate = 0.001, max_iterations = (), save_trained_network = False ):
#    assert softmax_function != network.layers[-1][1] or cost_function == softmax_neg_loss,\
#        "When using the `softmax` activation function, the cost function MUST be `softmax_neg_loss`."
#    assert cost_function != softmax_neg_loss or softmax_function == network.layers[-1][1],\
#        "When using the `softmax_neg_loss` cost function, the activation function in the final layer MUST be `softmax`."
#        
#    assert trainingset[0].features.shape[0] == network.n_inputs, \
#        "ERROR: input size varies from the defined input setting"
#    assert trainingset[0].targets.shape[0]  == network.layers[-1][0], \
#        "ERROR: output size varies from the defined output setting"
#    
#    training_data              = np.array( [instance.features for instance in trainingset ] )
#    training_targets           = np.array( [instance.targets  for instance in trainingset ] )
#                                
#    layer_indexes               = range( len(network.layers) )
#    epoch                       = 0
#                                
#    input_signals, derivatives  = network.update( training_data, trace=True )
#    error                       = cost_function(input_signals[-1], training_targets )
#    input_signals[-1]          -= training_targets
#    
#    while error > 0.01 and epoch < max_iterations:
#        epoch += 1
#        
#        for i in layer_indexes:
#            forgetting_term     = np.dot(network.weights[i], np.tril(np.dot( input_signals[i+1].T, input_signals[i+1] )))
#            activation_product  = np.dot(add_bias(input_signals[i]).T, input_signals[i+1])
#            dW                  = learning_rate * (activation_product - forgetting_term)
#            network.weights[i] += dW
#            
#            # normalize the weight to prevent the weights from growing unbounded
#            #network.weights[i]     /= np.sqrt(np.sum(network.weights[i]**2))
#        #end weight adjustment loop
#        
#        input_signals, derivatives  = network.update( training_data, trace=True )
#        error                       = cost_function(input_signals[-1], training_targets )
#        input_signals[-1]          -= training_targets
#        
#        if epoch % 1000 == 0:
#            print "[training] Error:", error
#    
#    print "[training] Finished:"
#    print "[training]   Trained for %d epochs." % epoch
#    
#    if save_trained_network and confirm( promt = "Do you wish to store the trained network?" ):
#        network.save_network_to_file()
## end hebbian