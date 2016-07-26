from ..tools import confirm
from ..activation_functions import softmax_function
from ..cost_functions import softmax_neg_loss
import numpy as np



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