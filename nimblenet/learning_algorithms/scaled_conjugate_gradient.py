from ..tools import add_bias, confirm
from ..activation_functions import softmax_function
from ..cost_functions import softmax_neg_loss
import numpy as np
import math


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