from ...tools import confirm
from ...activation_functions import softmax_function
from ...cost_functions import softmax_neg_loss
import numpy as np


all = ["check_network_structure", "verify_dataset_shape_and_modify", "print_training_status", "print_training_results"]

def check_network_structure( network, cost_function ):
    assert softmax_function != network.layers[-1][1] or cost_function == softmax_neg_loss,\
        "When using the `softmax` activation function, the cost function MUST be `softmax_neg_loss`."
    assert cost_function != softmax_neg_loss or softmax_function == network.layers[-1][1],\
        "When using the `softmax_neg_loss` cost function, the activation function in the final layer MUST be `softmax`."
#end



def verify_dataset_shape_and_modify( network, dataset ):   
    assert dataset[0].features.shape[0] == network.n_inputs, \
        "ERROR: input size varies from the defined input setting"
    assert dataset[0].targets.shape[0]  == network.layers[-1][0], \
        "ERROR: output size varies from the defined output setting"
    
    data              = np.array( [instance.features for instance in dataset ] )
    targets           = np.array( [instance.targets  for instance in dataset ] )
    
    return data, targets 
#end


def apply_regularizers( dataset, cost_function, regularizers, network ):
    dW_regularizer = lambda x: np.zeros( shape = x.shape )
    
    if regularizers != None:
        # Modify the cost function to add the regularizer
        for entry in regularizers:
            if type(entry) == tuple:
                regularizer, regularizer_settings = entry
                cost_function, dW_regularizer  = regularizer( dataset, cost_function, dW_regularizer, network, **regularizer_settings )
            else:
                regularizer    = entry
                cost_function, dW_regularizer  = regularizer( dataset, cost_function, dW_regularizer, network )
    
    return cost_function, dW_regularizer
#end