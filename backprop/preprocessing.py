import numpy as np
import copy

all = ["construct_preprocessor", "standarize", "replace_nan", "whiten", "pca"]


def construct_preprocessor( trainingset, list_of_processors ):
    combined_processor = lambda x: x
    
    for processor in list_of_processors:
        combined_processor = processor( combined_processor( trainingset ))
    
    return combined_processor
#end


def standarize( trainingset ):
    """
    Morph the input signal to a mean of 0 and scale the signal strength by 
    dividing with the standard deviation (rather that forcing a [0, 1] range)
    """
    
    def encoder( dataset ):
        manipulated_dataset = copy.deepcopy( dataset )
        for instance in manipulated_dataset:
            instance.features = (instance.features - means) / stds
        return manipulated_dataset
    #end
    
    training_data = np.array( [instance.features for instance in trainingset ] )
    means = training_data.mean(axis=0)
    stds = training_data.std(axis=0)
    
    return encoder
#end


def replace_nan( trainingset, replace_with = None ):
    """
    Replace instanced of "not a number" with either the mean of the signal feature
    or a specific value assigned by `replace_with`
    """
    training_data = np.array( [instance.features for instance in trainingset ] ).astype( np.float64 )
    
    def encoder( dataset ):
        manipulated_dataset = copy.deepcopy( dataset )
        for instance in manipulated_dataset:
            instance.features = instance.features.astype( np.float64 )
            
            if np.sum(np.isnan( instance.features )):
                if replace_with == None:
                    instance.features[ np.isnan( instance.features ) ] = means[ np.isnan( instance.features ) ]
                else:
                    instance.features[ np.isnan( instance.features ) ] = replace_with
        return manipulated_dataset
    #end
    
    if replace_with == None:
        means = np.mean( np.nan_to_num(training_data), axis=0 )
    
    return encoder
#end


def whiten( trainingset, epsilon = 1e-5 ):
    training_data = np.array( [instance.features for instance in trainingset ] )
    
    def encoder(dataset):
        manipulated_dataset = copy.deepcopy( dataset )
        for instance in manipulated_dataset:
            instance.features = np.dot(instance.features, W)
        return manipulated_dataset
    #end
    
    covariance = np.dot(training_data.T, training_data)
    d, V = np.linalg.eig( covariance )
    
    D = np.diag(1. / np.sqrt(d+epsilon))
    
    W = np.dot(np.dot(V, D), V.T)
    
    return encoder
#end
