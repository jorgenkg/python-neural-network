import numpy as np
import copy



def construct_preprocessor( trainingset, list_of_processors, **kwargs ):
    combined_processor = lambda x: x
    
    for entry in list_of_processors:
        if type(entry) == tuple:
            processor, processor_configuration = entry
            assert type(processor_configuration) == dict, \
                "The second argument to a preprocessor entry must be a dictionary of settings."
            combined_processor = processor( combined_processor( trainingset ), **processor_configuration )
        else:
            processor = entry
            combined_processor = processor( combined_processor( trainingset ))
    
    return lambda dataset: combined_processor( copy.deepcopy( dataset ))
#end


def standarize( trainingset ):
    """
    Morph the input signal to a mean of 0 and scale the signal strength by 
    dividing with the standard deviation (rather that forcing a [0, 1] range)
    """
    
    def encoder( dataset ):
        for instance in dataset:
            if np.any(stds == 0):
                nonzero_indexes = np.where(stds!=0)
                instance.features[nonzero_indexes] = (instance.features[nonzero_indexes] - means[nonzero_indexes]) / stds[nonzero_indexes]
            else:
                instance.features = (instance.features - means) / stds
        return dataset
    #end
    
    training_data = np.array( [instance.features for instance in trainingset ] )
    means = training_data.mean(axis=0)
    stds = training_data.std(axis=0)
    
    return encoder
#end


def replace_nan( trainingset, replace_with = None ): # if replace_with = None, replaces with mean value
    """
    Replace instanced of "not a number" with either the mean of the signal feature
    or a specific value assigned by `replace_nan_with`
    """
    training_data = np.array( [instance.features for instance in trainingset ] ).astype( np.float64 )
    
    def encoder( dataset ):
        for instance in dataset:
            instance.features = instance.features.astype( np.float64 )
            
            if np.sum(np.isnan( instance.features )):
                if replace_with == None:
                    instance.features[ np.isnan( instance.features ) ] = means[ np.isnan( instance.features ) ]
                else:
                    instance.features[ np.isnan( instance.features ) ] = replace_with
        return dataset
    #end
    
    if replace_nan_with == None:
        means = np.mean( np.nan_to_num(training_data), axis=0 )
    
    return encoder
#end


def subtract_mean( trainingset ):
    def encoder( dataset ):
        
        for instance in dataset:
            instance.features = instance.features - means
        return dataset
    #end
    
    training_data = np.array( [instance.features for instance in trainingset ] )
    means = training_data.mean(axis=0)
    
    return encoder
#end


def normalize( trainingset ):
    """
    Morph the input signal to a mean of 0 and scale the signal strength by 
    dividing with the standard deviation (rather that forcing a [0, 1] range)
    """
    
    def encoder( dataset ):
        for instance in dataset:
            if np.any(stds == 0):
                nonzero_indexes = np.where(stds!=0)
                instance.features[nonzero_indexes] = instance.features[nonzero_indexes] / stds[nonzero_indexes]
            else:
                instance.features = instance.features / stds
        return dataset
    #end
    
    training_data = np.array( [instance.features for instance in trainingset ] )
    stds = training_data.std(axis=0)
    
    return encoder
#end


def whiten( trainingset, epsilon = 1e-5 ):
    training_data = np.array( [instance.features for instance in trainingset ] )
    
    def encoder(dataset):
        
        for instance in dataset:
            instance.features = np.dot(instance.features, W)
        return dataset
    #end
    
    covariance = np.dot(training_data.T, training_data)
    d, V = np.linalg.eig( covariance )
    
    D = np.diag(1. / np.sqrt(d+epsilon))
    
    W = np.dot(np.dot(V, D), V.T)
    
    return encoder
#end
