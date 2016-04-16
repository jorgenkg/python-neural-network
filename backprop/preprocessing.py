import numpy as np

all = ["construct_preprocessor", "standarize", "replace_nan", "whiten"]


def construct_preprocessor( trainingset, list_of_processors ):
    combined_processor = lambda x: x
    
    for processor in list_of_processors:
        combined_processor = processor( combined_processor( trainingset ))
    
    return combined_processor
#end

def standarize( trainingset ):
    def encoder( dataset ):
        for instance in dataset:
            instance.features = (instance.features - means) / stds
        return dataset
    #end
    
    training_data = np.array( [instance.features for instance in trainingset ] )
    
    means = training_data.mean(axis=0, keepdims=True)
    stds = training_data.std(axis=0, keepdims=True)
    
    print means.shape
    print stds.shape
    return encoder
#end


def replace_nan( matrix, replace_with = None ):
    if np.sum(np.isnan( matrix ), axis=0).any() > 0:
        # Check if there are any NaN values in the data
        if replace_with == None:
            # Use the mean value
            mean = np.mean( matrix[matrix!=np.nan], axis=0 )
            matrix[matrix==np.nan] = mean
        else:
            matrix[matrix==np.nan] = replace_with
    return matrix
#end


def whiten( matrix, epsilon = 1e-5 ):
    def encoder(features):
        return np.dot(features, W)
    #end
    
    covariance = np.dot(matrix.T, matrix)
    d, V = np.linalg.eig( covariance )
    
    D = np.diag(1. / np.sqrt(d+epsilon))
    
    W = np.dot(np.dot(V, D), V.T)
    
    return encoder
#end
