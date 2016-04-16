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
    
    means = training_data.mean(axis=0)
    stds = training_data.std(axis=0)
    
    return encoder
#end


def replace_nan( trainingset, replace_with = None ):
    def encoder( dataset ):
        for instance in dataset:
            if np.isnan( instance.features ):
                if replace_with == None:
                    instance.features[ instance.features==np.nan ] = mean[ instance.features==np.nan ]
                else:
                    instance.features[ instance.features==np.nan ] = replace_with
        return dataset
    #end
    
    training_data = np.array( [instance.features for instance in trainingset ] )
    
    means = np.mean( training_data[training_data!=np.nan], axis=0 )
    
    return encoder
#end


def whiten( trainingset, epsilon = 1e-5 ):
    def encoder(dataset):
        for index, instance in enumerate(dataset):
            instance.features = np.dot(instance.features, W[index,:])
        return dataset
    #end
    
    training_data = np.array( [instance.features for instance in trainingset ] )
    
    print np.dot(training_data.T, training_data)
    sdpof
    covariance = np.dot(training_data.T, training_data)
    d, V = np.linalg.eig( covariance )
    
    D = np.diag(1. / np.sqrt(d+epsilon))
    
    W = np.dot(np.dot(V, D), V.T)
    
    return encoder
#end
