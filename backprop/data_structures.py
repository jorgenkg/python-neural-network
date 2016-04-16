from preprocessing import *

class Instance:
    # This is a simple encapsulation of a `input signal : output signal`
    # pair in our training set.
    def __init__(self, features, target):
        self.features = np.array(features)
        self.targets  = np.array(target)
#endclass Instance


class Dataset:
    def __init__(self, instance_list ):
        self.features = np.array( [instance.features for instance in instance_list ] )
        self.targets = np.array( [instance.targets for instance in instance_list ] )
    #end
    
    def preprocess(self, ):
        self._features = self.features
        
        decorrelate  = preprocessing_decorrelation( self.features )
        standardize  = preprocessing_standarize( decorrelate(self.features) )
        
        encoder = lambda matrix: standardize( decorrelate( matrix ))
        self.features = encoder( self.features )
        
        return encoder
    #end
#endclass Dataset
    