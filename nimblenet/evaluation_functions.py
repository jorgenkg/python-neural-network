import numpy as np

def binary_accuracy( outputs, targets, threshold = 0.7 ):
    # Calculate the binary accuracy between the output signals and the targets values.
    # The threshold value describes when a signal should switch from 
    # being interpreted as a zero to an active one.
    return 1 - np.count_nonzero((outputs > threshold) == targets) / float(targets.size)
#end

def categorical_accuracy( outputs, targets ):
    return 1.0 - 1.0 * np.count_nonzero(np.argmax(self.update( training_data ), axis=1) == np.argmax(training_targets, axis=1)) / training_data.shape[0]
#end
