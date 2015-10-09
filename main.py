import numpy as np
import theano
import lasagne
import theano.tensor as T
from AttentionRILayer import AttentionRILayer
from RiDictionary import RiDictionary

import PL_sentiment as PL

d = 2000
k = 2
batchsize = 1

# Open the dictionary to load context vectors form
path = "/media/tobiasnorlund/ac861917-9ad7-4905-93e9-ee6ab16360ad/bigdata/Dump/Wikipedia-3000000-2000-2"
dictionary = RiDictionary(path)

# Load a dataset to train and validate on
(X, Y) = PL.load_dataset()

# Build network
input_var = T.tensor4('input')
target_var = T.ivector('targets')

l_in = lasagne.layers.InputLayer((batchsize,2*k,d,None), input_var)
l_ri = AttentionRILayer(l_in, (1,d,2*k), parameter_mode="const")
l_hid = lasagne.layers.DenseLayer(l_in, num_units=120, nonlinearity=lasagne.nonlinearities.rectify)
l_out = lasagne.layers.DenseLayer(l_hid, num_units=PL.num_classes(), nonlinearity=lasagne.nonlinearities.softmax)

# Helper function for iterating mini batches
def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

# Create a loss expression for training, i.e., a scalar objective we want
# to minimize (for our multi-class problem, it is the cross-entropy loss):
prediction = lasagne.layers.get_output(l_out)
loss = lasagne.objectives.binary_crossentropy(prediction, target_var) # logistic regression
loss = loss.mean()