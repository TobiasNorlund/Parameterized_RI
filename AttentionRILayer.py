__author__ = 'tobiasnorlund'

from collections import OrderedDict
import numpy as np
import theano
import theano.tensor as T
import lasagne

class AttentionRILayer(lasagne.layers.MergeLayer):
    """
    AttentionRILayer()

    A layer that takes Random Indexing context matrices and weights the index vectors to context vectors.

    Parameters
    ----------
    incomings : a length-two list of [ InputLayer , (batchsize) ]
        The InputLayer which represents the context matrices for the words in the documents for each batch
        A tuple of the batchsize, which represents the input vector of the theta indexes in the batch
    parameter_mode: one of "global", "const"
       How to parameterize this layer.
        - "global" = a global theta vector for all words
        - "const" = a constant one vector
    name : a string of None
       An optional name to attach to this layer.

    """
    def __init__(self, incomings, k, name=None, theta_const=True, num_thetas = 1):
        assert len(incomings) == 2

        super(lasagne.layers.MergeLayer, self).__init__(incomings, name)

        if(theta_const):
            self.theta = self.add_param(T.shared(np.ones(1, 2*k), name="theta"), (1, 2*k), trainable=False)
        else:
            self.theta = self.add_param(T.shared(np.ones(num_thetas, 2*k), name="theta" ), (num_thetas, 2*k))


    def get_output_for(self, inputs, **kwargs):
        """

        :param inputs: list of Theano expressions
            The Theano expressions to propagate through this layer.
        :param kwargs:
        :return:
        """

        assert len(inputs) == 2

        return T.sum(None, axis=0)


    def get_output_shape_for(self, input_shape):
        """
        Gets the output shape of this layer

        :param input_shape:
        :return:
        """
        if input_shape is tuple and len(input_shape) == 4: # (batch_size, 2*k, d, ? )
            return input_shape # (batch_size, d)
        else:
            raise ValueError("Input shape to AttentionRILayer must be (batch_size, context_vector_dim)")