__author__ = 'tobiasnorlund'

from collections import OrderedDict
import numpy as np
import theano
import theano.tensor as T
import lasagne

class AttentionRILayer(lasagne.layers.Layer):
    """
    AttentionRILayer()

    A layer that takes Random Indexing context matrices and weights the index vectors to context vectors.

    Parameters
    ----------
    ctxs_layer: InputLayer
        The InputLayer which represents the context matrices for the words in the document

    parameter_mode: one of "global", "const"
       How to parameterize this layer.
        - "global" = a global theta vector for all words
        - "const" = a constant one vector
    name : a string of None
       An optional name to attach to this layer.

    """
    def __init__(self, ctxs_layer, theta_idxs, idx, k, name=None, theta_const=True, num_thetas = 1):
        assert isinstance(ctxs_layer, lasagne.layers.InputLayer)

        self.input_shape = ctxs_layer.output_shape
        self.input_layer = ctxs_layer
        self.name = name
        self.params = OrderedDict()

        self.theta_idxs = theta_idxs
        self.idx = idx

        if(theta_const):
            self.thetas = self.add_param(theano.shared(np.ones((1, 2*k), dtype="float32"), name="thetas"), (1, 2*k), trainable=False)
        else:
            self.thetas = self.add_param(theano.shared(np.ones((num_thetas, 2*k), dtype="float32"), name="thetas" ), (num_thetas, 2*k))


    def get_output_for(self, ctxs, **kwargs):
        """

        :param ctxs:
            Variable containing the context matrices
        :param kwargs:
        :return:
        """

        #ctxs = theano.printing.Print('in: ')(ctxs)

        def calc_scalar_prod(i):
            return T.dot(self.thetas[self.theta_idxs[i],:],ctxs[:, :, i])

        results, updates = theano.scan(calc_scalar_prod, self.idx)
        out = results.flatten()
        #out = T.sum(results, axis=0)
        #out = theano.printing.Print('attention out: ')(out)
        return out


    def get_output_shape_for(self, input_shape):
        """
        Gets the output shape of this layer

        :param input_shape:
        :return:
        """
        if len(input_shape) == 3: # (2*k, d, ? )
            return (1, input_shape[1]*2) # (d)
        else:
            raise ValueError("Input shape to AttentionRILayer must be (win size, context_vector_dim, ?)")