import numpy as np
import theano
import theano.tensor as T
from dictionary import RiDictionary

class AttentionRiEmbedding(object):
    
    def __init__(self, dict_path, words_to_load=None):
        self.dictionary = RiDictionary(dict_path, words_to_load=words_to_load)
        
        # Init thetas to ones
        self.thetas = np.ones((self.dictionary.n, self.d), dtype="float32")

        # Create theno variables
        self.contexts_var = T.ftensor3("contexts")
        self.thetas_var = theano.shared(self.thetas, "thetas")
        self.theta_idxs_var = T.ivector("theta_idxs")
        self.idx_var = T.ivector("idx")

    @property
    def d(self):
        return self.dictionary.d

    def get_embeddings_expr(self):
        res, upd = theano.scan(lambda i: T.dot(self.thetas_var[self.theta_idxs_var[i],:],self.contexts_var[:,:,i]), self.idx)
        return res

    def get_embeddings(self, words):
        # TODO: Implement
        pass

    def get_variable_vars(self):
        return [self.contexts_var, self.theta_idxs_var, self.idx_var]

    def get_variables(self, words):
        contexts = np.empty((2*self.dictionary.k, self.d, len(words)), dtype="float32")
        theta_idxs = []
        idx = []
        i = 0
        for word in words:
            context = self.dictionary.get_context(word) 
            if context is not None:
                contexts[:,:,i] = contexts
                theta_idxs.append(self.dictionary.get_word_meta(word).idx)
                idx.append(i)
                i += 1

        return [contexts[:,:,0:i], theta_idxs, idx]

    def get_parameter_vars(self):
        return []

    def get_update_parameter_vars(self):
        return [self.thetas_var]