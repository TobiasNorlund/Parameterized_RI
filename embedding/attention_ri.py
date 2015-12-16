import numpy as np
import theano
import theano.tensor as T
from dictionary import RiDictionary

class AttentionRiEmbedding(object):
    
    def __init__(self, dict_path, words_to_load=None, normalize=True):
        self.dictionary = RiDictionary(dict_path, words_to_include=words_to_load, normalize=normalize)

        # Init thetas to ones
        self.thetas = np.ones((self.dictionary.n+1, self.dictionary.k*2), dtype="float32") # +1 for the zero vector

        # Create theno variables
        self.contexts_var = T.ftensor3("contexts")
        self.thetas_var = theano.shared(self.thetas, "thetas")
        self.theta_idxs_var = T.ivector("theta_idxs")
        self.idx_var = T.ivector("idx")

    @property
    def d(self):
        return self.dictionary.d

    def get_embeddings_expr(self):
        res, upd = theano.scan(lambda i: T.dot(self.thetas_var[self.theta_idxs_var[i],:],self.contexts_var[:,:,i]), self.idx_var)
        return res

    def get_embeddings(self, words):
        # TODO: Implement
        pass

    def has(self, word):
        return self.dictionary.has(word)

    def get_variable_vars(self):
        return [self.contexts_var, self.theta_idxs_var, self.idx_var]

    def get_variables(self, words):
        contexts = np.empty((2*self.dictionary.k, self.d, len(words)), dtype="float32")
        theta_idxs = []
        idx = []
        i = 0
        for word in words:
            if word == "##zero##":
                contexts[:,:,i] = np.zeros((2*self.dictionary.k, self.d), dtype="float32")
                theta_idxs.append(self.dictionary.n)
                idx.append(i)
                i += 1
            else:
                context = self.dictionary.get_context(word)
                if context is not None:
                    contexts[:,:,i] = context
                    theta_idxs.append(self.dictionary.get_word_meta(word).dict_idx)
                    idx.append(i)
                    i += 1

        return [contexts[:,:,0:i], theta_idxs, idx]

    def get_update_parameter_vars(self):
        return [self.thetas_var]
        
    def reset(self):
        self.thetas_var.set_value(np.ones((self.dictionary.n+1, self.dictionary.k*2), dtype="float32"))
