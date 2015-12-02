import numpy as np
import theano
import theano.tensor as T

class DictionaryEmbedding(object):
    """
    The :class:'DictionaryEmbedding' class implements the Embedding interface using a dictionary of pre-loaded word vectors
    It also supports updating the embeddings through SGD by setting the enable_embedding_update parameter to the
    constructor.

    """

    def __init__(self, dictionary, enable_embedding_update=False):
        self.dictionary = dictionary
        (self.words, self.word_map) = dictionary.get_all_word_vectors()
        print "Total words: " + str(self.words.shape[0])

        # Theano variables
        self.words_var = theano.shared(self.words, "words")
        self.idxs_var = T.ivector("word_idxs")

        self.variables = [self.idxs_var]
        self.update_parameters = [self.words] if enable_embedding_update else []
        self.parameters = [self.words] if not enable_embedding_update else []

    @property
    def d(self):
        return self.dictionary.d


    def get_embeddings_expr(self):
        return self.words_var[self.idxs_var,:]

    def get_embeddings(self, words):

        """
        Returns an nd-array of the corresponding word embeddings

        Parameters
        ----------
        words : list of strings : The words

        Returns
        -------
        An nd-array of the corresponding word embeddings
        """

        # TODO: Implement

        # embs = np.empty((len(words), self.d), dtype="float32")
        # idxs = []
        # for word in words:
        #     if word in self.word_map:
        #         idxs.append(self.word_map[word])
        #     if emb is not None:
        #         embs[i,:] = emb

    def has(self, word):
        return self.dictionary.has(word)

    def get_variable_vars(self):
        return self.variables

    def get_variables(self, words):
        idxs = []
        for word in words:
            if word in self.word_map:
                idxs.append(self.word_map[word])

        return [idxs]

    def get_update_parameter_vars(self):
        return self.update_parameters