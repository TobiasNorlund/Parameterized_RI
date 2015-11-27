
"""

  INITIALIZATION OF 'embedding' PACKAGE

  Embedding interface :

    @property
    d                 : Dimensionality of the word embeddings

    get_embeddings_expr()                : Get the theano expression for the embeddings
    get_embeddings([word1, ...])         : corresponding raw embedding values

    get_variable_vars()                  : list of theano variables needed for the embeddings
    get_variables([word1, ...])          : list of variable values

    get_parameter_vars()                 : list of Theano shared variables
    get_update_parameter_vars()          : list of Theano shared variables of update params (passed to "updates")


"""

from dictionary_embedding import DictionaryEmbedding

from dictionary import *