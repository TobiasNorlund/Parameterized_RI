
"""

  INITIALIZATION OF 'dataset' PACKAGE

  Dataset interface :

    @property
    num_classes          : The number of classes in this dataset

    load()               : Returns a tuple (X, Y) where X is list of input documents, and Y list of corresponding labels

"""

from PL05 import PL05

## -- Define helper functions -------------------

def get_all_words(docs):
    """
    Returns a list of all unique words within a list of documents

    Parameters
    ----------
    docs : list of strings

    """

    words = []
    for doc in docs:
        for word in doc.split(" "):
            if word not in words:
                words.append(word)
    return words