
"""

  INITIALIZATION OF 'dataset' PACKAGE

  Dataset interface :

    @property
    num_classes          : The number of classes in this dataset

    load()               : Returns a tuple (X, Y) where X is list of input documents, and Y list of corresponding labels

"""

from PL05 import PL05
from SST import SST
from IMDB import IMDB

## -- Define helper functions -------------------

def get_all_words(docs):
    """
    Returns a list of all unique words within a list of documents

    Parameters
    ----------
    docs : list of strings

    """

    words = set()
    for doc in docs:
        for word in doc.split(" "):
            if word not in words:
                words.add(word)
    return words

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    import re
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()