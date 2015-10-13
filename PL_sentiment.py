__author__ = 'tobiasnorlund'

from RiDictionary import RiDictionary

def load_dataset():
    """
    Loads the Pang Lee 2005 sentiment dataset.

    :parameter: dictionary A RiDictionary to load word contexts from
    :return: Tuple: (X, Y) where
        X = [ "The cat sat on the mat",
              "Anther cat was also sitting on the mat",
              ... ]

        Y = [ 0, 1, 1, 1, 0, 1, 0, ... ]
    """

    path_prefix = "/home/tobiasnorlund/Code/Datasets/PL-05/rt-polaritydata/rt-polarity"

    X = []
    Y = []

    def load_file(f, y):
        i = 0
        for line in f:
            #if i > 100: break
            X.append(line.rstrip('\n'))
            Y.append(y)
            i += 1


    # Load positive samples
    f = open(path_prefix + ".pos")
    load_file(f, 1)
    f.close()

    # Load negative samples
    f = open(path_prefix + ".neg")
    load_file(f, 0)
    f.close()

    return (X, Y)


def num_classes():
    return 1