import dataset
import os.path
import pickle

from random import shuffle
from sklearn.cross_validation import KFold

class PL05(object):

    def __init__(self, path_prefix, clean_string=True):
        self.path_prefix = path_prefix
        self.clean_string = clean_string

    def load(self):
        """
        Loads the Pang Lee 2005 sentiment dataset.

        :return: Tuple: (X, Y) where
            X = [ "The cat sat on the mat",
                  "Anther cat was also sitting on the mat",
                  ... ]

            Y = [ 0, 1, 1, 1, 0, 1, 0, ... ]
        """
        X = []
        Y = []

        def load_file(f, y):
            for line in f:
                if self.clean_string:
                    line = dataset.clean_str(line.strip())
                X.append(line)
                Y.append(y)

        # Load positive samples
        f = open(self.path_prefix + ".pos")
        load_file(f, 1)
        f.close()

        # Load negative samples
        f = open(self.path_prefix + ".neg")
        load_file(f, 0)
        f.close()

        self.n = len(X)

        return (X, Y)

    @property
    def num_classes(self):
        return 1

    def get_splits(self):
        """

        Returns
        -------
        ( [fold1, fold2, ....] , test )
        where foldi = (train, val)
        and train, val and test are list of indexes
        """

        if os.path.isfile(self.path_prefix + ".splits"):
            return pickle.load(open(self.path_prefix + ".splits"))
        else:

            idxs = range(self.n)
            shuffle(idxs)

            train_val = idxs[:int(0.75*self.n)]
            test = idxs[int(0.75*self.n):]

            kf = KFold(len(train_val), n_folds=5, shuffle=True)

            folds = []
            for train_idxs, val_idxs in kf:
                folds.append(([train_val[i] for i in train_idxs], [train_val[i] for i in val_idxs]))

            splits = (folds, test)

            pickle.dump(splits, open(self.path_prefix + ".splits", "wb"))

            return splits