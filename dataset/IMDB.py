import os
import os.path
import pickle

from random import shuffle
from sklearn.cross_validation import KFold

class IMDB(object):

    def __init__(self, imdb_path):
        self.imdb_path = imdb_path

    def load(self):
        """
        Loads the IMDB sentiment dataset.

        :return: Tuple: (X, Y) where
            X = [ "The cat sat on the mat",
                  "Anther cat was also sitting on the mat",
                  ... ]

            Y = [ 0, 1, 1, 1, 0, 1, 0, ... ]
        """

        # Temporary for faster loading
        #if os.path.isfile(self.imdb_path + "/dataset.pkl"):
        #    return pickle.load(self.imdb_path + "/dataset.pkl")

        X = []
        Y = []

        def load_dir(dir, label):
            files = os.listdir(dir)
            for file in files:
                with open(dir + file,'r') as f:
                    text = f.read()
                    X.append(text)
                    Y.append(label)

        print "Loading imdb dataset (1/4) ..."
        load_dir(self.imdb_path + "/train/pos/", 1)   # Read train pos
        print "Loading imdb dataset (2/4) ..."
        load_dir(self.imdb_path + "/train/neg/", 0)
        print "Loading imdb dataset (3/4) ..."

        self.no_train = len(X)

        load_dir(self.imdb_path + "/test/pos/", 1)
        print "Loading imdb dataset (4/4) ..."
        load_dir(self.imdb_path + "/test/neg/", 0)
        print "Loaded imdb dataset!"

        self.n = len(X)

        pickle.dump((X,Y), open(self.imdb_path + "/dataset.pkl", "wb"))

        return (X,Y)

    @property
    def num_classes(self):
        return 1

    def get_splits(self):
        """

        Returns
        -------
        ( [fold1, fold2, ....] , test )
        where foldi = (train, val)
        and train, val and test are lists of indexes
        """

        if os.path.isfile(self.imdb_path + "/imdb.splits"):
            return pickle.load(open(self.imdb_path + "/imdb.splits"))
        else:

            idxs = range(self.n)

            train_val = idxs[:self.no_train]
            test = idxs[self.no_train:]

            kf = KFold(len(train_val), n_folds=5, shuffle=True)

            folds = []
            for train_idxs, val_idxs in kf:
                folds.append(([train_val[i] for i in train_idxs], [train_val[i] for i in val_idxs]))

            splits = (folds, test)

            pickle.dump(splits, open(self.imdb_path + "/imdb.splits", "wb"))

            return splits
