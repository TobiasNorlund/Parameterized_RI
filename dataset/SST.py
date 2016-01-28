import dataset
import math
import random

class SST(object):

    def __init__(self, folder_path, fine_grained_classes=False, clean_string=True):
        self.folder_path = folder_path
        self.fine_grained_classes=fine_grained_classes
        self.clean_string = clean_string

    def load(self):

        """
        Loads the Stanford Sentiment Treebank dataset. Uses the sentences from the "sentences_clean.txt"

        :return: Tuple: (X, Y) where
            X = [ "The cat sat on the mat",
                  "Anther cat was also sitting on the mat",
                  ... ]

            Y = [ 0, 1, 1, 1, 0, 1, 0, ... ]
        """

        f = open(self.folder_path + "/sst_cleaned.txt")
        f_splits = open(self.folder_path + "/sst_splits.txt")
        f_splits.readline()
        train = []
        val = []
        test = []

        X = []
        Y = []
        idx = 0
        for line in f:
            splitted = line.split("\t")
            split_split = f_splits.readline().strip().split(",")

            if self.clean_string:
                splitted[0] = dataset.clean_str(splitted[0].strip())

            if self.fine_grained_classes:
                X.append(splitted[0])
                Y.append(int(math.floor(float(splitted[1])*5)))
            else:
                if float(splitted[1]) <= 0.4:
                    X.append(splitted[0])
                    Y.append(0)
                elif float(splitted[1]) > 0.6:
                    X.append(splitted[0])
                    Y.append(1)
                else: continue

                if split_split[1] == "1":
                    train.append(idx)
                elif split_split[1] == "2":
                    test.append(idx)
                elif split_split[1] == "3":
                    val.append(idx)
                idx += 1

        random.shuffle(train)
        random.shuffle(val)

        self.splits = ([(train, val)], test)

        return (X, Y)

    @property
    def num_classes(self):
        return 1 if not self.fine_grained_classes else 5
        # TODO: Should be 2 instead of 1

    def get_splits(self):

        return self.splits