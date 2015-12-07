import dataset
import math

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

        f = open(self.folder_path + "/sentences_clean.txt")
        X = []
        Y = []

        for line in f:
            splitted = line.split("\t")
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

        return (X, Y)

    @property
    def num_classes(self):
        return 1 if not self.fine_grained_classes else 5
        # TODO: Should be 2 instead of 1