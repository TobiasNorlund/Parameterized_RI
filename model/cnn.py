import sys
sys.path.insert(0, 'CNN_sentence/')
import conv_net_sentence
from random import shuffle

class CNN(object):

    def __init__(self, n_epochs=25):
        self.n_epochs = n_epochs

    def evaluate(self, embedding, dataset):
        """

        Evaluates the 'embedding' using a convolutional neural network for NLP (from Yoon Kim [2014]) on 'dataset'

        Parameters
        ----------
        embedding : An embedding which implements the Embedding interface
        dataset   : A dataset which implements the Dataset interface

        Returns   : A float, with the top accuracy achieved
        -------

        """

        # Load dataset
        (input_docs, Y) = dataset.load()
        ds = []
        longest_doc = 0
        for i in range(len(input_docs)):
            ds.append((input_docs[i],  Y[i]))
            l = len(input_docs[i].split(" "))
            if l>longest_doc: longest_doc = l
        shuffle(ds)

        test_split = 0.33
        test_set = ds[:int(test_split*len(ds))]
        train_set = ds[int(test_split*len(ds)):]

        # Train CNN
        perf = conv_net_sentence.train_conv_net(datasets=(train_set, test_set),
                                                 embedding=embedding,
                                                 longest_doc=longest_doc,
                                                 lr_decay=0.95,
                                                 filter_hs=[3,4,5],
                                                 conv_non_linear="relu",
                                                 hidden_units=[100,2],
                                                 shuffle_batch=True,
                                                 n_epochs=self.n_epochs,
                                                 sqr_norm_lim=9,
                                                 non_static=False,
                                                 batch_size=50,
                                                 dropout_rate=[0.5])
        print "perf: " + str(perf)