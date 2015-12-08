import sys
sys.path.insert(0, 'CNN_sentence/')
import conv_net_sentence

class CNN(object):

    def __init__(self, n_epochs=25):
        self.n_epochs = n_epochs

    def evaluate(self, embedding, train_data, validation_data, test_data, num_classes):
        """

        Evaluates the 'embedding' using a convolutional neural network for NLP (from Yoon Kim [2014]) on 'dataset'

        Parameters
        ----------
        embedding      :     An embedding which implements the Embedding interface
        train_data     ;     A tuple of lists (docs, y) that constitutes the training data
        validation_data:     A tuple of lists (docs, y) that constitutes the validation data

        Returns        :     A float, with the top accuracy achieved
        -------

        """

        # Load dataset
        train_set = zip(*train_data)
        validation_set = zip(*validation_data)
        test_set = zip(*test_data)

        longest_doc = 0
        for (doc, y) in train_set + validation_set + test_set:
            l = len(doc.split(" "))
            if l>longest_doc: longest_doc = l


        # Train CNN
        perf = conv_net_sentence.train_conv_net(datasets=(train_set, validation_set, test_set),
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

        return perf