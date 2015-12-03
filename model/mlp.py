import sys
import model
import time
import numpy as np
import lasagne
import theano
import theano.tensor as T

class MLP(object):

    def __init__(self, num_epochs=100):
        self.num_epochs = num_epochs

    def evaluate(self, embedding, dataset):

        """

        Evaluates the 'embedding' using a neural network model on 'dataset'


        Parameters
        ----------
        embedding : An embedding which implements the Embedding interface
        dataset   : A dataset which implements the Dataset interface

        Returns   : A float, with the top accuracy achieved
        -------

        """

        # Load dataset
        (input_docs, Y) = dataset.load()
        (input_docs_train, input_docs_test, Y_train, Y_test) = model.train_test_split(input_docs, Y, test_size=0.33)

        # Fetch embeddings expression and represent the document as a sum of the words
        embeddings_var = embedding.get_embeddings_expr()
        doc_var = embeddings_var.sum(axis=0).dimshuffle('x',0)

        # Create theano symbolic variable for the target labels
        target_var = T.iscalar('target')

        # Build model using lasagne
        l_in = lasagne.layers.InputLayer((1, embedding.d), doc_var)
        l_hid = lasagne.layers.DenseLayer(l_in, num_units=120, nonlinearity=lasagne.nonlinearities.sigmoid)
        l_out = lasagne.layers.DenseLayer(l_hid, num_units=dataset.num_classes,
                                          nonlinearity=lasagne.nonlinearities.sigmoid)
        # TODO: support multiclass

        # Create a loss expression for training, i.e., a scalar objective we want
        # to minimize
        prediction = lasagne.layers.get_output(l_out)
        loss = lasagne.objectives.binary_crossentropy(prediction, target_var).mean()

        # Create update expression for training
        params = lasagne.layers.get_all_params(l_out, trainable=True) + embedding.get_update_parameter_vars()
        updates = lasagne.updates.sgd(loss, params, learning_rate=0.01)

        # Create an expression for the classification accuracy:
        test_acc = T.mean(T.eq(T.round(prediction), target_var), dtype=theano.config.floatX)

        # Compile a function performing a training step
        train_fn = theano.function([target_var] + embedding.get_variable_vars(), [loss, test_acc],
                                   updates=updates)

        # Compile a second function computing the validation loss and accuracy:
        val_fn = theano.function([target_var] + embedding.get_variable_vars(), [loss, test_acc])

        # Helper function for iterating the training set
        def iterate_training_data(input_docs, Y, shuffle=True):
            assert len(input_docs) == len(Y)
            if shuffle:
                indices = np.arange(len(input_docs))
                np.random.shuffle(indices)

            for i in range(len(input_docs)):
                excerpt = indices[i] if shuffle else i
                yield input_docs[excerpt], Y[excerpt]

        ## Perform the training
        print("Starting training...")
        for epoch in range(self.num_epochs):

            # Time it !
            start_time = time.time()

            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_acc = 0
            train_count = 0
            for doc, y in iterate_training_data(input_docs_train, Y_train, shuffle=False):

                words = doc.split(" ")
                if not any([embedding.has(word) for word in words]): continue # If no embeddings, skip this doc

                err, acc = train_fn(y, *embedding.get_variables(words))
                train_err += err
                train_acc += acc
                train_count += 1

                sys.stdout.write("\r" + "total train acc: \t{:.2f}".format(train_acc * 100 / train_count))

            # And a full pass over the validation data data again:
            val_err = 0
            val_acc = 0
            val_count = 0
            for doc, y in iterate_training_data(input_docs_test, Y_test, shuffle=False):

                words = doc.split(" ")
                if not any([embedding.has(word) for word in words]): continue # If no embeddings, skip this doc

                val_err, acc = val_fn(y, *embedding.get_variables(words))
                val_acc += acc
                val_count += 1

            # Then we print the results for this epoch:
            sys.stdout.write("\r" + "Epoch {} of {} took {:.3f}s \n".format(
                epoch + 1, self.num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_count))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_count))
            print("  training accuracy:\t\t{:.2f} %".format( train_acc / train_count * 100))
            print("  validation accuracy:\t\t{:.2f} %".format( val_acc / val_count * 100))

