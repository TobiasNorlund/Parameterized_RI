import sys
import numpy as np
import theano
import lasagne
import theano.tensor as T
import time
from AttentionRILayer import AttentionRILayer
from sklearn.preprocessing import normalize
from dictionary import RiDictionary, RandomDictionary, W2vDictionary
from sklearn.cross_validation import train_test_split

import SimLex as PL

d = 2000
k = 10

# Open the dictionary to load context vectors form
path = "/media/tobiasnorlund/ac861917-9ad7-4905-93e9-ee6ab16360ad/bigdata/Dump/Wikipedia-3000000-2000-10"
w2vpath = "/home/tobiasnorlund/Code/CNN_sentence/GoogleNews-vectors-negative300.bin"
dictionary = RiDictionary(path) #W2vDictionary(w2vpath)RandomDictionary(path)

# Load a dataset to train and validate on
(input_docs, Y) = PL.load_dataset()

# theta idx mapper
theta_idx_map = {}
i = 0
for entry in input_docs:
    splitted = entry.split()
    if splitted[0] not in theta_idx_map:
        theta_idx_map[splitted[0]] = i
        i += 1
    if splitted[1] not in theta_idx_map:
        theta_idx_map[splitted[1]] = i
        i += 1

# Build network
contexts = T.ftensor3('contexts')
theta_idxs = T.ivector('theta_idxs')
idx = T.ivector('idx')

target_var = T.fscalar('targets')

#l_in = lasagne.layers.InputLayer((1,d), contexts)
#l_ri = SumLayer(l_in)
l_in = lasagne.layers.InputLayer((2*k,d,2), contexts)
l_ri = AttentionRILayer(l_in, theta_idxs, idx, k, "AttentionLayer", theta_const=False, num_thetas=i)
#l_hid = lasagne.layers.DenseLayer(l_ri, num_units=120, nonlinearity=lasagne.nonlinearities.sigmoid)
l_out = lasagne.layers.DenseLayer(l_ri, num_units=PL.num_classes(), nonlinearity=lasagne.nonlinearities.sigmoid)

# Create a loss expression for training, i.e., a scalar objective we want
# to minimize (for our multi-class problem, it is the cross-entropy loss):
prediction = lasagne.layers.get_output(l_out)
#prediction = theano.printing.Print('prediction out: ')(prediction)
loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
#loss = theano.printing.Print('bin_cross out: ')(loss)
loss = loss.mean()

# Regularization
#loss += T.sum((l_ri.thetas-1)**2) * 1e-2

# Create update expressions for training, i.e., how to modify the
# parameters at each training step. Here, we'll use Stochastic Gradient
# Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
params = lasagne.layers.get_all_params(l_out, trainable=True)
updates = lasagne.updates.sgd(
        loss, params, learning_rate=0.5) #, momentum=0.9)

# Create an expression for the classification accuracy:
test_acc = T.mean(T.abs_(prediction - target_var),
                  dtype=theano.config.floatX)

# Compile a function performing a training step on a mini-batch (by giving
# the updates dictionary) and returning the corresponding training loss:
train_fn = theano.function([contexts, theta_idxs, idx, target_var], [loss, test_acc], updates=updates)

# Compile a second function computing the validation loss and accuracy:
val_fn = theano.function([contexts, theta_idxs, idx, target_var], [loss, test_acc])

# Helper function for iterating mini batches
def iterate_training_data(input_doc, Y, shuffle=True):
    assert len(input_doc) == len(Y)
    if shuffle:
        indices = np.arange(len(input_doc))
        np.random.shuffle(indices)
    for i in range(len(input_doc)):
        if shuffle:
            excerpt = indices[i]
        else:
            excerpt = i

        words = input_doc[excerpt].split(" ")
        X = np.empty((2*k, d, len(words)), dtype="float32")
        j = 0
        for word in words:
            ctx = dictionary.get_context(word)
            if ctx is not None:
                X[:,:, j] = ctx / np.linalg.norm(ctx)
                j += 1
        yield X[:,:,0:j], Y[excerpt], input_doc[excerpt]

# Calculate spearman
import scipy.stats
import scipy.spatial
def calc_dist(word1, theta1, word2, theta2):
    word1 = dictionary.get_context(word1)
    word2 = dictionary.get_context(word2)

    vec1 = np.dot(np.atleast_2d(theta1), word1)[0,:]
    vec2 = np.dot(np.atleast_2d(theta2), word2)[0,:]

    return np.dot(vec1, vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2))

print "Spearman correlation (before):"
dist = []
i = 0
for doc in input_docs:
    splitted = doc.split()
    cosdist = calc_dist(splitted[0], np.ones((1,2*k)), splitted[1], np.ones((1,2*k)))

    #print splitted[0] + "\t" + splitted[1] + "\t" + str(cosdist) + "\t" + str(Y[i])
    i += 1

    dist.append(cosdist)
print scipy.stats.spearmanr(dist, Y)


# Perform training
num_epochs = 1000
print("Starting training...")
for epoch in range(num_epochs):

    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_acc = 0
    train_accs = []
    train_count = 0
    start_time = time.time()
    for X, y, sentence in iterate_training_data(input_docs, Y, shuffle=False):

        err, acc = train_fn(X, [theta_idx_map[word] for word in sentence.split()], np.arange(X.shape[2], dtype="int32"), y)
        train_err += err
        train_acc += acc
        train_accs.append(acc)
        train_count += 1
        #print sentence
        #print "y = " + str(y)
        sys.stdout.write("\r" + "total train acc: \t{:.2f}".format(train_acc * 100 / train_count))

    # And a full pass over the training data again:
    # val_err = 0
    # val_acc = 0
    # val_count = 0
    # for X, y, s in iterate_training_data(input_docs_test, Y_test, shuffle=False):
    #     #y_arr = np.array([1, 0], dtype="int32") if y==0 else np.array([0, 1], dtype="int32")
    #     val_err, acc = val_fn(X, np.zeros(X.shape[2], dtype="int32"), np.arange(X.shape[2], dtype="int32"), y)
    #     val_acc += acc
    #     val_count += 1

    # Then we print the results for this epoch:
    sys.stdout.write("\r" + "Epoch {} of {} took {:.3f}s \n".format(
        epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_count))
    #print("  validation loss:\t\t{:.6f}".format(val_err / val_count))
    print("  training accuracy:\t\t{:.2f} %".format( train_acc / train_count * 100))
    #print("  validation accuracy:\t\t{:.2f} %".format( val_acc / val_count * 100))
    #print("  theta: " + str(l_ri.thetas.get_value()))

    if epoch % 30 == 0:
        # Print thetas
        thetas_cpy = l_ri.thetas.get_value()
        for word, idx in theta_idx_map.iteritems():
            print word + ": " + str(thetas_cpy[idx,:])


        print "Avg. abs distance:"
        print np.mean(np.abs(np.array(dist) - np.array(Y)))

        print "Spearman correlation (after):"
        dist = []
        for doc in input_docs:
            splitted = doc.split()
            dist.append(calc_dist(splitted[0], thetas_cpy[theta_idx_map[splitted[0]],:], splitted[1], thetas_cpy[theta_idx_map[splitted[1]],:]))
        print scipy.stats.spearmanr(dist, Y)