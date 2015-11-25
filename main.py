import scipy.stats
import sys
import time

import lasagne
import numpy as np
import numpy.matlib
import theano
import theano.tensor as T
from RiDictionary import RiDictionary
from sklearn.cross_validation import train_test_split

from AttentionRILayer import AttentionRILayer
from dataset import PL_sentiment as PL

d = 2000
k = 10
batchsize = 1

# Open the dictionary to load context vectors formsyste
path = "/media/tobiasnorlund/ac861917-9ad7-4905-93e9-ee6ab16360ad/bigdata/Dump/Wikipedia-3000000-2000-10"
w2vpath = "/home/tobiasnorlund/Code/CNN_sentence/GoogleNews-vectors-negative300.bin"
thetas_path = "SimLex-thetas-10.pkl"
dictionary = RiDictionary(path)# RandomDictionary(path) #RiDictionary(path) #W2vDictionary(w2vpath)

# Load a dataset to train and validate on
(input_docs, Y) = PL.load_dataset()
(input_docs_train, input_docs_test, Y_train, Y_test) = train_test_split(input_docs, Y, test_size=0.33, random_state=42)

# theta idx mapper
theta_idx_map = {}
i = 0
for entry in input_docs:
    splitted = entry.split()
    for word in splitted:
        if dictionary.get_context(word) is not None and word not in theta_idx_map:
            theta_idx_map[word] = i
            i += 1
            sys.stdout.write("\r"+str(i*100/17030))

# Build network
contexts = T.ftensor3('contexts')
theta_idxs = T.ivector('theta_idxs')
idx = T.ivector('idx')
default_theta = np.array([scipy.stats.norm.pdf(x, scale=3) for x in range(-10, 11) if x!=0], dtype="float32")
thetas = np.matlib.repmat(default_theta/sum(default_theta)*2*k,i,1)#np.ones((len(theta_idx_map), 2*k), dtype="float32")

target_var = T.iscalar('targets')

# Pre-load thetas
# import pickle
# (pretrained_thetas_idx_map, pretrained_thetas) = pickle.load(open(thetas_path))
# for (word, pt_idx) in pretrained_thetas_idx_map.iteritems():
#     if word in theta_idx_map:
#         thetas[theta_idx_map[word],:] = pretrained_thetas[pt_idx,:]

#l_in = lasagne.layers.InputLayer((1,d), contexts)
#l_ri = SumLayer(l_in)
l_in = lasagne.layers.InputLayer((2*k,d,None), contexts)
l_ri = AttentionRILayer(l_in, theta_idxs, idx, k, "AttentionLayer", theta_const=False, num_thetas=i, default_theta=None)
l_hid = lasagne.layers.DenseLayer(l_ri, num_units=120, nonlinearity=lasagne.nonlinearities.sigmoid)
l_out = lasagne.layers.DenseLayer(l_hid, num_units=PL.num_classes(), nonlinearity=lasagne.nonlinearities.sigmoid)

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
        loss, params, learning_rate=0.01) #, momentum=0.9)

# Create an expression for the classification accuracy:
test_acc = T.mean(T.eq(T.round(prediction), target_var),
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
                X[:,:, j] = ctx / np.sqrt(np.sum(np.sum(ctx, 0)**2))
                j += 1
        yield X[:,:,0:j], Y[excerpt], input_doc[excerpt]

# Perform training
num_epochs = 30
print("Starting training...")
for epoch in range(num_epochs):

    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_acc = 0
    train_count = 0
    start_time = time.time()
    for X, y, sentence in iterate_training_data(input_docs_train, Y_train, shuffle=False):
        # y_arr = np.array([1, 0], dtype="int32") if y==0 else np.array([0, 1], dtype="int32")
        if X.shape[2] == 0: continue
        idxs = [theta_idx_map[word] for word in sentence.split(" ") if word in theta_idx_map]
        err, acc = train_fn(X, idxs, np.arange(X.shape[2], dtype="int32"), y)
        train_err += err
        train_acc += acc
        train_count += 1
        #print sentence
        #print "y = " + str(y)
        sys.stdout.write("\r" + "total train acc: \t{:.2f}".format(train_acc * 100 / train_count))

    # And a full pass over the training data again:
    val_err = 0
    val_acc = 0
    val_count = 0
    for X, y, s in iterate_training_data(input_docs_test, Y_test, shuffle=False):
        #y_arr = np.array([1, 0], dtype="int32") if y==0 else np.array([0, 1], dtype="int32")
        if X.shape[2] == 0: continue
        val_err, acc = val_fn(X, [theta_idx_map[word] for word in s.split(" ") if word in theta_idx_map], np.arange(X.shape[2], dtype="int32"), y)
        val_acc += acc
        val_count += 1

    # Then we print the results for this epoch:
    sys.stdout.write("\r" + "Epoch {} of {} took {:.3f}s \n".format(
        epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_count))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_count))
    print("  training accuracy:\t\t{:.2f} %".format( train_acc / train_count * 100))
    print("  validation accuracy:\t\t{:.2f} %".format( val_acc / val_count * 100))
    print("  theta: " + str(l_ri.thetas.get_value()))


import pickle
pickle.dump((theta_idx_map, l_ri.thetas.get_value()), open("thetas.pkl", "wb"))