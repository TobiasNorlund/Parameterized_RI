
"""

  INITIALIZATION OF 'model' PACKAGE

  Model interface :

    evaluate(embedding, dataset)    : Evaluate the embeddings on the dataset using this model

"""

from mlp import MLP
from cnn import CNN

# ------ Define helper functiona --------

def train_test_split(X,Y, test_size):
    assert len(X) == len(Y)

    from random import shuffle
    n = len(X)

    index_shuf = range(n)
    shuffle(index_shuf)

    X_train = []
    X_test = []
    Y_train = []
    Y_test = []

    j = 0
    train_size = int((1-test_size)*n)
    for i in index_shuf:

        if j < train_size:
            X_train.append(X[i])
            Y_train.append(Y[i])
        else:
            X_test.append(X[i])
            Y_test.append(Y[i])

        j += 1

    return (X_train, X_test, Y_train, Y_test)