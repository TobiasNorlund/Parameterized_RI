__author__ = 'tobiasnorlund'

import numpy as np

def load_dataset():

    """
    Loads the SimLex dataset
    """

    f = open("/home/tobiasnorlund/Code/Datasets/SimLex/simlex.txt")

    X = []
    Y = []
    for line in f:
        line_split = line.rstrip('\n').split()
        X.append(line_split[0] + " " + line_split[1])
        Y.append(float(line_split[2])/10)

    return (X,Y)

def num_classes():
    return 1


def train_thetas():

    from dictionary import RiDictionary
    from numpy.linalg import norm
    d = 2000
    k = 10

    # Open the dictionary to load context vectors form
    path = "/media/tobiasnorlund/ac861917-9ad7-4905-93e9-ee6ab16360ad/bigdata/Dump/Wikipedia-3000000-2000-10"
    dictionary = RiDictionary(path)

    # Load a dataset to train and validate on
    (input_docs, Y) = load_dataset()

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

    # Thetas
    thetas = np.ones((i, 2*k))


    def cos_ang(v1, v2):
        return np.dot(v1, v2) / (norm(v1)*norm(v2))

    def iterate_dataset(X, Y):
        for i in range(len(X)):
            yield (X[i], Y[i])

    import scipy.stats
    import scipy.spatial
    def calc_dist(word1, theta1, word2, theta2):
        word1 = dictionary.get_context(word1)
        word2 = dictionary.get_context(word2)

        vec1 = np.dot(np.atleast_2d(theta1), word1)[0,:]
        vec2 = np.dot(np.atleast_2d(theta2), word2)[0,:]

        return np.dot(vec1, vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2))

    #Train
    num_epochs = 500
    learn_rate = 1
    for epoch in range(num_epochs):

        aggr_err = 0

        # Loop through entire training dataset
        for (doc, y) in iterate_dataset(input_docs, Y):

            splitted = doc.split()
            word1 = splitted[0]
            word2 = splitted[1]

            # Get context matrices
            C1 = dictionary.get_context(word1)
            C2 = dictionary.get_context(word2)
            theta1 = thetas[theta_idx_map[word1], :]
            theta2 = thetas[theta_idx_map[word2], :]

            # Calc gradient
            y1 = np.dot(C1.T, theta1)
            y2 = np.dot(C2.T, theta2)
            ang = cos_ang(y1, y2)
            dJdf = ang - y
            dfdy1 = (y2*norm(y1) - np.dot(y1, y2)*y1/norm(y1))/(norm(y1)**2*norm(y2))
            dy1dth1 = C1
            dJdth1 = np.dot(dy1dth1, dfdy1) * dJdf

            dfdy2 = (y1*norm(y2) - np.dot(y2, y1)*y2/norm(y2))/(norm(y2)**2*norm(y1))
            dy2dth2 = C2
            dJdth2 = np.dot(dy2dth2, dfdy2) * dJdf

            thetas[theta_idx_map[word1], :] -= learn_rate * dJdth1
            thetas[theta_idx_map[word2], :] -= learn_rate * dJdth2

            aggr_err += abs(dJdf)

        print "Avg. error: " + str(aggr_err / len(input_docs))

        if epoch % 30 == 0:

            print "Spearman correlation (after):"
            dist = []
            for doc in input_docs:
                splitted = doc.split()
                dist.append(calc_dist(splitted[0], thetas[theta_idx_map[splitted[0]],:], splitted[1], thetas[theta_idx_map[splitted[1]],:]))
            print scipy.stats.spearmanr(dist, Y)

    for word, idx in theta_idx_map.iteritems():
        print word + ": " + str(thetas[idx,:])

    import pickle
    pickle.dump((theta_idx_map, thetas), open("SimLex-thetas-" + str(k) + ".pkl", mode="wb"))


if __name__ == "__main__":

    train_thetas()