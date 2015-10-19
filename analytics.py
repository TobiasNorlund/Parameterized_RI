
import pickle
import matplotlib.pyplot as plt
import numpy as np

(theta_idx_map, thetas) = pickle.load(open("SimLex-thetas-10.pkl", mode="rb"))

rel_x = [i for i in range(-10, 11) if i != 0]


def plot_thetas():
    for (word, idx) in theta_idx_map.iteritems():
        plt.title(word)
        plt.ylim(-1, 4)
        plt.plot(rel_x, thetas[idx,:])

        plt.draw()
        plt.waitforbuttonpress()
        plt.clf()

def cluster_thetas():

    from sklearn.mixture import GMM

    k = 7

    gmm = GMM(k)
    gmm.fit(thetas)
    res = gmm.predict(thetas)

    clusters = [[] for i in range(k)]
    clusters_idx = [[] for i in range(k)]

    words = theta_idx_map.keys()
    idxs = theta_idx_map.values()
    for i in range(thetas.shape[0]):
        clusters[res[i]].append(words[i])
        clusters_idx[res[i]].append(idxs[i])


    for i in range(k):
        print clusters[i]

    for i in range(k):
        plt.title(i)
        plt.ylim(-1, 4)
        plt.plot(rel_x, np.mean(thetas[clusters_idx[i],:], axis=0))

        plt.draw()
        plt.waitforbuttonpress()
        plt.clf()


if __name__ == "__main__":

    plot_thetas()