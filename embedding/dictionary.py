import numpy as np, sys, struct
import pickle
import os.path
import h5py
import sklearn.preprocessing
from collections import OrderedDict, namedtuple
from scipy.sparse import csr_matrix, lil_matrix

"""
  Dictionary interface:

    n : The number of words in the dictionary
    d : The vector dimensionality

    __init__( ... ) :               Arbitrary args to init the dictionary

    has(word) :                     Returns True if word exists in dictionary, otherwise False
    get_word_vector(string word) :  Returns a d-dim vector if the word exists in the dictionary, else None
    get_all_word_vectors() :        Returns a (mtx, OrderedDict<word,idx>) tuple of all word vectors

    iter_words() :                  Returns a generator that iterates (word, vector) tuples

"""

__all__ = [
    "RiDictionary",
    "PmiRiDictionary",
    "W2vDictionary",
    "PyDsmDictionary",
    "GloVeDictionary",
    "Hdf5Dictionary", 
    "BowDictionary",
    "RandomDictionary",
    "StaticDictionary"
]

WordMeta = namedtuple("WordMeta", "idx dict_idx focus_count context_count")

class RiDictionary(object):

    """
     A Random Indexing Dictionary
    """

    def __init__(self, path, words_to_include=None, normalize=True):

        self.d = int(path.split("/")[-1].split("-")[2])
        self.k = int(path.split("/")[-1].split("-")[3])
        self.binary_len = np.dtype('float32').itemsize * self.d
        self.word_map = OrderedDict()
        self.normalize = normalize

        sys.stdout.write("Loading word meta data...")

        if words_to_include is not None: words_to_include = set(words_to_include) # Convert to set for speed
        idx = 0
        dict_idx = 0
        for line in open(path + ".map", 'r'):
            splitted = line.split("\t")
            if words_to_include is None or (words_to_include is not None and splitted[0] in words_to_include):
                self.word_map[splitted[0]] = WordMeta(idx, dict_idx, int(splitted[1]), int(splitted[2]))
                dict_idx += 1
            idx += 1

        self.n = len(self.word_map)
        if os.path.isfile(path + ".context.bin"):
            self.f_ctx = open(path + ".context.bin", mode="rb")
            self.cache = {}
        else:
            self.f_ctx = None
            print "WARNING: No context file was found"

        sys.stdout.write("\r")

    def has(self, word):
        return word in self.word_map

    def get_word_vector(self, word):
        ctx = self.get_context(word)
        if ctx is not None:
            return np.nan_to_num(ctx.sum(0))
        else:
            return None

    def get_word_meta(self, word):
        return self.word_map[word] if word in self.word_map else None

    def get_all_word_vectors(self):
        mtx = np.empty((self.n,self.d), dtype="float32")
        ordered_word_map = OrderedDict()
        i = 0
        for word in self.word_map.keys():
            mtx[i,:] = self.get_word_vector(word)
            ordered_word_map[word] = i
            i += 1
        return (mtx, ordered_word_map)

    def iter_words(self):
        for word in self.word_map:
            yield (word, self.get_word_vector(word))

    def get_context(self, word):
        if word in self.word_map and self.f_ctx is not None:
            if word not in self.cache:
                self.f_ctx.seek(self.word_map[word].idx * self.d * 2*self.k * np.dtype('float32').itemsize)
                ctx = np.reshape(np.fromstring(self.f_ctx.read(self.binary_len*2*self.k), dtype='float32'), newshape=( 2*self.k, self.d) )
                self.cache[word] = ctx if not self.normalize else ctx / np.sqrt(np.sum(np.sum(ctx, 0)**2))
            return self.cache[word]
        else:
            return None

class PmiRiDictionary(RiDictionary):

    """
     A Pointwise-mutal-information transformed Random Indexing Dictionary
    """

    def __init__(self, filepathprefix, epsilon, words_to_include=None, use_true_counts=False, normalize=False):
        super(PmiRiDictionary, self).__init__(filepathprefix, words_to_include, normalize=False)
        self.epsilon = epsilon
        self.normalize = normalize
        self.filepathprefix = filepathprefix
        self.cache = {}

        # Build sparse index vector matrix
        print "Loads index vectors..."
        if os.path.isfile(filepathprefix + ".index.pkl"):
            (self.R, self.context_counts) = pickle.load(open(filepathprefix + ".index.pkl"))
        else:
            f = open(filepathprefix + ".index.bin", mode="rb")
            f_map = open(filepathprefix + ".map")
            self.R = lil_matrix((self.n,self.d), dtype="int8")
            self.context_counts = np.empty(self.n, dtype="uint32")
            for i in range(self.n):
                for e in range(epsilon):
                    val = struct.unpack("h", f.read(2))[0]
                    idx = val >> 1
                    self.R[i,idx] = 1 if val % 2 == 1 else -1

                counts_str = f_map.readline().split("\t")
                self.context_counts[i] = int(counts_str[2]) if int(counts_str[2]) > 0 else 1

                sys.stdout.write("\r" + str(i))

            self.R = csr_matrix(self.R)
            pickle.dump((self.R, self.context_counts) , open(filepathprefix + ".index.pkl", mode="w"))
            f.close()
            f_map.close()

        # Load true counts if available and requested
        if use_true_counts and os.path.isfile(filepathprefix + ".counts"):
            self.f_count = open(filepathprefix + ".counts")

        self.sum_ctxs = np.sum(self.context_counts)
        print "Loaded!"

    def get_word_vector(self, word):

        if word not in self.cache:

            vec = super(PmiRiDictionary, self).get_word_vector(word)
            if vec is not None:
                if hasattr(self, "f_count"): # if we have the true counts
                    self.f_count.seek(self.n*4*self.word_map[word].idx)
                    bow = np.fromstring(self.f_count.read(self.n*4), dtype="uint32").astype("float32")
                else: # estimate the counts
                    bow = np.maximum(0, self.R.dot(vec) / self.epsilon)
                    bow = bow / (bow.sum() / self.word_map[word].context_count) # we know the total context counts. improve count estimates so that bow.sum() == true total count
                bow = np.log(bow * self.sum_ctxs / (self.word_map[word].focus_count*self.context_counts))
                bow[bow == -np.inf] = 0

                back_proj = self.R.T.dot(bow)
                self.cache[word] = back_proj if not self.normalize else back_proj / np.linalg.norm(back_proj)
            else:
                return None

        return self.cache[word]

class W2vDictionary(object):

    """
     A Dictionary that loads word embeddings from a binary word2vec dump
    """

    def __init__(self, path, words_to_load=None, normalize=True):

        f = open(path, mode="rb")
        header = f.readline().split(" ") # read header

        self.d = int(header[1])
        file_n = int(header[0])
        self.n = file_n if words_to_load is None else len(words_to_load)
        binary_len = np.dtype('float32').itemsize * self.d

        self.word_map = OrderedDict()
        self.word_vectors = np.empty((self.n,self.d), dtype="float32")

        idx = 0
        for line in xrange(file_n):
            word = []
            if line % 10000 == 0: sys.stdout.write("\rLoading w2v vectors... (" + str(line *100.0 / file_n) + "%)")
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)

            vec_raw = f.read(binary_len)
            if words_to_load is None or (words_to_load is not None and word in words_to_load):
                self.word_map[word] = idx
                self.word_vectors[idx,:] = np.fromstring(vec_raw, dtype='float32')
                if normalize:
                    self.word_vectors[idx,:] /= np.linalg.norm(self.word_vectors[idx,:])
                idx += 1

        self.word_vectors = self.word_vectors[:idx,:]
        self.n = idx

    def has(self, word):
        return word in self.word_map

    def get_all_word_vectors(self):
        return (self.word_vectors, self.word_map)

    def get_word_vector(self, word):
        if word in self.word_map:
            return self.word_vectors[self.word_map[word],:]
        else:
            return None

    def iter_words(self):
        for word in self.word_map:
            yield (word, self.get_word_vector(word))

class PyDsmDictionary(object):

    """
    A dictionary that uses pydsm and loads a .pydsm file
    """

    def __init__(self, file_to_load, words_to_include=None, normalize=True):
        import bz2
        print("Loading '" + file_to_load + "' pickled from pydsm...")
        (self.matrix, row2word, col2word) = pickle.load(bz2.BZ2File(file_to_load, "rb"))
        self.matrix = csr_matrix(self.matrix)
        self.word_map = OrderedDict(zip(row2word,range(len(row2word))))

        if words_to_include is not None:
            if type(words_to_include) is not set: words_to_load = set(words_to_include)
            words = [word for word in row2word if word in words_to_include]
            idxs = [self.word_map[word] for word in words]

            self.matrix = self.matrix[idxs,:]
            if normalize:
                sklearn.preprocessing.normalize(self.matrix, norm='l2', axis=1, copy=False)
            self.matrix = self.matrix.astype("float32")
            self.word_map = OrderedDict(zip(words, range(len(words))))

        print("\rLoaded!")

    @property
    def d(self):
        return self.matrix.shape[1]

    @property
    def n(self):
        return self.matrix.shape[0]

    def has(self, word):
        return word in self.word_map

    def get_all_word_vectors(self):
        return (self.matrix.toarray(), self.word_map)

    def get_word_vector(self, word):
        if word in self.word_map:
            return self.matrix[self.word_map[word],:].toarray().flatten()
        else:
            return None

    def iter_words(self):
        for word in self.word_map:
            yield (word, self.get_word_vector(word))

class GloVeDictionary(object):

    """
     A Dictionary that loads vectors from GloVe output files
    """

    def __init__(self, file_to_load, words_to_include=None, normalize=True):

        if words_to_include is None:
            # Find out total no of words by counting lines in txt file
            with open(file_to_load) as f:
                for i, l in enumerate(f):
                    pass
            self.n = i+1
        else:
            self.n = len(words_to_include)

        # Determine dimensionality d
        with open(file_to_load) as f:
            self.d = len(f.readline().split()) -1

        self.word_map = OrderedDict()
        self.word_vectors = np.empty((self.n, self.d), dtype="float32")

        idx = 0
        with open(file_to_load) as f:
            for line in f:
                line_split = line.split()

                if (words_to_include is not None and line_split[0] in words_to_include) or words_to_include is None:
                    self.word_map[line_split[0]] = idx
                    self.word_vectors[idx,:] = np.array(map(float, line_split[1:]))
                    if normalize:
                        self.word_vectors[idx,:] /= np.linalg.norm(self.word_vectors[idx,:])
                    idx += 1

            if idx < self.n:
                self.word_vectors = self.word_vectors[:idx,:]
                self.n = idx

    def has(self, word):
        return word in self.word_map

    def get_all_word_vectors(self):
        return (self.word_vectors, self.word_map)

    def get_word_vector(self, word):
        if word in self.word_map:
            return self.word_vectors[self.word_map[word],:]
        else:
            return None

    def iter_words(self):
        for word in self.word_map:
            yield (word, self.get_word_vector(word))

class Hdf5Dictionary(object):

    """
    A Dictionary that loads word embeddings from a hdf5 file
    """

    def __init__(self, filepath, words_to_include=None, normalize=True):
        f = h5py.File(filepath + ".hdf5", 'r')
        all_vectors = f["vectors"]
        all_vectors_map = pickle.load(open(filepath + ".pkl"))
        
        if words_to_include is not None:
            self.word_vectors = np.empty((len(words_to_include),all_vectors.shape[1]), dtype="float32")
            self.word_map = OrderedDict()

            i = 0
            for word in words_to_include:
                if word in all_vectors_map:
                    self.word_vectors[i,:] = all_vectors[all_vectors_map[word],:]
                    if normalize:
                        self.word_vectors[i,:] /= np.linalg.norm(self.word_vectors[i,:])
                    self.word_map[word] = i
                    i += 1
        else:
            self.word_vectors = all_vectors #/ np.linalg.norm(all_vectors, axis=1) TODO: normalization is not implemented when words_to_include==None
            self.word_map = all_vectors_map

    @property
    def d(self):
        return self.word_vectors.shape[1]

    @property
    def n(self):
        return self.word_vectors.shape[0]

    def has(self, word):
        return word in self.word_map
    
    def get_all_word_vectors(self):
        return (self.word_vectors, self.word_map)

    def get_word_vector(self, word):
        if word in self.word_map:
            return self.word_vectors[self.word_map[word],:]
        else:
            return None

    def iter_words(self):
        for word in self.word_map:
            yield (word, self.get_word_vector(word))


class RandomDictionary(object):

    """
     A Dictionary that randomizes the embeddings from a standard normal distribution
    """

    def __init__(self, d, words_to_load=None):
        self._d = d
        self.word_map = {}

        if words_to_load is not None:
            if type(words_to_load) is not set: words_to_load = set(words_to_load) # Convert to set for speed
            for word in words_to_load:
                self.get_word_vector(word)

    @property
    def d(self):
        return self._d

    @property
    def n(self):
        return len(self.word_map)

    def has(self, word):
        return word in self.word_map

    def get_word_vector(self, word):
        if word not in self.word_map:
            self.word_map[word] = np.random.uniform(-0.25,0.25,self.d)

        return self.word_map[word] if word in self.word_map else None

    def get_all_word_vectors(self):
        mtx = np.empty((self.n, self.d), dtype="float32")
        ord_dict = OrderedDict()
        i = 0
        for word, vec in self.word_map.iteritems():
            ord_dict[word] = i
            mtx[i,:] = vec
            i += 1

        return (mtx, ord_dict)

class BowDictionary(object):

    def __init__(self, words_to_include):

        self.word_vectors = np.diag(np.ones(len(words_to_include), dtype="float32"))
        self.word_map = OrderedDict()
        i = 0
        for word in words_to_include:
            self.word_map[word] = i
            i += 1

    @property
    def d(self):
        return self.word_vectors.shape[1]

    @property
    def n(self):
        return self.word_vectors.shape[0]

    def has(self, word):
        return word in self.word_map

    def get_word_vector(self, word):
        if word in self.word_map:
            return self.word_vectors[self.word_map[word],:]
        else:
            return None

    def get_all_word_vectors(self):
        return (self.word_vectors, self.word_map)

    def iter_words(self):
        for word in self.word_map:
            yield (word, self.get_word_vector(word))

class StaticDictionary(object):

    """
     A Dictionary that loads embeddings from an earlier dump of any Dictionary type
    """

    def __init__(self, file_to_load):

        print "Loading static embeddings: " + file_to_load.split("/")[-1]
        (self.word_vectors, self.word_map) = pickle.load(open(file_to_load))

    @property
    def d(self):
        return self.word_vectors.shape[1]

    def has(self, word):
        return word in self.word_map

    def get_word_vector(self, word):
        if word in self.word_map:
            return self.word_vectors[self.word_map[word],:]
        else:
            return None

    def get_all_word_vectors(self):
        return (self.word_vectors, self.word_map)

    def iter_words(self):
        for word in self.word_map:
            yield (word, self.get_word_vector(word))

# --- Helper functions ---------------------------------------

def dump(dictionary_object, file_path):
    pickle.dump(dictionary_object.get_all_word_vectors(), open(file_path, mode="w"))