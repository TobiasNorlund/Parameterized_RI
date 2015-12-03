import numpy as np, sys, struct
import pickle
import os.path
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
    "RandomDictionary"
]

WordMeta = namedtuple("WordMeta", "idx dict_idx focus_count context_count")

class RiDictionary(object):

    def __init__(self, path, words_to_load=None, normalize=True):

        self.d = int(path.split("/")[-1].split("-")[2])
        self.k = int(path.split("/")[-1].split("-")[3])
        self.binary_len = np.dtype('float32').itemsize * self.d
        self.word_map = OrderedDict()
        self.normalize = normalize

        sys.stdout.write("Loading word meta data...")

        if words_to_load is not None: words_to_load = set(words_to_load) # Convert to set for speed
        idx = 0
        dict_idx = 0
        for line in open(path + ".map", 'r'):
            splitted = line.split("\t")
            if words_to_load is None or (words_to_load is not None and splitted[0] in words_to_load):
                self.word_map[splitted[0]] = WordMeta(idx, dict_idx, int(splitted[1]), int(splitted[2]))
                dict_idx += 1
            idx += 1

        self.n = len(self.word_map)
        self.f_ctx = open(path + ".context.bin", mode="rb")

        sys.stdout.write("\r")

    def has(self, word):
        return word in self.word_map

    def get_word_vector(self, word):
        ctx = self.get_context(word)
        if ctx is not None:
            return ctx.sum(0)
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
        if word in self.word_map:
            self.f_ctx.seek(self.word_map[word].idx * self.d * 2*self.k * np.dtype('float32').itemsize)
            ctx = np.reshape(np.fromstring(self.f_ctx.read(self.binary_len*2*self.k), dtype='float32'), newshape=( 2*self.k, self.d) )
            return ctx if not self.normalize else ctx / np.sqrt(np.sum(np.sum(ctx, 0)**2))
        else:
            return None

class PmiRiDictionary(RiDictionary):

    def __init__(self, filepathprefix, epsilon, words_to_load=None, use_true_counts=False, normalize=False, cachefile=None):
        super(PmiRiDictionary, self).__init__(filepathprefix, words_to_load, normalize=False)
        self.epsilon = epsilon
        self.normalize = normalize
        self.filepathprefix = filepathprefix
        self.cache = {} if cachefile is None else pickle.load(open(cachefile))

        # Build sparse index vector matrix
        print "Loads index vectors...\n"
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

    def save_cache(self):
        pickle.dump(self.cache, open(self.filepathprefix + ".context.cache.pkl", "w"))

class W2vDictionary(object):


    def __init__(self, path, words_to_load=None):

        f = open(path, mode="rb")
        header = f.readline().split(" ") # read header

        self.d = int(header[1])
        self.n = int(header[0])
        binary_len = np.dtype('float32').itemsize * self.d

        self.word_map = OrderedDict()
        self.word_vectors = np.empty((self.n,self.d), dtype="float32")

        idx = 0
        for line in xrange(self.n):
            word = []
            if line % 10000 == 0: sys.stdout.write("\rLoading w2v vectors... (" + str(line *100.0 / self.n) + "%)")
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)

            vec = np.fromstring(f.read(binary_len), dtype='float32')
            if words_to_load is None or (words_to_load is not None and word in words_to_load):
                self.word_map[word] = idx
                self.word_vectors[idx,:] = vec
                idx += 1

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

    def __init__(self, path, words_to_load=None):
        self.d = int(path.split("/")[-1].split("-")[2])
        self.k = int(path.split("/")[-1].split("-")[3])
        self.binary_len = np.dtype('float32').itemsize * self.d

        if words_to_load is not None:
            words_to_load = set(words_to_load) # Convert to set for speed
            self.locked = True
        else:
            self.locked = False

        idx = 0
        self.word_map = {}
        for line in open(path + ".map", 'r'):
            splitted = line.split("\t")
            if words_to_load is not None and splitted[0] in words_to_load:
                self.word_map[splitted[0]] = np.random.randn(self.d).astype('float32')
                idx += 1

    @property
    def n(self):
        return len(self.word_map)

    def has(self, word):
        return word in self.word_map

    def get_word_vector(self, word):
        if word not in self.word_map and not self.locked:
            self.word_map[word] = np.random.randn(self.d)

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