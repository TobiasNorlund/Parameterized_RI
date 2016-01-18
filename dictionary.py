__author__ = 'tobiasnorlund'

import numpy as np
import sys

class RiDictionary(object):


    def __init__(self, path):

        self.d = int(path.split("/")[-1].split("-")[2])
        self.k = int(path.split("/")[-1].split("-")[3])
        self.binary_len = np.dtype('float32').itemsize * self.d

        f_map = open(path + ".map__", 'r')
        self.lines = f_map.read()
        f_map.close()

        self.f_ctx = open(path + ".context.bin", mode="rb")

        self.word_map = {}

    def get_context(self, word):

        if word not in self.word_map:

            idx = self.lines.find(word + " ")
            if idx != -1:
                idx = self.lines.count(" ", 0, idx)
                self.f_ctx.seek(idx * self.d * 2*self.k * np.dtype('float32').itemsize)
                ctx = np.fromstring(self.f_ctx.read(self.binary_len*2*self.k), dtype='float32')
                self.word_map[word] = np.reshape(ctx, newshape=( 2*self.k, self.d) )
            else:
                self.word_map[word] = None

        return self.word_map[word]



class W2vDictionary(object):


    def __init__(self, path):

        f = open(path, mode="rb")
        self.data = f.read()
        f.close()

        layer1_size = 300
        vocab_size = 3000000
        self.binary_len = np.dtype('float32').itemsize * layer1_size

        self.word_map = {}

        i = 12
        for line in xrange(vocab_size):
            word = []
            if line % 10000 == 0: sys.stdout.write("\r" + str(line *100.0 / vocab_size) + "%")
            while True:
                ch = self.data[i]
                i += 1
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)

            self.word_map[word] = i
            i += self.binary_len


    def get_context(self, word):
        if word in self.word_map:
            idx = self.word_map[word]
            return np.fromstring(self.data[idx:idx + self.binary_len], dtype='float32')
        else:
            return None

class RandomDictionary(object):

    def __init__(self, path):
        self.d = int(path.split("/")[-1].split("-")[2])
        self.k = int(path.split("/")[-1].split("-")[3])
        self.binary_len = np.dtype('float32').itemsize * self.d

        f_map = open(path + ".map", 'r')
        self.lines = f_map.read()
        f_map.close()

        self.word_map = {}

    def get_context(self, word):

        if word not in self.word_map:
            idx = self.lines.find(word + " ")
            if idx != -1:
                self.word_map[word] = np.random.randn(2*self.k, self.d)
            else:
                self.word_map[word] = None

        return self.word_map[word]