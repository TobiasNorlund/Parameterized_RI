__author__ = 'tobiasnorlund'

import numpy as np

class RiDictionary(object):


    def __init__(self, path):

        self.d = int(path.split("/")[-1].split("-")[2])
        self.k = int(path.split("/")[-1].split("-")[3])
        self.binary_len = np.dtype('float32').itemsize * self.d

        f_map = open(path + ".map", 'r')
        self.lines = f_map.read()
        f_map.close()

        self.f_ctx = open(path + ".context.bin", mode="rb")

        self.word_map = {}

    def get_context(self, word):

        if word not in self.word_map:

            idx = self.lines.find(word + " ")
            if idx != -1 and word != "":
                idx = self.lines.count(" ", 0, idx)
                self.f_ctx.seek(idx * self.d * 2*self.k * np.dtype('float32').itemsize)
                ctx = np.fromstring(self.f_ctx.read(self.binary_len*2*self.k), dtype='float32')
                self.word_map[word] = np.reshape(ctx, newshape=( 2*self.k, self.d) )
            else:
                self.word_map[word] = None

        return self.word_map[word]


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