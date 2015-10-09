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

    def get_context(self, word):

        idx = self.lines.find(word + " ")
        if idx != -1:
            idx = self.lines.count(" ", 0, idx)
            self.f_ctx.seek(idx * self.d * 2*self.k * np.dtype('float32').itemsize)
            ctx = np.fromstring(self.f_ctx.read(self.binary_len*2*self.k), dtype='float32')
            print ctx.shape
            return np.reshape(ctx, newshape=( 2*self.k, self.d) )
        else:
            return None