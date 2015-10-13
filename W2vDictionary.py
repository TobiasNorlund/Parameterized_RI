import numpy as np
import sys

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