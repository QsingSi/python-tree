import numpy as np
from math import *


class Entropy(object):
    def __init__(self):
        pass

    def get_entropy(self, dataset):
        length = len(dataset)
        cnt = {}
        for num in dataset:
            if cnt.has_key(num):
                cnt[num] += 1
            else:
                cnt[num] = 0
        entropy = 0.0
        l = cnt.values()
        for i in l:
            entropy -= i / length * log2(i / length)
        return entropy

    def info_gain(self, origin, left, right):
        left_entropy = self.get_entropy(left)
        right_entropy = self.get_entropy(right)
        origin_entropy = self.get_entropy(origin)
        infoGain = left_entropy + right_entropy - origin_entropy
        return infoGain
