import numpy as np
import logging
from math import log
from collections import Counter
'''
明天实例化一个试试看
'''


class CART(object):
    def __init__(self, dataset):
        self.data, self.target = dataset[:, 1:], dataset[:, 0]
        self.samples, self.numFeat = dataset.shape

    def calGini(self, retData):
        '''
        只计算最后一列的基尼系数即可
        '''
        tot = len(retData)
        probs = list(map(lambda x: x / tot, Counter(retData).values()))
        Gini = 1.0 - sum(probs)
        return Gini

    def splitData(self, dataset, axis, value):
        '''
        保留除去axis列之外的所有信息
        '''
        retData = []
        for data in dataset:
            if data[axis] == value:
                reduceData = data[:axis]
                reduceData.extend(data[axis + 1:])
                retData.append(reduceData)
        return retData

    def choseBestFeat(self, dataset):
        baseGini = self.calGini(dataset[-1])
        minGini = np.inf
        bestFeature = -1
        tot = dataset.shape[0]
        for i in range(dataset.shape[1]):
            tmpData = [exam[i] for exam in dataset]
            tmpData = filter(lambda x: np.isnan(x), tmpData)
            uniqueData = set(tmpData)
            newGini = 0.0
            GiniGain = 0.0
            for val in uniqueData:
                retData = self.splitData(dataset, i, val)
                newGini += len(retData) / tot * self.calGini(retData[-1])
            newGini = len(tmpData) / tot * newGini
            GiniGain = baseGini - newGini
            if GiniGain < minGini:
                minGini = GiniGain
                bestFeature = i
        return bestFeature

    def majorCnt(self, classList):
        cnt = Counter(classList).most_common(1)[0][0]
        return cnt

    def creatTree(self, dataset, labels):
        classList = [exam[-1] for exam in dataset]
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        if len(dataset[0]) == 1:
            return self.majorCnt(classList)
        bestFeat = self.choseBestFeat(dataset)
        bestFeatLabel = labels[bestFeat]
        myTree = {bestFeatLabel: {}}
        del(labels[bestFeat])
        featVal = (exam[bestFeat] for exam in dataset)
        uniqueVal = set(featVal)
        for val in uniqueVal:
            sublabel = labels[:]
            myTree[bestFeatLabel][val] = self.creatTree(
                self.splitData(dataset, bestFeat, val), sublabel)
        return myTree
