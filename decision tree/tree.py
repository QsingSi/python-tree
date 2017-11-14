import pandas as pd
import os
from math import *
import numpy as np
import pickle
"""
封箱和缺失值处理已经完成
"""


class decisionTree(object):
    def __init__(self):
        pass

    def creatData(self):
        dataSet = [[0, 0, 0, 0, 'N'],
                   [0, 0, 0, 1, 'N'],
                   [1, 0, 0, 0, 'Y'],
                   [2, 1, 0, 0, 'Y'],
                   [2, 2, 1, 0, 'Y'],
                   [2, 2, 1, 1, 'N'],
                   [1, 2, 1, 1, 'Y']]
        labels = ['outlook', 'temperature', 'humidity', 'windy']
        return dataSet, labels

    def calEntropy(self, dataset):
        samples = len(dataset)
        labelcnt = {}
        for feaVec in dataset:
            label = feaVec[-1]
            if label not in labelcnt.keys():
                labelcnt[label] = 0
            labelcnt[label] += 1
        Ent = 0.0
        for key in labelcnt:
            prob = float(labelcnt[key] / samples)
            Ent -= prob * log2(prob)
        return Ent

    def splitDataSet(self, dataset, axis, value):
        retDataSet = []
        for featVec in dataset:
            if featVec[axis] == value:
                reduceVec = featVec[:axis]
                reduceVec.extend(featVec[axis + 1:])
                retDataSet.append(reduceVec)
        return retDataSet

    def processData(self, dataset):
        numBin = 5
        if type(dataset).__name__ == 'list':
            dataset = np.mat(dataset)
        for i in range(dataset.shape[1] - 1):
            colData = dataset[:, i]
            maxData, minData = max(colData), min(colData)
            gap = (maxData - minData) / numBin
            for j in range(colData.length):
                colData[j] = int((colData[j] - minData) / gap)
        return dataset

    def chooseBestFeature(self, dataset):
        numFeature = len(dataset[0]) - 1
        baseEnt = self.calEntropy(dataset)
        maxInfoGain = 0.0
        bestFeature = -1
        for i in range(numFeature):
            featurevalue = [example[i] for example in dataset]
            uniqueVal = set(featurevalue)
            newEnt = 0.0
            splitInfoGain = 0.0
            for val in uniqueVal:
                retDataSet = self.splitDataSet(dataset, i, val)
                prob = float(len(retDataSet) / len(dataset))
                newEnt += prob * self.calEntropy(retDataSet)
                splitInfoGain -= prob * log2(prob)
            infoGain = baseEnt - newEnt
            if splitInfoGain == 0:
                continue
            infoGainRatio = infoGain / splitInfoGain
            if infoGainRatio > maxInfoGain:
                maxInfoGain = infoGainRatio
                bestFeature = i
        return bestFeature

    def majorCnt(self, classList):
        classCnt = {}
        for label in classList:
            if label not in classCnt.keys():
                classCnt[label] = 0
            classCnt += 1
        maxnum = -1
        maxlabel = -1
        for key, val in classCnt.items():
            if val > maxnum:
                maxlabel = key
        return maxlabel

    def creatTree(self, dataset, labels):
        classList = [example[-1] for example in dataset]
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        if len(dataset[0]) == 1:
            return self.majorCnt(classList)
        bestFeat = self.chooseBestFeature(dataset)
        bestFeatLabel = labels[bestFeat]
        myTree = {bestFeatLabel: {}}
        del(labels[bestFeat])
        featVal = [example[bestFeat] for example in dataset]
        uniqueVal = set(featVal)
        for val in uniqueVal:
            sublabel = labels[:]
            myTree[bestFeatLabel][val] = self.creatTree(
                self.splitDataSet(dataset, bestFeat, val), sublabel)
        return myTree

    def classify(self, inputTree, labels, testVec):
        firstLabel = list(inputTree.keys())[0]
        secondDict = inputTree[firstLabel]
        featIndex = labels.index(firstLabel)
        for key in secondDict.keys():
            if testVec[featIndex] == key:
                if type(secondDict[key]).__name__ == 'dict':
                    classLabel = self.classify(secondDict, labels, testVec)
                else:
                    classLabel = secondDict[key]
        return classLabel


if __name__ == '__main__':
    t = decisionTree()
    dataset, labels = t.creatData()
    labels_tmp = labels[:]
    myTree = t.creatTree(dataset, labels_tmp)
    print(myTree)
    # t.main()
