import numpy as np
import pandas as pd
from sklearn.utils import shuffle

class wdAnDEParametersIndexedBig(wdAnDEParameters):
    SENTINEL = -1
    PROBA_VALUE_WHEN_ZERO_COUNT = -25
    GRADIENT_VALUE_WHEN_ZERO_COUNT = 0.0
    
    def __init__(self):
        super().__init__()
        self.indexes = None
        self.actualNumberParameters = None
        self.combinationRequired = None
        self.nLines = None
        self.remainders = None
        
    def updateFirstPass(self, inst):
        x_C = int(inst.classValue())
        self.setCombinationRequired(x_C)
        self.N += 1
        if self.level == 0:
            for u1 in range(self.n):
                x_u1 = int(inst.value(u1))
                index = self.getAttributeIndex(u1, x_u1, x_C)
                self.setCombinationRequired(index)
        elif self.level == 1:
            for u1 in range(self.n):
                x_u1 = int(inst.value(u1))
                index = self.getAttributeIndex(u1, x_u1, x_C)
                self.setCombinationRequired(index)
                for u2 in range(u1):
                    x_u2 = int(inst.value(u2))
                    index = self.getAttributeIndex(u1, x_u1, u2, x_u2, x_C)
                    self.setCombinationRequired(index)
        elif self.level == 2:
            for u1 in range(self.n):
                x_u1 = int(inst.value(u1))
                index = self.getAttributeIndex(u1, x_u1, x_C)
                self.setCombinationRequired(index)
                for u2 in range(u1):
                    x_u2 = int(inst.value(u2))
                    index = self.getAttributeIndex(u1, x_u1, u2, x_u2, x_C)
                    self.setCombinationRequired(index)
                    for u3 in range(u2):
                        x_u3 = int(inst.value(u3))
                        index = self.getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, x_C)
                        self.setCombinationRequired(index)
        elif self.level == 3:
            for u1 in range(self.n):
                x_u1 = int(inst.value(u1))
                index = self.getAttributeIndex(u1, x_u1, x_C)
                self.setCombinationRequired(index)
                for u2 in range(u1):
                    x_u2 = int(inst.value(u2))
                    index = self.getAttributeIndex(u1, x_u1, u2, x_u2, x_C)
                    self.setCombinationRequired(index)
                    for u3 in range(u2):
                        x_u3 = int(inst.value(u3))
                        index = self.getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, x_C)
                        self.setCombinationRequired(index)
                        for u4 in range(u3):
                            x_u4 = int(inst.value(u4))
                            index = self.getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, u4, x_u4, x_C)
                            self.setCombinationRequired(index)
        elif self.level == 4:
            for u1 in range(self.n):
                x_u1 = int(inst.value(u1))
                index = self.getAttributeIndex(u1, x_u1, x_C)
                self.setCombinationRequired(index)
                for u2 in range(u1):
                    x_u2 = int(inst.value(u2))
                    index = self.getAttributeIndex(u1, x_u1, u2, x_u2, x_C)
                    self.setCombinationRequired(index)
                    for u3 in range(u2):
                        x_u3 = int(inst.value(u3))
                        index = self.getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, x_C)
                        self.setCombinationRequired(index)
                        for u4 in range(u3):
                            x_u4 = int(inst.value(u4))
                            index = self.getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, u4, x_u4, x_C)
                            self.setCombinationRequired(index)
                            for u5 in range(u4):
                                x_u5 = int(inst.value(u5))
                                index = self.getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, u4, x_u4, u5, x_u5, x_C)
                                self.setCombinationRequired(index)
    
    def finishedFirstPass(self):
        self.indexes = np.zeros((self.nLines, MAX_TAB_LENGTH), dtype=int)
        self.actualNumberParameters = 0
        for l in range(self.indexes.shape[0]):
            for i in range(self.indexes.shape[1]):
                if self.combinationRequired[l][i]:
                    self.indexes[l][i] = self.actualNumberParameters
                    self.actualNumberParameters += 1
                else:
                    self.indexes[l][i] = self.SENTINEL
            self.combinationRequired[l] = None
        print("    Original number of parameters: " + str(self.np) + " (" + str(self.np/(1024*1024*1024)) + "gb)")
        print("    Compressed number of parameters: " + str(self.actualNumberParameters) + " (" + str(self.actualNumberParameters/(1024*1024*1024)) + "gb)")
        ratio = self.actualNumberParameters/self.np
        print("    Compression of: " + str(ratio) + "")
        self.allocateMemoryForCountsParametersProbabilitiesAndGradients(self.actualNumberParameters)
    
    def needSecondPass(self):
        return True
    
    def updateAfterFirstPass(self, inst):
        x_C = int(inst.classValue())
        self.incCountAtFullIndex(x_C)
        if self.level == 0:
            for u1 in range(self.n):
                x_u1 = int(inst.value(u1))
                index = self.getAttributeIndex(u1, x_u1, x_C)
                self.incCountAtFullIndex(index)
        elif self.level == 1:
            for u1 in range(self.n):
                x_u1 = int(inst.value(u1))
                index = self.getAttributeIndex(u1, x_u1, x_C)
                self.incCountAtFullIndex(index)
                for u2 in range(u1):
                    x_u2 = int(inst.value(u2))
                    index = self.getAttributeIndex(u1, x_u1, u2, x_u2, x_C)
                    self.incCountAtFullIndex(index)
        elif self.level == 2:
            for u1 in range(self.n):
                x_u1 = int(inst.value(u1))
                index = self.getAttributeIndex(u1, x_u1, x_C)
                self.incCountAtFullIndex(index)
                for u2 in range(u1):
                    x_u2 = int(inst.value(u2))
                    index = self.getAttributeIndex(u1, x_u1, u2, x_u2, x_C)
                    self.incCountAtFullIndex(index)
                    for u3 in range(u2):
                        x_u3 = int(inst.value(u3))
                        index = self.getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, x_C)
                        self.incCountAtFullIndex(index)
        elif self.level == 3:
            for u1 in range(self.n):
                x_u1 = int(inst.value(u1))
                index = self.getAttributeIndex(u1, x_u1, x_C)
                self.incCountAtFullIndex(index)
                for u2 in range(u1):
                    x_u2 = int(inst.value(u2))
                    index = self.getAttributeIndex(u1, x_u1, u2, x_u2, x_C)
                    self.incCountAtFullIndex(index)
                    for u3 in range(u2):
                        x_u3 = int(inst.value(u3))
                        index = self.getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, x_C)
                        self.incCountAtFullIndex(index)
                        for u4 in range(u3):
                            x_u4 = int(inst.value(u4))
                            index = self.getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, u4, x_u4, x_C)
                            self.incCountAtFullIndex(index)
        elif self.level == 4:
            for u1 in range(self.n):
                x_u1 = int(inst.value(u1))
                index = self.getAttributeIndex(u1, x_u1, x_C)
                self.incCountAtFullIndex(index)
                for u2 in range(u1):
                    x_u2 = int(inst.value(u2))
                    index = self.getAttributeIndex(u1, x_u1, u2, x_u2, x_C)
                    self.incCountAtFullIndex(index)
                    for u3 in range(u2):
                        x_u3 = int(inst.value(u3))
                        index = self.getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, x_C)
                        self.incCountAtFullIndex(index)
                        for u4 in range(u3):
                            x_u4 = int(inst.value(u4))
                            index = self.getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, u4, x_u4, x_C)
                            self.incCountAtFullIndex(index)
                            for u5 in range(u4):
                                x_u5 = int(inst.value(u5))
                                index = self.getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, u4, x_u4, u5, x_u5, x_C)
                                self.incCountAtFullIndex(index)
    
    def getCountAtFullIndex(self, index):
        indexCompact = self.getIndexCompact(index)
        if indexCompact == self.SENTINEL:
            return 0
        else:
            return self.xyCount[indexCompact]
    
    def incCountAtFullIndex(self, index):
        indexCompact = self.getIndexCompact(index)
        if indexCompact != self.SENTINEL:
            self.xyCount[indexCompact] += 1
    
    def getIndexCompact(self, index):
        indexL = int(index / MAX_TAB_LENGTH)
        indexC = int(index % MAX_TAB_LENGTH)
        return self.indexes[indexL][indexC]
    
    def setCombinationRequired(self, index):
        indexL = int(index / MAX_TAB_LENGTH)
        indexC = int(index % MAX_TAB_LENGTH)
        self.combinationRequired[indexL][indexC] = True
