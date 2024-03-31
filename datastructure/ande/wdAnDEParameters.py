import sys
import datetime
from datastructure.Parameter import Parameter
from datastructure.indexTrie import indexTrie
from utils import Globals
from weka.core import Instance

class wdAnDEParameters(Parameter):
    MAX_TAB_LENGTH = sys.maxsize - 8

    def __init__(self):
        super().__init__()
        self.scheme = 0
        self.indexTrie_ = []
        self.xyCount = []
        self.timestamp = []
        self.adaptiveControl = ""
        self.adaptiveControlParameter = 0
        self.experimentType = ""
        self.n = Globals.getNumAttributes()
        self.nc = Globals.getNumClasses()
        self.paramsPerAtt = Globals.getParamsPerAtt()
        self.isNumericTrue = Globals.getIsNumericTrue()
        self.adaptiveControl = Globals.getAdaptiveControl()
        self.adaptiveControlParameter = Globals.getAdaptiveControlParameter()
        self.experimentType = Globals.getExperimentType()
        self.level = Globals.getLevel()
        self.indexTrie_ = [indexTrie() for _ in range(self.n)]
        if self.level == 0:
            self.np = self.nc
            for u1 in range(self.n):
                self.indexTrie_[u1].set(self.np)
                self.np += (self.paramsPerAtt[u1] * self.nc)
        elif self.level == 1:
            self.np = self.nc
            for u1 in range(self.n):
                self.indexTrie_[u1].set(self.np)
                self.np += (self.paramsPerAtt[u1] * self.nc)
                self.indexTrie_[u1].children = [indexTrie() for _ in range(self.n)]
                for u2 in range(u1):
                    self.indexTrie_[u1].children[u2].set(self.np)
                    self.np += (self.paramsPerAtt[u1] * self.paramsPerAtt[u2] * self.nc)
        elif self.level == 2:
            self.np = self.nc
            for u1 in range(self.n):
                self.indexTrie_[u1].set(self.np)
                self.np += (self.paramsPerAtt[u1] * self.nc)
                self.indexTrie_[u1].children = [indexTrie() for _ in range(self.n)]
                for u2 in range(u1):
                    self.indexTrie_[u1].children[u2].set(self.np)
                    self.np += (self.paramsPerAtt[u1] * self.paramsPerAtt[u2] * self.nc)
                    self.indexTrie_[u1].children[u2].children = [indexTrie() for _ in range(self.n)]
                    for u3 in range(u2):
                        self.indexTrie_[u1].children[u2].children[u3].set(self.np)
                        self.np += (self.paramsPerAtt[u1] * self.paramsPerAtt[u2] * self.paramsPerAtt[u3] * self.nc)

    def updateFirstPass(self, inst: Instance):
        pass

    def finishedFirstPass(self):
        pass

    def needSecondPass(self):
        pass

    def updateAfterFirstPass(self, inst: Instance):
        pass

    def getAttributeIndex(self, att1, att1val, c):
        offset = self.indexTrie_[att1].offset
        return offset + c * (self.paramsPerAtt[att1]) + att1val

    def getAttributeIndex(self, att1, att1val, att2, att2val, c):
        offset = self.indexTrie_[att1].children[att2].offset
        return offset + c * (self.paramsPerAtt[att1] * self.paramsPerAtt[att2]) + att2val * (self.paramsPerAtt[att1]) + att1val

    def getAttributeIndex(self, att1, att1val, att2, att2val, att3, att3val, c):
        offset = self.indexTrie_[att1].children[att2].children[att3].offset
        return offset + c * (self.paramsPerAtt[att1] * self.paramsPerAtt[att2] * self.paramsPerAtt[att3]) + att3val * (self.paramsPerAtt[att1] * self.paramsPerAtt[att2]) + att2val * (self.paramsPerAtt[att1]) + att1val

    def getAttributeIndex(self, att1, att1val, att2, att2val, att3, att3val, att4, att4val, c):
        offset = self.indexTrie_[att1].children[att2].children[att3].children[att4].offset
        return offset + c * (self.paramsPerAtt[att1] * self.paramsPerAtt[att2] * self.paramsPerAtt[att3] * self.paramsPerAtt[att4]) + att4val * (self.paramsPerAtt[att1] * self.paramsPerAtt[att2] * self.paramsPerAtt[att3]) + att3val * (self.paramsPerAtt[att1] * self.paramsPerAtt[att2]) + att2val * (self.paramsPerAtt[att1]) + att1val

    def getAttributeIndex(self, att1, att1val, att2, att2val, att3, att3val, att4, att4val, att5, att5val, c):
        offset = self.indexTrie_[att1].children[att2].children[att3].children[att4].children[att5].offset
        return offset + c * (self.paramsPerAtt[att1] * self.paramsPerAtt[att2] * self.paramsPerAtt[att3] * self.paramsPerAtt[att4] * self.paramsPerAtt[att5]) + att5val * (self.paramsPerAtt[att1] * self.paramsPerAtt[att2] * self.paramsPerAtt[att3] * self.paramsPerAtt[att4]) + att4val * (self.paramsPerAtt[att1] * self.paramsPerAtt[att2] * self.paramsPerAtt[att3]) + att3val * (self.paramsPerAtt[att1] * self.paramsPerAtt[att2]) + att2val * (self.paramsPerAtt[att1]) + att1val

    def initializeParametersWithVal(self, val):
        pass

    def convertToProbs(self):
        pass

    def getCountAtFullIndex(self, index):
        pass

    def allocateMemoryForCountsParametersProbabilitiesAndGradients(self, size):
        self.xyCount = [0.0] * size
        self.setNp(size)
        if self.experimentType.lower() == "prequential" or self.experimentType.lower() == "flowMachines" or self.experimentType.lower() == "drift":
            if self.adaptiveControl.lower() == "decay":
                self.timestamp = [datetime.datetime.now() for _ in range(size)]