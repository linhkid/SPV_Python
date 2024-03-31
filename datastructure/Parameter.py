import numpy as np
import pandas as pd

class Parameter:
    def __init__(self):
        self.np = 0
        self.originalNP = 0
        self.nc = 0
        self.n = 0
        self.N = 0
        self.paramsPerAtt = []
        self.isNumericTrue = []
        self.level = 1

    def getN(self):
        return self.N

    def getNC(self):
        return self.nc

    def getn(self):
        return self.n

    def getNp(self):
        return self.np

    def getOriginalNP(self):
        return self.originalNP

    def setOriginalNP(self, originalNP):
        self.originalNP = originalNP

    def getParamsPerAtt(self):
        return self.paramsPerAtt

    def setNp(self, newNP):
        self.np = newNP

    def getTotalNumberParameters(self):
        return self.np

    def initializeParametersWithVal(self, val):
        pass

    def convertToProbs(self):
        pass

    def updateFirstPass(self, row):
        pass

    def finishedFirstPass(self):
        pass

    def needSecondPass(self):
        pass

    def updateAfterFirstPass(self, row):
        pass

    def determineNP(self):
        return 0

    def unUpdateAfterFirstPass(self, row):
        pass

    def startFSPass(self, fsScore):
        pass

    def getAttributeIndex(self, att1, att1valindex, c):
        return 0

    def getAttributeIndex(self, att1, att1valindex, att2, att2valindex, c):
        return 0

    def getAttributeIndex(self, att1, att1valindex, att2, att2valindex, att3, att3valindex, c):
        return 0

    def getAttributeIndex(self, att1, att1valindex, att2, att2valindex, att3, att3valindex, att4, att4valindex, c):
        return 0

    def getAttributeIndex(self, att1, att1valindex, att2, att2valindex, att3, att3valindex, att4, att4valindex, att5, att5valindex, c):
        return 0

    def getCompactIndexAtFullIndex(self, index):
        return 0

    def getProbAtFullIndex(self, index):
        return 0

    def getParameterAtFullIndex(self, index):
        return 0

    def setParameterAtFullIndex(self, index, p):
        pass

    def getGradientAtFullIndex(self, index):
        return 0

    def setGradientAtFullIndex(self, index, p):
        pass

    def finishedSecondPass(self):
        pass

    def startThirdPass(self):
        pass

    def getClassProbabilities(self):
        return None

    def getParameters(self):
        return None

    def getOrder(self):
        return None

    def getParents(self):
        return None

    def getCountAtFullIndex(self, index):
        return 0

    def getAttributeIndex(self, att1, att1valindex, c, k):
        return 0