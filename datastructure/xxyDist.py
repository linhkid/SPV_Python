import math
import numpy as np
from weka.core import Instance, Instances
from xyDist import xyDist

class xxyDist:
    def __init__(self, instances):
        self.N = instances.numInstances()
        self.n = instances.numAttributes() - 1
        self.nc = instances.numClasses()
        self.paramsPerAtt = np.zeros(self.n, dtype=int)
        for u in range(self.n):
            self.paramsPerAtt[u] = instances.attribute(u).numValues()
        self.xyDist_ = xyDist(instances)
        self.counts_ = np.zeros((self.n, self.n, self.nc), dtype=float)
    
    def addToCount(self, instances):
        for ii in range(self.N):
            inst = instances.instance(ii)
            self.update(inst)
    
    def update(self, inst):
        self.xyDist_.update(inst)
        x_C = int(inst.classValue())
        for u1 in range(1, self.n):
            x_u1 = int(inst.value(u1))
            for u2 in range(u1):
                x_u2 = int(inst.value(u2))
                pos1 = u1 * x_u1 + u2
                pos2 = x_u2 * self.nc + x_C
                self.counts_[u1][pos1][pos2] += 1
    
    def countsToProbs(self):
        self.xyDist_.countsToProbs()
        self.probs_ = np.zeros((self.n, self.n, self.nc), dtype=float)
        for c in range(self.nc):
            for u1 in range(1, self.n):
                for u1val in range(self.paramsPerAtt[u1]):
                    for u2 in range(u1):
                        for u2val in range(self.paramsPerAtt[u2]):
                            pos1 = u1 * u1val + u2
                            pos2 = u2val * self.nc + c
                            pos3 = self.paramsPerAtt[u2] * self.nc + (u2val * self.nc + c)
                            self.probs_[u1][pos1][pos2] = math.log(max(self.MEsti(self.ref(u1, u1val, u2, u2val, c), self.xyDist_.getCount(u2, u2val, c), self.paramsPerAtt[u1]), 1e-75))
                            self.probs_[u1][pos1][pos3] = math.log(max(self.MEsti(self.ref(u1, u1val, u2, u2val, c), self.xyDist_.getCount(u1, u1val, c), self.paramsPerAtt[u2]), 1e-75))
    
    def countsToAJEProbs(self):
        self.xyDist_.countsToProbs()
        self.probs_ = np.zeros((self.n, self.n, self.nc), dtype=float)
        for c in range(self.nc):
            for u1 in range(1, self.n):
                for u1val in range(self.paramsPerAtt[u1]):
                    for u2 in range(u1):
                        for u2val in range(self.paramsPerAtt[u2]):
                            pos1 = u1 * u1val + u2
                            pos2 = u2val * self.nc + c
                            self.probs_[u1][pos1][pos2] = max(self.MEsti(self.ref(u1, u1val, u2, u2val, c), self.xyDist_.getClassCount(c), self.paramsPerAtt[u1] * self.paramsPerAtt[u2]), 1e-75)
    
    def rawJointP(self, x1, v1, x2, v2, y):
        return self.ref(x1, v1, x2, v2, y) / self.N
    
    def jointP(self, x1, v1, x2, v2, y):
        return self.MEsti(self.ref(x1, v1, x2, v2, y), self.N, self.paramsPerAtt[x1] * self.paramsPerAtt[x2] * self.nc)
    
    def jointP(self, x1, v1, x2, v2):
        return self.MEsti(self.getCount(x1, v1, x2, v2), self.N, self.paramsPerAtt[x1] * self.paramsPerAtt[x2])
    
    def p(self, x1, v1, x2, v2, y):
        return self.MEsti(self.ref(x1, v1, x2, v2, y), self.xyDist_.getCount(x2, v2, y), self.paramsPerAtt[x1])
    
    def pp(self, x1, v1, x2, v2, y):
        return self.pref(x1, v1, x2, v2, y)
    
    def jp(self, x1, v1, x2, v2, y):
        return self.jref(x1, v1, x2, v2, y)
    
    def getCount(self, x1, v1, x2, v2):
        c = 0
        for y in range(self.nc):
            c += self.ref(x1, v1, x2, v2, y)
        return c
    
    def getCount(self, x1, v1, x2, v2, y):
        return self.ref(x1, v1, x2, v2, y)
    
    def ref(self, x1, v1, x2, v2, y):
        if x2 > x1:
            x1, x2 = x2, x1
            v1, v2 = v2, v1
        pos1 = v1 * x1 + x2
        pos2 = v2 * self.nc + y
        return self.counts_[x1][pos1][pos2]
    
    def pref(self, x1, v1, x2, v2, y):
        isX2gX1 = False
        if x2 > x1:
            isX2gX1 = True
            x1, x2 = x2, x1
            v1, v2 = v2, v1
        pos1 = v1 * x1 + x2
        pos2 = self.paramsPerAtt[x2] * self.nc + (v2 * self.nc + y) if isX2gX1 else (v2 * self.nc + y)
        return self.probs_[x1][pos1][pos2]
    
    def jref(self, x1, v1, x2, v2, y):
        if x2 > x1:
            x1, x2 = x2, x1
            v1, v2 = v2, v1
        pos1 = v1 * x1 + x2
        pos2 = v2 * self.nc + y
        return self.probs_[x1][pos1][pos2]
    
    def getNoAtts(self):
        return self.n
    
    def getNoCatAtts(self):
        return self.n
    
    def getNoValues(self, a):
        return self.paramsPerAtt[a]
    
    def getNoData(self):
        return self.N
    
    def setNoData(self):
        self.xyDist_.setNoData()
        self.N += 1
    
    def getNoClasses(self):
        return self.nc
    
    def getNoValues(self):
        return self.paramsPerAtt
