import numpy as np

class xyDist:
    def __init__(self, instances):
        self.N = instances.numInstances()
        self.n = instances.numAttributes() - 1
        self.nc = instances.numClasses()
        self.paramsPerAtt = np.zeros(self.n, dtype=int)
        for u in range(self.n):
            self.paramsPerAtt[u] = instances.attribute(u).numValues()
        self.classCounts_ = np.zeros(self.nc, dtype=int)
        self.counts_ = np.zeros((self.n, np.sum(self.paramsPerAtt) * self.nc), dtype=int)
    
    def addToCount(self, instances):
        for ii in range(self.N):
            inst = instances.instance(ii)
            self.update(inst)
    
    def update(self, inst):
        x_C = int(inst.classValue())
        self.classCounts_[x_C] += 1
        self.N += 1
        for u1 in range(self.n):
            x_u1 = int(inst.value(u1))
            pos = x_u1 * self.nc + x_C
            self.counts_[u1][pos] += 1
    
    def countsToProbs(self):
        self.classProbs_ = np.zeros(self.nc)
        self.probs_ = np.zeros((self.n, np.sum(self.paramsPerAtt) * self.nc))
        for c in range(self.nc):
            self.classProbs_[c] = np.log(SUtils.MEsti(self.classCounts_[c], self.N, self.nc))
        for c in range(self.nc):
            for u in range(self.n):
                for uval in range(self.paramsPerAtt[u]):
                    pos = uval * self.nc + c
                    self.probs_[u][pos] = np.log(max(SUtils.MEsti(self.counts_[u][pos], self.classCounts_[c], self.paramsPerAtt[u]), 1e-75))
    
    def getClassProbs(self):
        return self.classProbs_
    
    def p(self, u1, u1val, y):
        pos = u1val * self.nc + y
        return SUtils.MEsti(self.counts_[u1][pos], self.classCounts_[y], self.paramsPerAtt[u1])
    
    def pp(self, u1, u1val, y):
        pos = u1val * self.nc + y
        return self.probs_[u1][pos]
    
    def p(self, y):
        return SUtils.MEsti(self.classCounts_[y], self.N, self.nc)
    
    def pp(self, y):
        return self.classProbs_[y]
    
    def p(self, u1, u1val):
        return SUtils.MEsti(self.getCount(u1, u1val), self.N, self.paramsPerAtt[u1])
    
    def jointP(self, u1, u1val, y):
        pos = u1val * self.nc + y
        return SUtils.MEsti(self.counts_[u1][pos], self.N, self.paramsPerAtt[u1] * self.nc)
    
    def getCount(self, u1, u1val, y):
        pos = u1val * self.nc + y
        return self.counts_[u1][pos]
    
    def getCount(self, u1, u1val):
        c = 0
        for y in range(self.nc):
            pos = u1val * self.nc + y
            c += self.counts_[u1][pos]
        return c
    
    def getClassCount(self, y):
        return self.classCounts_[y]
    
    def getNoClasses(self):
        return self.nc
    
    def getNoAtts(self):
        return self.n
    
    def getNoCatAtts(self):
        return self.n
    
    def getNoData(self):
        return self.N
    
    def setNoData(self):
        self.N += 1
    
    def getNoValues(self, u):
        return self.paramsPerAtt[u]
    
    def setClassProbs(self, c, d):
        self.classProbs_[c] = d