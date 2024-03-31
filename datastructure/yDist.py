import numpy as np

class yDist:
    def __init__(self, nc):
        self.counts = np.zeros(nc, dtype=int)
        self.total = 0
    
    def p(self, y):
        return self.counts[y] / self.total
    
    def rawP(self, y):
        return self.counts[y] / self.total
    
    def count(self, y):
        return self.counts[y]
    
    def update(self, inst):
        x_C = int(inst.classValue())
        self.counts[x_C] += 1
        self.total += 1
    
    def getNoClasses(self):
        return len(self.counts)