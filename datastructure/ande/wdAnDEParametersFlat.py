import sys
import time
import datetime
import queue
import numpy as np
import pandas as pd
import weka.core.Instance

class wdAnDEParametersFlat(wdAnDEParameters):
    def __init__(self):
        super().__init__()
        self.queue = queue.Queue()
        self.counter = 0
        if Globals.isVerbose():
            print("In the Constructor of wdAnDEParametersFlat(), np = " + str(np))
        if np > MAX_TAB_LENGTH:
            print("CRITICAL ERROR: --structureParameter: 'Flat' not implemented for such dimensionalities. Use 'IndexedBig' or 'BitMap' or 'Hash'")
            sys.exit(-1)
        self.allocateMemoryForCountsParametersProbabilitiesAndGradients(np)
    
    def updateFirstPass(self, inst):
        if adaptiveControl.equalsIgnoreCase("Decay"):
            self.applyDecay()
        elif adaptiveControl.equalsIgnoreCase("Window"):
            if self.counter < adaptiveControlParameter:
                self.queue.put(inst)
            else:
                self.unUpdateFirstPass(self.queue.get())
                self.queue.put(inst)
            self.counter += 1
        x_C = int(inst.classValue())
        xyCount[x_C] += 1
        N += 1
        if level == 0:
            for u1 in range(n):
                x_u1 = int(inst.value(u1))
                index = int(getAttributeIndex(u1, x_u1, x_C))
                self.incCountAtFullIndex(index)
        elif level == 1:
            for u1 in range(n):
                x_u1 = int(inst.value(u1))
                index = int(getAttributeIndex(u1, x_u1, x_C))
                self.incCountAtFullIndex(index)
                for u2 in range(u1):
                    x_u2 = int(inst.value(u2))
                    index = int(getAttributeIndex(u1, x_u1, u2, x_u2, x_C))
                    self.incCountAtFullIndex(index)
        elif level == 2:
            for u1 in range(n):
                x_u1 = int(inst.value(u1))
                index = int(getAttributeIndex(u1, x_u1, x_C))
                self.incCountAtFullIndex(index)
                for u2 in range(u1):
                    x_u2 = int(inst.value(u2))
                    index = int(getAttributeIndex(u1, x_u1, u2, x_u2, x_C))
                    self.incCountAtFullIndex(index)
                    for u3 in range(u2):
                        x_u3 = int(inst.value(u3))
                        index = int(getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, x_C))
                        self.incCountAtFullIndex(index)
    
    def unUpdateFirstPass(self, inst):
        x_C = int(inst.classValue())
        xyCount[x_C] -= 1
        N -= 1
        if level == 0:
            for u1 in range(n):
                x_u1 = int(inst.value(u1))
                index = int(getAttributeIndex(u1, x_u1, x_C))
                self.decCountAtFullIndex(index)
        elif level == 1:
            for u1 in range(n):
                x_u1 = int(inst.value(u1))
                index = int(getAttributeIndex(u1, x_u1, x_C))
                self.decCountAtFullIndex(index)
                for u2 in range(u1):
                    x_u2 = int(inst.value(u2))
                    index = int(getAttributeIndex(u1, x_u1, u2, x_u2, x_C))
                    self.decCountAtFullIndex(index)
        elif level == 2:
            for u1 in range(n):
                x_u1 = int(inst.value(u1))
                index = int(getAttributeIndex(u1, x_u1, x_C))
                self.decCountAtFullIndex(index)
                for u2 in range(u1):
                    x_u2 = int(inst.value(u2))
                    index = int(getAttributeIndex(u1, x_u1, u2, x_u2, x_C))
                    self.decCountAtFullIndex(index)
                    for u3 in range(u2):
                        x_u3 = int(inst.value(u3))
                        index = int(getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, x_C))
                        self.decCountAtFullIndex(index)
    
    def applyDecay(self):
        a = adaptiveControlParameter
        for i in range(np):
            xyCount[i] = xyCount[i] * np.exp(-a)
        N = N * np.exp(-a)
    
    def finishedFirstPass(self):
        pass
    
    def needSecondPass(self):
        return False
    
    def updateAfterFirstPass(self, inst):
        pass
    
    def getCountAtFullIndex(self, index):
        return xyCount[int(index)]
    
    def incCountAtFullIndex(self, index):
        xyCount[int(index)] += 1
    
    def decCountAtFullIndex(self, index):
        xyCount[int(index)] -= 1