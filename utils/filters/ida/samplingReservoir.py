import collections
import heapq

class SamplingReservoir:
    def __init__(self, nBins, sampleSize, attIndex):
        self.attIndex = attIndex
        self.nBins = nBins
        self.sampleSize = sampleSize
        self.values = [collections.deque() for _ in range(nBins)]
        self.windowValues = collections.deque()
        self.nbSamples = 0
    
    def getNbSamples(self):
        return self.nbSamples
    
    def __str__(self):
        buffer = []
        buffer.append("Attribute [" + str(self.attIndex) + "] \t")
        for i in range(len(self.values)):
            if self.values[i]:
                buffer.append(
                    "[" + str(self.values[i][0]) + ";" + str(self.values[i][-1]) + "](" + str(len(self.values[i])) + ") ")
            else:
                buffer.append("[;]")
        return ''.join(buffer)
    
    def getBin(self, v):
        cv = 0
        while cv < self.nBins and self.values[cv] and v > self.values[cv][-1]:
            cv += 1
        if cv < self.nBins - 1 and self.values[cv + 1] and v == self.values[cv + 1][-1]:
            cv += 1
        return cv
    
    def replaceValue(self, r, v):
        oldV = r
        newV = v
        if oldV == newV:
            return
        oldBin = 0
        newBin = 0
        while oldBin < self.nBins - 1 and self.values[oldBin + 1] and oldV >= self.values[oldBin + 1][0]:
            oldBin += 1
        self.values[oldBin].remove(oldV)
        while newBin < self.nBins - 1 and self.values[newBin + 1] and newV > self.values[newBin + 1][0]:
            newBin += 1
        while newBin < oldBin and newV >= self.values[newBin][-1]:
            newBin += 1
        loc = newBin
        if oldBin >= newBin:
            while loc < oldBin:
                valToMove = self.values[loc].pop()
                self.values[loc + 1].append(valToMove)
                loc += 1
        else:
            while loc > oldBin:
                valToMove = self.values[loc].popleft()
                self.values[loc - 1].appendleft(valToMove)
                loc -= 1
        self.values[newBin].append(newV)
        self.nbSamples += 1
        self.checkOrder()
        self.checkSize()
    
    def insertWithWindow(self, v):
        if len(self.windowValues) < self.sampleSize:
            self.windowValues.append(v)
            self.insertValue(v)
        else:
            r = self.windowValues[0]
            self.windowValues.popleft()
            if r != v:
                self.replaceValue(r, v)
            self.windowValues.append(v)
    
    def insertValue(self, v):
        targetbin = self.nbSamples % self.nBins
        loc = 0
        while loc < self.nBins - 1 and self.values[loc + 1] and v > self.values[loc + 1][0]:
            loc += 1
        while loc < targetbin and v >= self.values[loc][-1]:
            loc += 1
        insertLoc = loc
        if targetbin >= loc:
            while loc < targetbin:
                valToMove = self.values[targetbin - 1].pop()
                self.values[targetbin].append(valToMove)
                targetbin -= 1
        else:
            while loc > targetbin:
                valToMove = self.values[targetbin + 1].popleft()
                self.values[targetbin].appendleft(valToMove)
                targetbin += 1
        self.values[insertLoc].append(v)
        self.nbSamples += 1
        self.checkOrder()
        self.checkSize()
    
    def replace(self, index, v):
        replacementBin = 0
        while index >= len(self.values[replacementBin]):
            index -= len(self.values[replacementBin])
            replacementBin += 1
        vToReplace = 0.0
        count = 0
        for it in self.values[replacementBin]:
            if count == index:
                vToReplace = it
                break
            count += 1
        self.replaceValue(vToReplace, v)
    
    def checkValueInQueues(self, v):
        res = False
        for i in range(len(self.values)):
            res = res or v in self.values[i]
        return res
    
    def checkSize(self):
        for i in range(len(self.values) - 1):
            if self.values[i] and self.values[i + 1] and len(self.values[i]) < len(self.values[i + 1]):
                print("wrong size")
                print(self.__str__())
                print()
    
    def checkOrder(self):
        for i in range(len(self.values) - 1):
            if self.values[i] and self.values[i + 1] and self.values[i][-1] > self.values[i + 1][0]:
                print("wrong order")
                print()

