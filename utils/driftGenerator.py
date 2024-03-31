import random
import math
import numpy as np

class DriftGenerator:
    def __init__(self):
        self.nAttributes = 2
        self.nValuesPerAttribute = 2
        self.burnInNInstances = 10000
        self.driftMagnitudePrior = 0.5
        self.driftMagnitudeConditional = 0.5
        self.precisionDriftMagnitude = 0.01
        self.driftConditional = False
        self.driftPriors = False
        self.seed = -1

    def getHeaderAttributes(self, nAttributes, nValuesPerAttribute):
        attributes = []
        attributeValues = []
        for v in range(nValuesPerAttribute):
            attributeValues.append("v" + str(v + 1))
        for i in range(nAttributes):
            attributes.append(("x" + str(i + 1), attributeValues))
        classValues = []
        for v in range(nValuesPerAttribute):
            classValues.append("class" + str(v + 1))
        attributes.append(("class", classValues))
        return attributes

    @staticmethod
    def generateRandomPxAfterCloseToBefore(sigma, base_px, drift_px):
        for a in range(len(drift_px)):
            sum = 0.0
            for v in range(len(drift_px[a])):
                drift_px[a][v] = abs(random.gauss(base_px[a][v], sigma))
                sum += drift_px[a][v]
            for v in range(len(drift_px[a])):
                drift_px[a][v] /= sum

    @staticmethod
    def generateRandomPyGivenX(pygx, alphaDirichlet=None):
        for i in range(len(pygx)):
            lineCPT = pygx[i]
            sum = 0
            for c in range(len(lineCPT)):
                if alphaDirichlet is None:
                    lineCPT[c] = random.gammavariate(1.0, 1.0)
                else:
                    lineCPT[c] = random.gammavariate(alphaDirichlet, 1.0)
                sum += lineCPT[c]
            for c in range(len(lineCPT)):
                lineCPT[c] /= sum

    @staticmethod
    def generateRandomPx(px, verbose=False):
        for a in range(len(px)):
            sum = 0.0
            for v in range(len(px[a])):
                px[a][v] = random.gammavariate(1.0, 1.0)
                sum += px[a][v]
            for v in range(len(px[a])):
                px[a][v] /= sum
            if verbose:
                print("p(x_" + str(a) + ")=" + str(px[a]))

    @staticmethod
    def generateRandomPxWithMissing(px, nMissing, verbose=False):
        for a in range(len(px)):
            sum = 0.0
            for v in range(len(px[a])):
                px[a][v] = random.gammavariate(1.0, 1.0)
                sum += px[a][v]
            missing = random.sample(range(len(px[a])), nMissing)
            for p in missing:
                sum -= px[a][p]
                px[a][p] = 0.0
            for v in range(len(px[a])):
                px[a][v] /= sum
            if verbose:
                print("p(x_" + str(a) + ")=" + str(px[a]))

    @staticmethod
    def generateUniformPx(px):
        for a in range(len(px)):
            for v in range(len(px[a])):
                px[a][v] = 1.0 / len(px[a])

    @staticmethod
    def computeMagnitudePX(nbCombinationsOfValuesPX, base_px, drift_px):
        indexes = [0] * len(base_px)
        m = 0.0
        for i in range(nbCombinationsOfValuesPX):
            DriftGenerator.getIndexes(i, indexes, len(base_px[0]))
            p = 1.0
            q = 1.0
            for a in range(len(indexes)):
                p *= base_px[a][indexes[a]]
                q *= drift_px[a][indexes[a]]
            diff = math.sqrt(p) - math.sqrt(q)
            m += diff * diff
        m = math.sqrt(m) / math.sqrt(2)
        return m

    @staticmethod
    def computeMagnitudePYGX(base_pygx, drift_pygx):
        magnitude = 0.0
        for i in range(len(base_pygx)):
            partialM = 0.0
            for c in range(len(base_pygx[i])):
                diff = math.sqrt(base_pygx[i][c]) - math.sqrt(drift_pygx[i][c])
                partialM += diff * diff
            partialM = math.sqrt(partialM) / math.sqrt(2)
            assert (partialM == 0.0 or partialM == 1.0)
            magnitude += partialM
        magnitude /= len(base_pygx)
        return magnitude

    @staticmethod
    def computeMagnitudeClassPrior(baseClassP, driftClassP):
        magnitude = 0.0
        for c in range(len(baseClassP)):
            diff = math.sqrt(baseClassP[c]) - math.sqrt(driftClassP[c])
            magnitude += diff * diff
        magnitude = math.sqrt(magnitude) / math.sqrt(2)
        return magnitude

    @staticmethod
    def getIndexes(index, indexes, nValuesPerAttribute):
        for i in range(len(indexes) - 1, 0, -1):
            dim = nValuesPerAttribute
            indexes[i] = index % dim
            index //= dim
        indexes[0] = index

    @staticmethod
    def computeClassPrior(px, pygx):
        nClasses = len(pygx[0])
        nAttributes = len(px)
        nValuesPerAttribute = len(px[0])
        classPrior = [0.0] * nClasses
        indexes = [0] * nAttributes
        for lineCPT in range(len(pygx)):
            DriftGenerator.getIndexes(lineCPT, indexes, nValuesPerAttribute)
            probaLine = 1.0
            for a in range(len(indexes)):
                probaLine *= px[a][indexes[a]]
            for c in range(nClasses):
                classPrior[c] += probaLine * pygx[lineCPT][c]
        return classPrior


