import numpy as np
from moa.core import FastVector, InstanceExample, ObjectRepository
from moa.options import AbstractOptionHandler, FlagOption, FloatOption, IntOption
from moa.streams import InstanceStream
from moa.tasks import TaskMonitor
from sklearn.utils import check_random_state
from scipy.stats import truncnorm

class GradualDriftGeneratorLR(DriftGenerator):
    def __init__(self):
        self.driftLength = IntOption("driftLength", 'l', "The number of instances the concept drift will be applied acrros", 1000, 0, Integer.MAX_VALUE)
        self.nTrialsForGeneratingPYGX = 10000
        self.streamHeader = None
        self.pxbd = None
        self.betaInterceptBeforeDrift = None
        self.betasFirstOrderBeforeDrift = None
        self.betasSecondOrderBeforeDrift = None
        self.pxad = None
        self.betaInterceptAfterDrift = None
        self.betasFirstOrderAfterDrift = None
        self.betasSecondOrderAfterDrift = None
        self.r = None
        self.nInstancesGeneratedSoFar = None

    def estimatedRemainingInstances(self):
        return -1

    def hasMoreInstances(self):
        return True

    def isRestartable(self):
        return False

    def restart(self):
        pass

    def getDescription(self, sb, indent):
        pass

    def getPurposeString(self):
        return "Generates a stream with an abrupt drift of given magnitude."

    def getHeader(self):
        return self.streamHeader

    def generateHeader(self):
        attributes = self.getHeaderAttributes(self.nAttributes.getValue(), self.nValuesPerAttribute.getValue())
        self.streamHeader = InstancesHeader(Instances(getCLICreationString(InstanceStream.class), attributes, 0))
        self.streamHeader.setClassIndex(self.streamHeader.numAttributes() - 1)

    def nextInstance(self):
        probSecondDistrib = 0.0
        if self.nInstancesGeneratedSoFar > self.burnInNInstances.getValue():
            probSecondDistrib = 1.0
        elif self.nInstancesGeneratedSoFar > self.burnInNInstances.getValue() + self.driftLength.getValue():
            probSecondDistrib = 1.0 * (self.nInstancesGeneratedSoFar - self.burnInNInstances.getValue()) / self.driftLength.getValue()
        
        isSecondDistrib = self.r.uniform(0.0, 1.0) <= probSecondDistrib
        if isSecondDistrib:
            px = self.pxad
            betaIntercept = self.betaInterceptAfterDrift
            betasFirstOrder = self.betasFirstOrderAfterDrift
            betasSecondOrder = self.betasSecondOrderAfterDrift
        else:
            px = self.pxbd
            betaIntercept = self.betaInterceptBeforeDrift
            betasFirstOrder = self.betasFirstOrderBeforeDrift
            betasSecondOrder = self.betasSecondOrderBeforeDrift
        
        inst = DenseInstance(self.streamHeader.numAttributes())
        inst.setDataset(self.streamHeader)
        indexes = np.zeros(self.nAttributes.getValue(), dtype=int)
        
        for a in range(indexes.shape[0]):
            rand = self.r.uniform(0.0, 1.0)
            chosenVal = 0
            sumProba = px[a][chosenVal]
            while rand > sumProba:
                chosenVal += 1
                sumProba += px[a][chosenVal]
            indexes[a] = chosenVal
            inst.setValue(a, chosenVal)
        
        py = self.calculatePy(betaIntercept, betasFirstOrder, betasSecondOrder, indexes)
        
        y = 1 if self.r.uniform(0.0, 1.0) < py else 0
        inst.setClassValue(y)
        self.nInstancesGeneratedSoFar += 1
        
        return InstanceExample(inst)

    def generateRandomPyGivenX(self, betasFirstOrder, betasSecondOrder):
        for a in range(betasFirstOrder.shape[0]):
            for v in range(betasFirstOrder.shape[1]):
                betasFirstOrder[a][v] = truncnorm.rvs(-3.0, 3.0, size=1)[0]
        
        for a in range(betasSecondOrder.shape[0]):
            for v in range(betasSecondOrder.shape[1]):
                betasSecondOrder[a][v] = truncnorm.rvs(-3.0, 3.0, size=1)[0]

    def generateRandomPyGivenXFlatPrior(self, nCombinationsValuesForPX, px, betasFirstOrder, betasSecondOrder):
        for a in range(betasFirstOrder.shape[0]):
            for v in range(betasFirstOrder.shape[1]):
                betasFirstOrder[a][v] = truncnorm.rvs(-3.0, 3.0, size=1)[0]
        
        for a in range(betasSecondOrder.shape[0]):
            for v in range(betasSecondOrder.shape[1]):
                betasSecondOrder[a][v] = truncnorm.rvs(-3.0, 3.0, size=1)[0]
        
        intercept = truncnorm.rvs(-3.0, 3.0, size=1)[0]
        
        precision = 0.001
        targetPrior = 0.5
        prior = self.calculateClassPrior(nCombinationsValuesForPX, px, intercept, betasFirstOrder, betasSecondOrder)
        while abs(prior - targetPrior) > precision:
            if prior > targetPrior:
                intercept -= 0.001
            else:
                intercept += 0.001
            prior = self.calculateClassPrior(nCombinationsValuesForPX, px, intercept, betasFirstOrder, betasSecondOrder)
        
        return intercept

    def prepareForUseImpl(self, monitor, repository):
        print("burnIn=" + self.burnInNInstances.getValue())
        self.generateHeader()
        nCombinationsValuesForPX = 1
        for a in range(self.nAttributes.getValue()):
            nCombinationsValuesForPX *= self.nValuesPerAttribute.getValue()
        
        self.pxbd = np.zeros((self.nAttributes.getValue(), self.nValuesPerAttribute.getValue()))
        self.betasFirstOrderBeforeDrift = np.zeros((self.nAttributes.getValue(), self.nValuesPerAttribute.getValue()))
        self.betasSecondOrderBeforeDrift = np.zeros((self.nAttributes.getValue() - 1, self.pxbd.shape[0] * self.pxbd.shape[1]))
        for a in range(self.betasSecondOrderBeforeDrift.shape[0]):
            self.betasSecondOrderBeforeDrift[a] = np.zeros(self.betasSecondOrderBeforeDrift[a].shape)
        
        self.r = check_random_state(self.seed.getValue())
        
        self.generateRandomPx(self.pxbd, self.r)
        
        if self.driftPriors.isSet():
            self.pxad = np.zeros((self.nAttributes.getValue(), self.nValuesPerAttribute.getValue()))
            print("Sampling p(x) for required magnitude...")
            sigma = 0.001
            self.generateRandomPxAfterCloseToBefore(sigma, self.pxbd, self.pxad, self.r)
            obtainedMagnitude = self.computeMagnitudePX(nCombinationsValuesForPX, self.pxbd, self.pxad)
            haveReducedSigma = True
            coef = 2.0
            while abs(obtainedMagnitude - self.driftMagnitudePrior.getValue()) > self.precisionDriftMagnitude.getValue():
                if obtainedMagnitude > self.driftMagnitudePrior.getValue():
                    if not haveReducedSigma:
                        coef = 1.0 + coef / 2.0
                    sigma /= coef
                    haveReducedSigma = True
                else:
                    if haveReducedSigma:
                        coef = 1.0 + coef / 2.0
                    sigma *= coef
                    haveReducedSigma = False
                self.generateRandomPxAfterCloseToBefore(sigma, self.pxbd, self.pxad, self.r)
                obtainedMagnitude = self.computeMagnitudePX(nCombinationsValuesForPX, self.pxbd, self.pxad)
            print(sigma)
            print("exact magnitude for p(x)=" + self.computeMagnitudePX(nCombinationsValuesForPX, self.pxbd, self.pxad) + "\tasked=" + self.driftMagnitudePrior.getValue())
        else:
            self.pxad = self.pxbd
        
        self.betaInterceptBeforeDrift = self.generateRandomPyGivenXFlatPrior(nCombinationsValuesForPX, self.pxbd, self.betasFirstOrderBeforeDrift, self.betasSecondOrderBeforeDrift, self.r)
        self.betaInterceptAfterDrift = self.betaInterceptBeforeDrift
        
        if self.driftConditional.isSet():
            self.betasFirstOrderAfterDrift = np.zeros((self.nAttributes.getValue(), self.nValuesPerAttribute.getValue()))
            self.betasSecondOrderAfterDrift = np.zeros((self.nAttributes.getValue() - 1, self.pxbd.shape[0] * self.pxbd.shape[1]))
            for a in range(self.betasSecondOrderAfterDrift.shape[0]):
                self.betasSecondOrderAfterDrift[a] = np.zeros(self.betasSecondOrderAfterDrift[a].shape)
            
            sigma = 0.1
            self.generateRandomPyGxAfterSameClassPriorLR(self.r, nCombinationsValuesForPX)
            obtainedMagnitude = self.computeMagnitudePYGXLRWeighted(nCombinationsValuesForPX)
            haveReducedSigma = True
            coef = 2.0
            nTrials = 0
            bestMagDiff = 2.0
            bestBetaIntercept = self.betaInterceptAfterDrift
            bestBetasFirstAfter = np.zeros(self.betasFirstOrderBeforeDrift.shape)
            bestBetasSecondAfter = np.zeros(self.betasSecondOrderBeforeDrift.shape)
            while nTrials < self.nTrialsForGeneratingPYGX and abs(obtainedMagnitude - self.driftMagnitudeConditional.getValue()) > self.precisionDriftMagnitude.getValue():
                if obtainedMagnitude > self.driftMagnitudeConditional.getValue():
                    if not haveReducedSigma:
                        coef = 1.0 + coef / 2.0
                    sigma /= coef
                    sigma = max(sigma, 10e-5)
                    haveReducedSigma = True
                else:
                    if haveReducedSigma:
                        coef = 1.0 + coef / 2.0
                    sigma *= coef
                    sigma = min(sigma, 2)
                    haveReducedSigma = False
                
                self.generateRandomPyGxAfterSameClassPriorLR(self.r, nCombinationsValuesForPX)
                obtainedMagnitude = self.computeMagnitudePYGXLRWeighted(nCombinationsValuesForPX)
                magDiff = abs(obtainedMagnitude - self.driftMagnitudeConditional.getValue())
                if magDiff < bestMagDiff:
                    bestBetaIntercept = self.betaInterceptAfterDrift
                    bestBetasFirstAfter = np.copy(self.betasFirstOrderAfterDrift)
                    bestBetasSecondAfter = np.copy(self.betasSecondOrderAfterDrift)
                    bestMagDiff = magDiff
                
                nTrials += 1
            
            if nTrials == self.nTrialsForGeneratingPYGX:
                print("Warning, didn't manage to generate the requested magnitude")
                self.betaInterceptAfterDrift = bestBetaIntercept
                self.betasFirstOrderAfterDrift = np.copy(bestBetasFirstAfter)
                self.betasSecondOrderAfterDrift = np.copy(bestBetasSecondAfter)
            
            print("exact magnitude for p(y|x)=" + self.computeMagnitudePYGXLRWeighted(nCombinationsValuesForPX) + "\tasked=" + self.driftMagnitudeConditional.getValue())
        else:
            self.betasFirstOrderAfterDrift = self.betasFirstOrderBeforeDrift
            self.betasSecondOrderAfterDrift = self.betasSecondOrderBeforeDrift
        
        print("prior class before: " + self.calculateClassPrior(nCombinationsValuesForPX, self.pxbd, self.betaInterceptBeforeDrift, self.betasFirstOrderBeforeDrift, self.betasSecondOrderBeforeDrift))
        print("prior class after: " + self.calculateClassPrior(nCombinationsValuesForPX, self.pxad, self.betaInterceptAfterDrift, self.betasFirstOrderAfterDrift, self.betasSecondOrderAfterDrift))
        print("intercept before: " + self.betaInterceptBeforeDrift)
        print("intercept after: " + self.betaInterceptAfterDrift)
        for a in range(self.betasFirstOrderBeforeDrift.shape[0]):
            print("first order before: " + str(self.betasFirstOrderBeforeDrift[a]))
            print("first order after: " + str(self.betasFirstOrderAfterDrift[a]))
        for a in range(self.betasFirstOrderBeforeDrift.shape[0] - 1):
            print("second order before: " + str(self.betasSecondOrderBeforeDrift[a]))
            print("second order after: " + str(self.betasSecondOrderAfterDrift[a]))
        print()
        
        self.nInstancesGeneratedSoFar = 0

    def swapCoefficients(self, tmp1, tmp2, betasFirstOrderAfterDrift2, betasSecondOrderAfterDrift2, r):
        nAttributes = len(betasFirstOrderAfterDrift2)
        a1 = r.nextInt(0, nAttributes-1)
        a2 = None
        while a1 == a2:
            a2 = r.nextInt(0, nAttributes-1)
        for a in range(len(betasFirstOrderBeforeDrift)):
            tmpA = None
            if a == a1:
                tmpA = a2
            elif a == a2:
                tmpA = a1
            else:
                tmpA = a
            for v in range(len(betasFirstOrderBeforeDrift[a])):
                betasFirstOrderAfterDrift2[a][v] = tmp1[tmpA][v]

    def getIndex(self, *indexes):
        index = indexes[0]
        for i in range(1, len(indexes)):
            index *= self.nValuesPerAttribute.getValue()
            index += indexes[i]
        return index

    def generateRandomPyGxAfterCloseToBeforeLR(self, sigma, r2):
        for a in range(len(betasFirstOrderBeforeDrift)):
            for v in range(len(betasFirstOrderBeforeDrift[a])):
                while True:
                    betasFirstOrderAfterDrift[a][v] = r2.nextGaussian(betasFirstOrderBeforeDrift[a][v], sigma)
                    if abs(betasFirstOrderAfterDrift[a][v]) <= 2.0:
                        break
        for a in range(len(betasSecondOrderBeforeDrift)):
            for v in range(len(betasSecondOrderBeforeDrift[a])):
                while True:
                    betasSecondOrderAfterDrift[a][v] = r2.nextGaussian(betasSecondOrderBeforeDrift[a][v], sigma)
                    if abs(betasSecondOrderAfterDrift[a][v]) <= 2.0:
                        break

    def generateRandomPyGxAfterCloseToBeforeSameClassPriorLR(self, sigma, r2, nCombinationsValuesForPX):
        if self.driftPriors.isSet():
            raise RuntimeError("Shouldn't use generateRandomPyGxAfterCloseToBeforeSameClassPriorLR if prior is being drifted")
        m = 0.0
        priorBefore = self.calculateClassPrior(nCombinationsValuesForPX, self.pxbd, self.betaInterceptBeforeDrift, self.betasFirstOrderBeforeDrift, self.betasSecondOrderBeforeDrift)
        for a in range(len(self.betasFirstOrderBeforeDrift)):
            for v in range(len(self.betasFirstOrderBeforeDrift[a])):
                while True:
                    self.betasFirstOrderAfterDrift[a][v] = r2.nextGaussian(self.betasFirstOrderBeforeDrift[a][v], sigma)
                    if abs(self.betasFirstOrderAfterDrift[a][v]) <= 2.0:
                        break
        for a in range(len(self.betasSecondOrderBeforeDrift)):
            for v in range(len(self.betasSecondOrderBeforeDrift[a])):
                while True:
                    self.betasSecondOrderAfterDrift[a][v] = r2.nextGaussian(self.betasSecondOrderBeforeDrift[a][v], sigma)
                    if abs(self.betasSecondOrderAfterDrift[a][v]) <= 2.0:
                        break
        precision = 0.01
        priorAfter = self.calculateClassPrior(nCombinationsValuesForPX, self.pxad, self.betaInterceptAfterDrift, self.betasFirstOrderAfterDrift, self.betasSecondOrderAfterDrift)
        while abs(priorAfter-priorBefore) > precision:
            if priorAfter > priorBefore:
                self.betaInterceptAfterDrift -= 0.01
            else:
                self.betaInterceptAfterDrift += 0.01
            priorAfter = self.calculateClassPrior(nCombinationsValuesForPX, self.pxad, self.betaInterceptAfterDrift, self.betasFirstOrderAfterDrift, self.betasSecondOrderAfterDrift)

    def generateRandomPyGxAfterSameClassPriorLR(self, r2, nCombinationsValuesForPX):
        priorBefore = self.calculateClassPrior(nCombinationsValuesForPX, self.pxbd, self.betaInterceptBeforeDrift, self.betasFirstOrderBeforeDrift, self.betasSecondOrderBeforeDrift)
        self.generateRandomPyGivenX(self.betasFirstOrderAfterDrift, self.betasSecondOrderAfterDrift, r2)
        precision = 0.001
        priorAfter = self.calculateClassPrior(nCombinationsValuesForPX, self.pxad, self.betaInterceptAfterDrift, self.betasFirstOrderAfterDrift, self.betasSecondOrderAfterDrift)
        while abs(priorAfter-priorBefore) > precision:
            if priorAfter > priorBefore:
                self.betaInterceptAfterDrift -= 0.001
            else:
                self.betaInterceptAfterDrift += 0.001
            priorAfter = self.calculateClassPrior(nCombinationsValuesForPX, self.pxad, self.betaInterceptAfterDrift, self.betasFirstOrderAfterDrift, self.betasSecondOrderAfterDrift)

    @staticmethod
    def calculateClassPrior(nCombinationsValuesForPX, px, betaIntercept, betasFirstOrder, betasSecondOrder):
        prior = 0.0
        indexes = np.zeros(len(px))
        for i in range(nCombinationsValuesForPX):
            GradualDriftGeneratorLR.getIndexes(i, indexes, px[0].length)
            tmpPrior = 1.0
            for a in range(len(indexes)):
                tmpPrior *= px[a][indexes[a]]
            py = GradualDriftGeneratorLR.calculatePy(betaIntercept, betasFirstOrder, betasSecondOrder, indexes)
            prior += tmpPrior * py
        return prior

    def computeMagnitudePYGXLR(self, nCombinationsValuesForPX):
        indexes = np.zeros(len(self.pxbd))
        m = 0.0
        for i in range(nCombinationsValuesForPX):
            GradualDriftGeneratorLR.getIndexes(i, indexes, self.nValuesPerAttribute.getValue())
            pyBefore = GradualDriftGeneratorLR.calculatePy(self.betaInterceptBeforeDrift, self.betasFirstOrderBeforeDrift, self.betasSecondOrderBeforeDrift, indexes)
            pyAfter = GradualDriftGeneratorLR.calculatePy(self.betaInterceptAfterDrift, self.betasFirstOrderAfterDrift, self.betasSecondOrderAfterDrift, indexes)
            partialM = 0.0
            diff = math.sqrt(pyBefore) - math.sqrt(pyAfter)
            partialM += diff * diff
            diff = math.sqrt(1.0 - pyBefore) - math.sqrt(1.0 - pyAfter)
            partialM += diff * diff
            partialM = math.sqrt(partialM) / math.sqrt(2)
            m += partialM
        m /= nCombinationsValuesForPX
        return m

    def computeMagnitudePYGXLRWeighted(self, nCombinationsValuesForPX):
        if self.driftPriors.isSet():
            raise RuntimeError("Shouldn't use computeMagnitudePYGXLRWeighted if prior is being drifted")
        indexes = np.zeros(len(self.pxbd))
        m = 0.0
        for i in range(nCombinationsValuesForPX):
            GradualDriftGeneratorLR.getIndexes(i, indexes, self.nValuesPerAttribute.getValue())
            px = 1.0
            for a in range(len(indexes)):
                px *= self.pxbd[a][indexes[a]]
            pyBefore = GradualDriftGeneratorLR.calculatePy(self.betaInterceptBeforeDrift, self.betasFirstOrderBeforeDrift, self.betasSecondOrderBeforeDrift, indexes)
            pyAfter = GradualDriftGeneratorLR.calculatePy(self.betaInterceptAfterDrift, self.betasFirstOrderAfterDrift, self.betasSecondOrderAfterDrift, indexes)
            partialM = 0.0
            diff = math.sqrt(pyBefore) - math.sqrt(pyAfter)
            partialM += diff * diff
            diff = math.sqrt(1.0 - pyBefore) - math.sqrt(1.0 - pyAfter)
            partialM += diff * diff
            partialM = math.sqrt(partialM) / math.sqrt(2)
            m += px * partialM
        return m

    @staticmethod
    def calculatePy(intercept, betasFirstOrder, betasSecondOrder, instanceValuesIndexes):
        py = intercept
        for a in range(len(instanceValuesIndexes)):
            py += betasFirstOrder[a][instanceValuesIndexes[a]]
        for a in range(len(instanceValuesIndexes) - 1):
            index = instanceValuesIndexes[a] * len(betasFirstOrder[a + 1]) + instanceValuesIndexes[a + 1]
            beta = betasSecondOrder[a][index]
            py += beta
        py = 1.0 / (1.0 + math.exp(-py))
        if math.isnan(py):
            for a in range(len(instanceValuesIndexes)):
                print(np.array(betasFirstOrder[a]))
            for a in range(len(instanceValuesIndexes) - 1):
                print(np.array(betasSecondOrder[a]))
        return py

    def getPxbd(self):
        return self.pxbd

    def setPxbd(self, pxbd):
        self.pxbd = pxbd

    def getBetaInterceptBeforeDrift(self):
        return self.betaInterceptBeforeDrift

    def setBetaInterceptBeforeDrift(self, betaInterceptBeforeDrift):
        self.betaInterceptBeforeDrift = betaInterceptBeforeDrift

    def getBetasFirstOrderBeforeDrift(self):
        return self.betasFirstOrderBeforeDrift

    def setBetasFirstOrderBeforeDrift(self, betasFirstOrderBeforeDrift):
        self.betasFirstOrderBeforeDrift = betasFirstOrderBeforeDrift

    def getBetasSecondOrderBeforeDrift(self):
        return self.betasSecondOrderBeforeDrift

    def setBetasSecondOrderBeforeDrift(self, betasSecondOrderBeforeDrift):
        self.betasSecondOrderBeforeDrift = betasSecondOrderBeforeDrift

    def getPxad(self):
        return self.pxad

    def setPxad(self, pxad):
        self.pxad = pxad

    def getBetaInterceptAfterDrift(self):
        return self.betaInterceptAfterDrift

    def setBetaInterceptAfterDrift(self, betaInterceptAfterDrift):
        self.betaInterceptAfterDrift = betaInterceptAfterDrift

    def getBetasFirstOrderAfterDrift(self):
        return self.betasFirstOrderAfterDrift

    def setBetasFirstOrderAfterDrift(self, betasFirstOrderAfterDrift):
        self.betasFirstOrderAfterDrift = betasFirstOrderAfterDrift

    def getBetasSecondOrderAfterDrift(self):
        return self.betasSecondOrderAfterDrift

    def setBetasSecondOrderAfterDrift(self, betasSecondOrderAfterDrift):
        self.betasSecondOrderAfterDrift = betasSecondOrderAfterDrift

    def getR(self):
        return self.r

    def setR(self, r):
        self.r = r

