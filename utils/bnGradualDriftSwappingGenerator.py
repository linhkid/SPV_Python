import numpy as np
from moa.core import FastVector, InstanceExample, ObjectRepository
from moa.options import AbstractOptionHandler, FlagOption, FloatOption, IntOption
from moa.streams import InstanceStream
from moa.tasks import TaskMonitor
from sklearn.utils import check_random_state
from sklearn.utils.extmath import softmax

class BNGradualDriftSwappingGenerator(DriftGenerator):
    def __init__(self):
        self.driftLength = IntOption("driftLength", 'l', "The number of instances the concept drift will be applied acrros", 1000, 0, np.inf)
        self.streamHeader = None
        self.px = None
        self.pygxInit = None
        self.pygxDrifting = None
        self.r = None
        self.nInstancesGeneratedSoFar = None
        self.nLinesToChange = None
        self.linesToChange = None
        self.nLinesChanged = None

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
        if self.nInstancesGeneratedSoFar > self.burnInNInstances.getValue() and self.nInstancesGeneratedSoFar <= self.burnInNInstances.getValue() + self.driftLength.getValue():
            nInstancesInDrift = self.nInstancesGeneratedSoFar - self.burnInNInstances.getValue()
            driftLength = self.driftLength.getValue()
            nLinesShouldHaveSwapped = np.round(1.0 * self.nLinesToChange * nInstancesInDrift / driftLength)
            nLinesToSwapThisInstance = nLinesShouldHaveSwapped - self.nLinesChanged
            copyNLinesChanged = self.nLinesChanged
            for i in range(copyNLinesChanged, nLinesToSwapThisInstance + copyNLinesChanged):
                lineNo = self.linesToChange[i]
                self.pygxDrifting[lineNo] = np.zeros(self.nValuesPerAttribute.getValue())
                lineCPT = self.pygxDrifting[lineNo]
                chosenClass = np.random.randint(0, lineCPT.shape[0] - 1)
                while self.pygxInit[lineNo][chosenClass] == 1.0:
                    chosenClass = np.random.randint(0, lineCPT.shape[0] - 1)
                lineCPT[chosenClass] = 1.0
                self.nLinesChanged += 1

        if self.nInstancesGeneratedSoFar > self.burnInNInstances.getValue() + self.driftLength.getValue():
            if self.nLinesChanged < self.nLinesToChange:
                raise RuntimeError("Should have swapped {} - actually done {}".format(self.nLinesToChange, self.nLinesChanged))

        inst = DenseInstance(self.streamHeader.numAttributes())
        inst.setDataset(self.streamHeader)
        indexes = np.zeros(self.nAttributes.getValue(), dtype=int)

        for a in range(indexes.shape[0]):
            rand = np.random.uniform(0.0, 1.0)
            chosenVal = 0
            sumProba = self.px[a][chosenVal]
            while rand > sumProba:
                chosenVal += 1
                sumProba += self.px[a][chosenVal]
            indexes[a] = chosenVal
            inst.setValue(a, chosenVal)

        lineNoCPT = self.getIndex(indexes)
        chosenClassValue = 0
        while self.pygxDrifting[lineNoCPT][chosenClassValue] != 1.0:
            chosenClassValue += 1
        inst.setClassValue(chosenClassValue)
        self.nInstancesGeneratedSoFar += 1

        return InstanceExample(inst)

    def prepareForUseImpl(self, monitor, repository):
        print("burnIn=" + self.burnInNInstances.getValue())
        self.generateHeader()
        nCombinationsValuesForPX = 1
        for a in range(self.nAttributes.getValue()):
            nCombinationsValuesForPX *= self.nValuesPerAttribute.getValue()
        self.px = np.zeros((self.nAttributes.getValue(), self.nValuesPerAttribute.getValue()))
        self.pygxInit = np.zeros((nCombinationsValuesForPX, self.nValuesPerAttribute.getValue()))
        self.r = check_random_state(self.seed.getValue())

        self.generateRandomPx(self.px, self.r)

        self.generateRandomPyGivenX(self.pygxInit, self.r)
        if self.driftPriors.isSet():
            raise RuntimeError("Drifiting priors not implemented for this generator")

        if self.driftConditional.isSet():
            self.pygxDrifting = np.zeros((nCombinationsValuesForPX, self.nValuesPerAttribute.getValue()))
            for line in range(self.pygxDrifting.shape[0]):
                self.pygxDrifting[line] = self.pygxInit[line]
            self.nLinesToChange = np.round(self.driftMagnitudeConditional.getValue() * nCombinationsValuesForPX).astype(int)
            if self.nLinesToChange == 0:
                print("Not enough drift to be noticeable in p(y|x) - unchanged")
                self.pygxDrifting = self.pygxInit
            else:
                self.linesToChange = np.random.permutation(nCombinationsValuesForPX)[:self.nLinesToChange]
                self.nLinesChanged = 0
                for line in range(self.pygxInit.shape[0]):
                    self.pygxDrifting[line] = self.pygxInit[line].copy()
                print("exact magnitude for p(y|x)={:.2f} asked={:.2f}".format(self.computeMagnitudePYGX(self.pygxInit, self.pygxDrifting), self.driftMagnitudeConditional.getValue()))
        else:
            self.pygxDrifting = self.pygxInit

        self.nInstancesGeneratedSoFar = 0

    def getIndex(self, *indexes):
        index = indexes[0]
        for i in range(1, len(indexes)):
            index *= self.nValuesPerAttribute.getValue()
            index += indexes[i]
        return index


