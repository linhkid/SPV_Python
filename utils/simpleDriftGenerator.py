import numpy as np
from moa.core import FastVector, InstanceExample, ObjectRepository
from moa.options import AbstractOptionHandler, FlagOption, FloatOption, IntOption
from moa.streams import InstanceStream
from moa.tasks import TaskMonitor
from sklearn.utils import check_random_state
from sklearn.datasets import make_classification
from sklearn.preprocessing import OneHotEncoder

class SimpleDriftGenerator(DriftGenerator):
    def __init__(self):
        self.driftLength = IntOption("driftLength", 'l', "The number of instances the concept drift will be applied acrros", 0, 0, 10000000)
        self.frequency = IntOption("frequency", 'F', "The number of instances after which to change an entry in pygxbd", 0, 0, 100000)
        self.streamHeader = None
        self.pxbd = None
        self.pygxbd = None
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
        attributes = self.getHeaderAttributes(nAttributes.getValue(), nValuesPerAttribute.getValue())
        self.streamHeader = InstancesHeader(Instances(getCLICreationString(InstanceStream.class), attributes, 0))
        self.streamHeader.setClassIndex(self.streamHeader.numAttributes() - 1)

    def nextInstance(self):
        inst = DenseInstance(self.streamHeader.numAttributes())
        inst.setDataset(self.streamHeader)
        indexes = np.zeros(nAttributes.getValue(), dtype=int)

        for a in range(indexes.shape[0]):
            rand = self.r.nextUniform(0.0, 1.0, True)
            chosenVal = 0
            sumProba = self.pxbd[a][chosenVal]
            while rand > sumProba:
                chosenVal += 1
                sumProba += self.pxbd[a][chosenVal]
            indexes[a] = chosenVal
            inst.setValue(a, chosenVal)

        lineNoCPT = self.getIndex(indexes)

        rand = self.r.nextUniform(0.0, 1.0, True)
        chosenClassValue = 0
        sumProba = self.pygxbd[lineNoCPT][chosenClassValue]
        while rand > sumProba:
            chosenClassValue += 1
            sumProba += self.pygxbd[lineNoCPT][chosenClassValue]
        inst.setClassValue(chosenClassValue)
        self.nInstancesGeneratedSoFar += 1
        return InstanceExample(inst)

    def prepareForUseImpl(self, monitor, repository):
        self.generateHeader()
        nCombinationsValuesForPX = 1
        for a in range(nAttributes.getValue()):
            nCombinationsValuesForPX *= nValuesPerAttribute.getValue()
        self.pxbd = np.zeros((nAttributes.getValue(), nValuesPerAttribute.getValue()))
        self.pygxbd = np.zeros((nCombinationsValuesForPX, nValuesPerAttribute.getValue()))
        self.r = check_random_state(seed.getValue())

        self.generateUniformPx(self.pxbd)

        self.generateRandomPyGivenX(self.pygxbd, self.r)

        self.nInstancesGeneratedSoFar = 0

    def getIndex(self, *indexes):
        index = indexes[0]
        for i in range(1, len(indexes)):
            index *= nValuesPerAttribute.getValue()
            index += indexes[i]
        return index


