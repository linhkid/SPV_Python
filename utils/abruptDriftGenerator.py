from moa.core import Example, FastVector, InstanceExample, ObjectRepository
from moa.options import AbstractOptionHandler, FlagOption, FloatOption, IntOption
from moa.streams import InstanceStream
from moa.tasks import TaskMonitor
from com.yahoo.labs.samoa.instances import Attribute, DenseInstance, Instance, Instances, InstancesHeader

class AbruptDriftGenerator(DriftGenerator):
    def __init__(self):
        self.serialVersionUID = 1291115908166720203
        self.streamHeader = None
        self.pxbd = None
        self.pygxbd = None
        self.pxad = None
        self.pygxad = None
        self.r = None
        self.nInstancesGeneratedSoFar = 0

    def estimatedRemainingInstances(self):
        return -1

    def hasMoreInstances(self):
        return True

    def isRestartable(self):
        return True

    def restart(self):
        self.nInstancesGeneratedSoFar = 0

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
        px = self.pxbd if self.nInstancesGeneratedSoFar <= self.burnInNInstances.getValue() else self.pxad
        pygx = self.pygxbd if self.nInstancesGeneratedSoFar <= self.burnInNInstances.getValue() else self.pygxad
        inst = DenseInstance(self.streamHeader.numAttributes())
        inst.setDataset(self.streamHeader)
        indexes = [0] * self.nAttributes.getValue()

        for a in range(len(indexes)):
            rand = random.uniform(0.0, 1.0)
            chosenVal = 0
            sumProba = px[a][chosenVal]
            while rand > sumProba:
                chosenVal += 1
                sumProba += px[a][chosenVal]
            indexes[a] = chosenVal
            inst.setValue(a, chosenVal)

        lineNoCPT = self.getIndex(indexes)
        chosenClassValue = 0
        while pygx[lineNoCPT][chosenClassValue] != 1.0:
            chosenClassValue += 1
        inst.setClassValue(chosenClassValue)
        self.nInstancesGeneratedSoFar += 1

        return InstanceExample(inst)

    def prepareForUseImpl(self, monitor, repository):
        print("burnIn=" + str(self.burnInNInstances.getValue()))
        self.generateHeader()
        nCombinationsValuesForPX = 1
        for a in range(self.nAttributes.getValue()):
            nCombinationsValuesForPX *= self.nValuesPerAttribute.getValue()
        self.pxbd = [[0.0] * self.nValuesPerAttribute.getValue() for _ in range(self.nAttributes.getValue())]
        self.pygxbd = [[0.0] * self.nValuesPerAttribute.getValue() for _ in range(nCombinationsValuesForPX)]
        rg = random.Random()
        rg.seed(self.seed.getValue())
        self.r = RandomDataGenerator(rg)

        self.generateRandomPx(self.pxbd, self.r)

        self.generateRandomPyGivenX(self.pygxbd, self.r)

        if self.driftPriors.isSet():
            self.pxad = [[0.0] * self.nValuesPerAttribute.getValue() for _ in range(self.nAttributes.getValue())]
            obtainedMagnitude = 0.0
            print("Sampling p(x) for required magnitude...")
            while abs(obtainedMagnitude - self.driftMagnitudePrior.getValue()) > self.precisionDriftMagnitude.getValue():
                if self.driftMagnitudePrior.getValue() >= 0.2:
                    self.generateRandomPx(self.pxad, self.r)
                elif self.driftMagnitudePrior.getValue() < 0.2:
                    self.generateRandomPxAfterCloseToBefore(self.driftMagnitudePrior.getValue(), self.pxbd, self.pxad, self.r)
                obtainedMagnitude = self.computeMagnitudePX(nCombinationsValuesForPX, self.pxbd, self.pxad)
            print("exact magnitude for p(x)=" + str(self.computeMagnitudePX(nCombinationsValuesForPX, self.pxbd, self.pxad)) + "\tasked=" + str(self.driftMagnitudePrior.getValue()))
        else:
            self.pxad = self.pxbd

        if self.driftConditional.isSet():
            self.pygxad = [None] * nCombinationsValuesForPX
            for line in range(len(self.pygxad)):
                self.pygxad[line] = self.pygxbd[line]
            nLinesToChange = round(self.driftMagnitudeConditional.getValue() * nCombinationsValuesForPX)
            if nLinesToChange == 0.0:
                print("Not enough drift to be noticeable in p(y|x) - unchanged")
                self.pygxad = self.pygxbd
            else:
                linesToChange = self.r.nextPermutation(nCombinationsValuesForPX, nLinesToChange)
                for line in linesToChange:
                    self.pygxad[line] = [0.0] * self.nValuesPerAttribute.getValue()
                    lineCPT = self.pygxad[line]
                    chosenClass = 0
                    while self.pygxbd[line][chosenClass] == 1.0:
                        chosenClass = self.r.nextInt(0, len(lineCPT) - 1)
                    for c in range(len(lineCPT)):
                        if c == chosenClass:
                            lineCPT[c] = 1.0
                        else:
                            lineCPT[c] = 0.0
                print("exact magnitude for p(y|x)=" + str(self.computeMagnitudePYGX(self.pygxbd, self.pygxad)) + "\tasked=" + str(self.driftMagnitudeConditional.getValue()))
        else:
            self.pygxad = self.pygxbd

        self.nInstancesGeneratedSoFar = 0

    def getIndex(self, *indexes):
        index = indexes[0]
        for i in range(1, len(indexes)):
            index *= self.nValuesPerAttribute.getValue()
            index += indexes[i]
        return index


