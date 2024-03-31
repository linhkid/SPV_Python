from moa.core import FastVector, InstanceExample, ObjectRepository

from moa.options import AbstractOptionHandler
from moa.streams import InstanceStream
from moa.tasks import TaskMonitor

from com.yahoo.labs.samoa.instances import Attribute, DenseInstance, Instance, Instances, InstancesHeader

class GradualDriftGenerator(DriftGenerator):
    serialVersionUID = 1291115908166720203L
    nAttributes = IntOption("nAttributes", 'n', "Number of attributes as parents of the class", 2, 1, 10)
    nValuesPerAttribute = IntOption("nValuesPerAttribute", 'v', "Number of values per attribute", 2, 2, 5)
    burnInNInstances = IntOption("burnInNInstances", 'b', "Number of instances before the start of the drift", 10000, 1, Integer.MAX_VALUE)
    driftMagnitude = FloatOption("driftMagnitude", 'm', "Magnitude of the drift between the starting probability and the one after the drift. Magnitude is expressed as the Hellinger distance [0,1]", 0.5, 1e-20, 0.9)
    precisionDriftMagnitude = FloatOption("epsilon", 'e', "Precision of the drift magnitude for p(x) (how far from the set magnitude is acceptable)", 0.01, 1e-20, 1.0)
    driftLength = IntOption("driftLength", 'l', "The number of instances the concept drift will be applied acrros", 0, 0, 100)
    driftConditional = FlagOption("driftConditional", 'c', "States if the drift should apply to the conditional distribution p(y|x).")
    driftPriors = FlagOption("driftPriors", 'p', "States if the drift should apply to the prior distribution p(x). ")
    seed = IntOption("seed", 'r', "Seed for random number generator", -1, Integer.MIN_VALUE, Integer.MAX_VALUE)
    streamHeader = InstancesHeader()
    
    pxbd = []
    pygxbd = []
    pxad = []
    pygxad = []
    r = RandomDataGenerator()
    nInstancesGeneratedSoFar = 0
    
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
    
    def getIntermidiateProbability(self, base_prob, drift_prob):
        inter_prob = []
        for i in range(len(base_prob)):
            inter_prob.append([])
            for j in range(len(base_prob[i])):
                increment_per_unit = (base_prob[i][j] + drift_prob[i][j]) / (self.driftLength.getValue() + 1)
                units_to_add = self.nInstancesGeneratedSoFar - self.burnInNInstances.getValue()
                inter_prob[i].append(base_prob[i][j] + increment_per_unit * units_to_add)
        
        return inter_prob
    
    def nextInstance(self):
        px = []
        pygx = []
        if self.burnInNInstances.getValue() < self.nInstancesGeneratedSoFar and self.nInstancesGeneratedSoFar < self.burnInNInstances.getValue() + self.driftLength.getValue():
            print("Getting intermediary probability at " + str(self.nInstancesGeneratedSoFar) + " Instance")
            px = self.getIntermidiateProbability(self.pxbd, self.pxad)
            pygx = self.getIntermidiateProbability(self.pygxbd, self.pygxad)
            print("Done!")
        else:
            px = self.pxbd if self.nInstancesGeneratedSoFar <= self.burnInNInstances.getValue() else self.pxad
            pygx = self.pygxbd if self.nInstancesGeneratedSoFar <= self.burnInNInstances.getValue() else self.pygxad
        inst = DenseInstance(self.streamHeader.numAttributes())
        inst.setDataset(self.streamHeader)
        indexes = [0] * self.nAttributes.getValue()
        print("Setting Values for x_n")
        for a in range(len(indexes)):
            print("a: " + str(a))
            rand = self.r.nextUniform(0.0, 1.0, True)
            chosenVal = 0
            sumProba = px[a][chosenVal]
            while rand > sumProba:
                print("class val: " + str(chosenVal))
                chosenVal += 1
                sumProba += px[a][chosenVal]
            indexes[a] = chosenVal
            inst.setValue(a, chosenVal)
        lineNoCPT = self.getIndex(indexes)
        print("Setting Class Values")
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
        self.pxbd = [[0] * self.nValuesPerAttribute.getValue() for _ in range(self.nAttributes.getValue())]
        self.pygxbd = [[0] * self.nValuesPerAttribute.getValue() for _ in range(nCombinationsValuesForPX)]
        rg = JDKRandomGenerator()
        rg.setSeed(self.seed.getValue())
        self.r = RandomDataGenerator(rg)
        self.generateRandomPx(self.pxbd, self.r)
        self.generateRandomPyGivenX(self.pygxbd, self.r)
        if self.driftPriors.isSet():
            self.pxad = [[0] * self.nValuesPerAttribute.getValue() for _ in range(self.nAttributes.getValue())]
            obtainedMagnitude = 0
            print("Sampling p(x) for required magnitude...")
            while abs(obtainedMagnitude - self.driftMagnitude.getValue()) > self.precisionDriftMagnitude.getValue():
                if self.driftMagnitude.getValue() >= 0.2:
                    self.generateRandomPx(self.pxad, self.r)
                elif self.driftMagnitude.getValue() < 0.2:
                    self.generateRandomPxAfterCloseToBefore(self.driftMagnitude.getValue(), self.pxbd, self.pxad, self.r)
                obtainedMagnitude = self.computeMagnitudePX(nCombinationsValuesForPX, self.pxbd, self.pxad)
            print("exact magnitude for p(x)=" + str(self.computeMagnitudePX(nCombinationsValuesForPX, self.pxbd, self.pxad)) + "\tasked=" + str(self.driftMagnitude.getValue()))
        else:
            self.pxad = self.pxbd
        if self.driftConditional.isSet():
            self.pygxad = [self.pygxbd[line] for line in range(nCombinationsValuesForPX)]
            nLinesToChange = round(self.driftMagnitude.getValue() * nCombinationsValuesForPX)
            if nLinesToChange == 0.0:
                print("Not enough drift to be noticeable in p(y|x) - unchanged")
                self.pygxad = self.pygxbd
            else:
                linesToChange = self.r.nextPermutation(nCombinationsValuesForPX, nLinesToChange)
                for line in linesToChange:
                    self.pygxad[line] = [0] * self.nValuesPerAttribute.getValue()
                    lineCPT = self.pygxad[line]
                    chosenClass = 0
                    while self.pygxbd[line][chosenClass] == 1.0:
                        chosenClass = self.r.nextInt(0, len(lineCPT) - 1)
                    for c in range(len(lineCPT)):
                        if c == chosenClass:
                            lineCPT[c] = 1.0
                        else:
                            lineCPT[c] = 0.0
                print("exact magnitude for p(y|x)=" + str(self.computeMagnitudePYGX(self.pygxbd, self.pygxad)) + "\tasked=" + str(self.driftMagnitude.getValue()))
        else:
            self.pygxad = self.pygxbd
        self.nInstancesGeneratedSoFar = 0

    def getIndex(self, *indexes):
        index = indexes[0]
        for i in range(1, len(indexes)):
            index *= self.nValuesPerAttribute.getValue()
            index += indexes[i]
        return index


