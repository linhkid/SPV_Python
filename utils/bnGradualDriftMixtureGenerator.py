import moa.core.FastVector
import moa.core.InstanceExample
import moa.core.ObjectRepository
import moa.options.AbstractOptionHandler
import moa.streams.InstanceStream
import moa.tasks.TaskMonitor
import org.apache.commons.math3.random.JDKRandomGenerator
import org.apache.commons.math3.random.RandomDataGenerator
import org.apache.commons.math3.random.RandomGenerator
import com.github.javacliparser.FlagOption
import com.github.javacliparser.FloatOption
import com.github.javacliparser.IntOption
from com.yahoo.labs.samoa.instances import Attribute, DenseInstance, Instance, Instances, InstancesHeader


class BNGradualDriftMixtureGenerator(DriftGenerator):
    driftLength = IntOption("driftLength", 'l', "The number of instances the concept drift will be applied acrros", 0, 0, 10000000)
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
        attributes = self.getHeaderAttributes(nAttributes.getValue(), nValuesPerAttribute.getValue())
        self.streamHeader = InstancesHeader(Instances(getCLICreationString(InstanceStream.class), attributes, 0))
        self.streamHeader.setClassIndex(self.streamHeader.numAttributes() - 1)

    def nextInstance(self):
        px = []
        pygx = []
        probSecondDistrib = 0.0
        if self.nInstancesGeneratedSoFar <= burnInNInstances.getValue():
            probSecondDistrib = 0.0
        elif self.nInstancesGeneratedSoFar > burnInNInstances.getValue() + driftLength.getValue():
            probSecondDistrib = 1.0
        else:
            nInstancesInDrift = int(self.nInstancesGeneratedSoFar - burnInNInstances.getValue())
            probSecondDistrib = 1.0 * nInstancesInDrift / driftLength.getValue()
        isSecondDistrib = self.r.nextUniform(0.0, 1.0, True) <= probSecondDistrib
        if isSecondDistrib:
            px = self.pxad
            pygx = self.pygxad
        else:
            px = self.pxbd
            pygx = self.pygxbd
        inst = DenseInstance(self.streamHeader.numAttributes())
        inst.setDataset(self.streamHeader)
        indexes = [0] * nAttributes.getValue()
        for a in range(len(indexes)):
            rand = self.r.nextUniform(0.0, 1.0, True)
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
        print("burnIn=" + burnInNInstances.getValue())
        self.generateHeader()
        nCombinationsValuesForPX = 1
        for a in range(nAttributes.getValue()):
            nCombinationsValuesForPX *= nValuesPerAttribute.getValue()
        self.pxbd = [[0] * nValuesPerAttribute.getValue() for _ in range(nAttributes.getValue())]
        self.pygxbd = [[0] * nValuesPerAttribute.getValue() for _ in range(nCombinationsValuesForPX)]
        rg = JDKRandomGenerator()
        rg.setSeed(seed.getValue())
        self.r = RandomDataGenerator(rg)
        self.generateRandomPx(self.pxbd, self.r)
        self.generateRandomPyGivenX(self.pygxbd, self.r)
        if driftPriors.isSet():
            self.pxad = [[0] * nValuesPerAttribute.getValue() for _ in range(nAttributes.getValue())]
            obtainedMagnitude = 0
            print("Sampling p(x) for required magnitude...")
            while abs(obtainedMagnitude - driftMagnitudePrior.getValue()) > precisionDriftMagnitude.getValue():
                if driftMagnitudePrior.getValue() >= 0.2:
                    self.generateRandomPx(self.pxad, self.r)
                elif driftMagnitudePrior.getValue() < 0.2:
                    self.generateRandomPxAfterCloseToBefore(driftMagnitudePrior.getValue(), self.pxbd, self.pxad, self.r)
                obtainedMagnitude = self.computeMagnitudePX(nCombinationsValuesForPX, self.pxbd, self.pxad)
            print("exact magnitude for p(x)=" + self.computeMagnitudePX(nCombinationsValuesForPX, self.pxbd, self.pxad) + "\tasked=" + driftMagnitudePrior.getValue())
        else:
            self.pxad = self.pxbd
        if driftConditional.isSet():
            self.pygxad = [self.pygxbd[line] for line in range(len(self.pygxbd))]
            nLinesToChange = round(driftMagnitudeConditional.getValue() * nCombinationsValuesForPX)
            if nLinesToChange == 0.0:
                print("Not enough drift to be noticeable in p(y|x) - unchanged")
                self.pygxad = self.pygxbd
            else:
                linesToChange = self.r.nextPermutation(nCombinationsValuesForPX, nLinesToChange)
                for line in linesToChange:
                    self.pygxad[line] = [0] * nValuesPerAttribute.getValue()
                    lineCPT = self.pygxad[line]
                    chosenClass = 0
                    while pygxbd[line][chosenClass] == 1.0:
                        chosenClass = self.r.nextInt(0, len(lineCPT) - 1)
                    for c in range(len(lineCPT)):
                        if c == chosenClass:
                            lineCPT[c] = 1.0
                        else:
                            lineCPT[c] = 0.0
                print("exact magnitude for p(y|x)=" + self.computeMagnitudePYGX(self.pygxbd, self.pygxad) + "\tasked=" + driftMagnitudeConditional.getValue())
        else:
            self.pygxad = self.pygxbd
        self.nInstancesGeneratedSoFar = 0

    def getIndex(self, *indexes):
        index = indexes[0]
        for i in range(1, len(indexes)):
            index *= nValuesPerAttribute.getValue()
            index += indexes[i]
        return index


