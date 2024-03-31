import random
from abc import ABC, abstractmethod
from typing import List

class IDADiscretizer(ABC):
    def __init__(self, nBins: int, sampleSize: int, type: str):
        self.nBins = nBins
        self.sampleSize = sampleSize
        self.type = type
        self.init = False
        self.nbSeenInstances = 0
        self.nbAttributes = 0
        self.nbNumericalAttributes = 0
        self.sReservoirs = []
        self.discretizedHeader = None

    def getHeader(self):
        return self.discretizedHeader

    def getDescription(self, sb, indent):
        raise NotImplementedError("Not implemented")

    def restartImpl(self):
        self.nBins = self.nBins
        self.sampleSize = self.sampleSize
        self.type = self.type
        self.init = False

    def nextInstance(self):
        if not self.init:
            self.init()

        self.nbSeenInstances += 1
        instEx = self.inputStream.nextInstance().copy()
        inst = instEx.getData()
        discretizedInstance = DenseInstance(self.discretizedHeader.numAttributes())
        discretizedInstance.setDataset(self.discretizedHeader)
        if self.type == "IDA":
            self.updateRandomSample(inst)
        elif self.type == "IDAW":
            self.updateWindowSample(inst)
        nbNumericalAttributesCount = 0
        for i in range(self.nbAttributes):
            if inst.attribute(i).isNumeric() and not inst.isMissing(i):
                v = inst.value(i)
                bin = self.sReservoirs[nbNumericalAttributesCount].getBin(v)
                discretizedInstance.setValue(i, bin)
                nbNumericalAttributesCount += 1
            else:
                if not inst.isMissing(i):
                    discretizedInstance.setValue(i, inst.value(i))
                else:
                    discretizedInstance.setValue(i, float('nan'))

        discretizedInstance.setClassValue(inst.classValue())
        return InstanceExample(discretizedInstance)

    def updateWindowSample(self, inst):
        nbNumericalAttributesCount = 0
        for i in range(self.nbAttributes):
            v = inst.value(i)

            if inst.attribute(i).isNumeric() and not inst.isMissing(i):
                self.sReservoirs[nbNumericalAttributesCount].insertWithWindow(v)
                if not self.sReservoirs[nbNumericalAttributesCount].checkValueInQueues(v):
                    print("Value not added.")
            if inst.attribute(i).isNumeric():
                nbNumericalAttributesCount += 1

    def updateRandomSample(self, inst):
        nbNumericalAttributesCount = 0
        for i in range(self.nbAttributes):
            v = inst.value(i)

            if inst.attribute(i).isNumeric() and not inst.isMissing(i):
                if self.sReservoirs[nbNumericalAttributesCount].getNbSamples() < self.sampleSize:
                    self.sReservoirs[nbNumericalAttributesCount].insertValue(v)
                else:
                    rValue = random.random()
                    if rValue <= self.sampleSize / self.nbSeenInstances:
                        randval = random.randint(0, self.sampleSize - 1)
                        self.sReservoirs[nbNumericalAttributesCount].replace(randval, v)
            if inst.attribute(i).isNumeric():
                nbNumericalAttributesCount += 1

    def init(self):
        self.generateNewHeader()
        self.init = True

        self.nbAttributes = self.getHeader().numAttributes() - 1
        self.sReservoirs = [SamplingReservoir(self.nBins, self.sampleSize, i) for i in range(self.nbNumericalAttributes)]

    def generateNewHeader(self):
        streamHeader = self.inputStream.getHeader()

        nbAttributes = streamHeader.numAttributes()
        attributes = []
        for i in range(nbAttributes):
            attr = streamHeader.attribute(i)

            if attr.isNumeric():
                newAttrLabels = ["b" + str(j) for j in range(self.nBins)]
                attributes.append(Attribute(attr.name(), newAttrLabels))
                self.nbNumericalAttributes += 1
            else:
                attributes.append(attr)
        self.discretizedHeader = InstancesHeader(Instances(getCLICreationString(InstanceStream), attributes, 0))

        self.discretizedHeader.setClassIndex(self.discretizedHeader.numAttributes() - 1)


