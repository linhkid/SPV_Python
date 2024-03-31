import random
from moa.core import FastVector, InstanceExample, ObjectRepository
from moa.options import AbstractOptionHandler, FlagOption, FloatOption, IntOption
from moa.streams import InstanceStream
from moa.tasks import TaskMonitor
from yahoo.labs.samoa.instances import Attribute, DenseInstance, Instance, Instances, InstancesHeader

class SimpleTANGenerator(DriftGenerator):
    def __init__(self):
        self.driftLength = IntOption("driftLength", 'l', "The number of instances the concept drift will be applied acrros", 0, 0, 10000000)
        self.frequency = IntOption("frequency", 'F', "The number of instances after which to change an entry in pygxbd", 0, 0, 100000)
        self.streamHeader = None
        self.p_y = None
        self.p_yx = None
        self.p_yxx = None
        self.parents = None
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
        n = self.nAttributes.getValue()
        nvals = self.nValuesPerAttribute.getValue()
        attributes = FastVector()
        attributeValues = []
        for v in range(nvals):
            attributeValues.append("v" + str(v + 1))
        for i in range(n):
            attributes.addElement(Attribute("x" + str(i + 1), attributeValues))
        classValues = []
        for v in range(Globals.getNumClasses()):
            classValues.append("class" + str(v + 1))
        attributes.addElement(Attribute("class", classValues))
        self.streamHeader = InstancesHeader(Instances(getCLICreationString(InstanceStream.class), attributes, 0))
        self.streamHeader.setClassIndex(self.streamHeader.numAttributes() - 1)

    def nextInstance(self):
        inst = DenseInstance(self.streamHeader.numAttributes())
        inst.setDataset(self.streamHeader)
        nc = Globals.getNumClasses()
        n = self.streamHeader.numAttributes() - 1
        y = random.randint(0, nc - 1)
        inst.setClassValue(y)
        x = [0] * n
        x[0] = SUtils.sampleFromNonUniformDistribution(self.p_yx[y], self.r)
        inst.setValue(0, x[0])
        for i in range(1, n):
            p = self.parents[i]
            xp = x[p]
            x[i] = SUtils.sampleFromNonUniformDistribution(self.p_yxx[i][y][xp], self.r)
            inst.setValue(i, x[i])
        self.nInstancesGeneratedSoFar += 1
        return InstanceExample(inst)

    def prepareForUseImpl(self, monitor, repository):
        self.generateHeader()
        nc = Globals.getNumClasses()
        n = self.streamHeader.numAttributes() - 1
        nvals = self.nValuesPerAttribute.getValue()
        rg = random.Random()
        rg.seed(self.seed.getValue())
        self.r = RandomDataGenerator(rg)
        self.parents = [0] * n
        self.parents[1] = 0
        for i in range(2, n):
            p = self.r.nextInt(0, i-1)
            self.parents[i] = p
        print("Seed = " + str(self.seed.getValue()))
        print(self.parents)
        self.p_y = [0] * nc
        self.p_yx = [[0] * nvals for _ in range(nc)]
        for y in range(nc):
            self.p_yx[y] = [0] * nvals
        self.p_yxx = [[[0] * nvals for _ in range(nvals)] for _ in range(n)]
        sum = 0
        for y in range(nc):
            self.p_y[y] = self.r.nextUniform(0, 1)
            sum += self.p_y[y]
        for y in range(nc):
            self.p_y[y] /= sum
        for y in range(nc):
            sum = 0
            for x1 in range(nvals):
                self.p_yx[y][x1] = self.r.nextUniform(0, 1)
                sum += self.p_yx[y][x1]
            for x1 in range(nvals):
                self.p_yx[y][x1] /= sum
        for i in range(1, n):
            for y in range(nc):
                for x1 in range(nvals):
                    sum = 0
                    for x2 in range(nvals):
                        self.p_yxx[i][y][x1][x2] = self.r.nextUniform(0, 1)
                        sum += self.p_yxx[i][y][x1][x2]
                    for x2 in range(nvals):
                        self.p_yxx[i][y][x1][x2] /= sum
        self.nInstancesGeneratedSoFar = 0


