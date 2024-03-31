import random
from moa.core import FastVector, InstanceExample, ObjectRepository
from moa.options import AbstractOptionHandler, FlagOption, FloatOption, IntOption
from moa.streams import InstanceStream
from moa.tasks import TaskMonitor
from yahoo.labs.samoa.instances import Attribute, DenseInstance, Instance, Instances, InstancesHeader

class SimpleKDBGeneratorWithDriftBinaryValues(DriftGenerator):
    def __init__(self):
        self.driftLength = IntOption("driftLength", 'l', "The number of instances the concept drift will be applied acrros", 0, 0, 10000000)
        self.frequency = IntOption("frequency", 'F', "The number of instances after which to change an entry in pygxbd", 0, 0, 100000)
        self.streamHeader = None
        self.p_y = None
        self.p_yx = None
        self.p_yxx = None
        self.p_yxxx = None
        self.L1 = 0
        self.L2 = 0
        self.L3 = 0
        self.L4 = 0
        self.d_yx = None
        self.d_yxx = None
        self.d_yxxx = None
        self.randAttributes = None
        self.r = None
        self.nInstancesGeneratedSoFar = None
        self.delta = Globals.getDriftDelta()

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
        n = nAttributes.getValue()
        nvals = 2

        self.L1 = 0
        self.L2 = 2
        self.L3 = 50
        self.L4 = n
        print("Out of " + str(n) + " attributes, " + str((self.L2 - self.L1)) + " of them will be order 1 and " + str((self.L3 - self.L2)) + " will be order 2, and " + str((self.L4 - 1 - self.L3)) + " will be order 3.")
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
        n = self.streamHeader.numAttributes() - 1

        numDriftAttributes = int(Globals.getDriftMagnitude())
        if numDriftAttributes > n:
            print("Number of attributes with drift can't be greater than actual number of attributes")
            sys.exit(-1)
        if Globals.getDriftMagnitude2() != 0:
            if self.nInstancesGeneratedSoFar % Globals.getDriftMagnitude2() == 0:
                self.randAttributes = [-1] * numDriftAttributes
                size = 0
                while size < numDriftAttributes:
                    p = random.randint(2, n-1)
                    if p not in self.randAttributes:
                        self.randAttributes[size] = p
                        size += 1

        for i in range(numDriftAttributes):
            p = self.randAttributes[i]
            if p >= self.L1 and p < self.L2:
                self.intoducePosteriorDrift(self.p_yx[p], self.d_yx[p], self.delta)
            elif p >= self.L2 and p < self.L3:
                self.intoducePosteriorDrift(self.p_yxx[p], self.d_yxx[p], self.delta)
            elif p >= self.L3 and p < self.L4:
                self.intoducePosteriorDrift(self.p_yxxx[p], self.d_yxxx[p], self.delta)

        inst = DenseInstance(self.streamHeader.numAttributes())
        inst.setDataset(self.streamHeader)
        nc = Globals.getNumClasses()
        y = random.randint(0, nc - 1)
        inst.setClassValue(y)
        x = [-1] * n
        x[0] = SUtils.sampleFromNonUniformDistribution(self.p_yx[0][y], self.r)
        inst.setValue(0, x[0])
        x[1] = SUtils.sampleFromNonUniformDistribution(self.p_yxx[1][y][x[0]], self.r)
        inst.setValue(1, x[1])
        for i in range(2, n):
            if i >= self.L1 and i < self.L2:
                x[i] = SUtils.sampleFromNonUniformDistribution(self.p_yx[i][y], self.r)
            elif i >= self.L2 and i < self.L3:
                p1 = 0
                xp1 = x[p1]
                x[i] = SUtils.sampleFromNonUniformDistribution(self.p_yxx[i][y][xp1], self.r)
            elif i >= self.L3 and i < self.L4:
                p1 = random.randint(0, 1)
                p2 = 1 - p1
                xp1 = x[p1]
                xp2 = x[p2]
                x[i] = SUtils.sampleFromNonUniformDistribution(self.p_yxxx[i][y][xp1][xp2], self.r)
            inst.setValue(i, x[i])
        self.nInstancesGeneratedSoFar += 1
        return InstanceExample(inst)

    def intoducePosteriorDrift(self, p_yx, d_yx, delta):
        for y in range(2):
            if p_yx[y][0] < delta:
                d_yx[y] = 1
                p_yx[y][0] += delta
                p_yx[y][1] = 1.0 - p_yx[y][0]
            elif p_yx[y][0] > 1.0 - delta:
                d_yx[y] = 0
                p_yx[y][0] -= delta
                p_yx[y][1] = 1.0 - p_yx[y][0]
            elif d_yx[y] == 1:
                p_yx[y][0] += delta
                p_yx[y][1] = 1.0 - p_yx[y][0]
            else:
                p_yx[y][0] -= delta
                p_yx[y][1] = 1.0 - p_yx[y][0]

    def intoducePosteriorDrift(self, p_yxx, d_yxx, delta):
        for p in range(2):
            for y in range(2):
                if p_yxx[y][p][0] < delta:
                    d_yxx[y][p] = 1
                    p_yxx[y][p][0] += delta
                    p_yxx[y][p][1] = 1.0 - p_yxx[y][p][0]
                elif p_yxx[y][p][0] > 1.0 - delta:
                    d_yxx[y][p] = 0
                    p_yxx[y][p][0] -= delta
                    p_yxx[y][p][1] = 1.0 - p_yxx[y][p][0]
                elif d_yxx[y][p] == 1:
                    p_yxx[y][p][0] += delta
                    p_yxx[y][p][1] = 1.0 - p_yxx[y][p][0]
                else:
                    p_yxx[y][p][0] -= delta
                    p_yxx[y][p][1] = 1.0 - p_yxx[y][p][0]

    def intoducePosteriorDrift(self, p_yxxx, d_yxxx, delta):
        for p1 in range(2):
            for p2 in range(2):
                for y in range(2):
                    if p_yxxx[y][p1][p2][0] < delta:
                        d_yxxx[y][p1][p2] = 1
                        p_yxxx[y][p1][p2][0] += delta
                        p_yxxx[y][p1][p2][1] = 1.0 - p_yxxx[y][p1][p2][0]
                    elif p_yxxx[y][p1][p2][0] > 1.0 - delta:
                        d_yxxx[y][p1][p2] = 0
                        p_yxxx[y][p1][p2][0] -= delta
                        p_yxxx[y][p1][p2][1] = 1.0 - p_yxxx[y][p1][p2][0]
                    elif d_yxxx[y][p1][p2] == 1:
                        p_yxxx[y][p1][p2][0] += delta
                        p_yxxx[y][p1][p2][1] = 1.0 - p_yxxx[y][p1][p2][0]
                    else:
                        p_yxxx[y][p1][p2][0] -= delta
                        p_yxxx[y][p1][p2][1] = 1.0 - p_yxxx[y][p1][p2][0]

    def prepareForUseImpl(self, monitor, repository):
        rg = random.Random()
        rg.seed(seed.getValue())
        self.r = RandomDataGenerator(rg)
        self.generateHeader()
        nc = Globals.getNumClasses()
        n = self.streamHeader.numAttributes() - 1
        nvals = 2
        print("Seed = " + str(seed.getValue()))

        self.p_y = [0.0] * nc
        self.p_yx = [[0.0] * nc for _ in range(n)]
        self.p_yx[0] = [[0.0] * nvals for _ in range(nc)]
        for i in range(self.L1, self.L2):
            self.p_yx[i] = [[0.0] * nvals for _ in range(nc)]
        self.p_yxx = [[[[0.0] * nvals for _ in range(nvals)] for _ in range(nc)] for _ in range(n)]
        self.p_yxx[1] = [[[0.0] * nvals for _ in range(nvals)] for _ in range(nc)]
        for i in range(self.L2, self.L3):
            self.p_yxx[i] = [[[0.0] * nvals for _ in range(nvals)] for _ in range(nc)]
        self.p_yxxx = [[[[[0.0] * nvals for _ in range(nvals)] for _ in range(nvals)] for _ in range(nc)] for _ in range(n)]
        for i in range(self.L3, self.L4):
            self.p_yxxx[i] = [[[[0.0] * nvals for _ in range(nvals)] for _ in range(nvals)] for _ in range(nc)]
        self.d_yx = [[0] * nc for _ in range(n)]
        for i in range(self.L1, self.L2):
            self.d_yx[i] = [0] * nc
        self.d_yxx = [[[0] * nvals for _ in range(nc)] for _ in range(n)]
        for i in range(self.L2, self.L3):
            for y in range(nc):
                self.d_yxx[i][y] = [0] * nvals
        self.d_yxxx = [[[[0] * nvals for _ in range(nvals)] for _ in range(nc)] for _ in range(n)]
        for i in range(self.L3, self.L4):
            for y in range(nc):
                for x1 in range(nvals):
                    self.d_yxxx[i][y][x1] = [0] * nvals

        self.intializeCPTBinary()
        self.nInstancesGeneratedSoFar = 0

    def intializeCPTBinary(self):
        nc = Globals.getNumClasses()
        n = self.streamHeader.numAttributes() - 1
        nvals = 2
        self.p_y[0] = self.r.nextUniform(0, 1)
        self.p_y[1] = 1.0 - self.p_y[0]
        for y in range(nc):
            self.p_yx[0][y][0] = self.r.nextUniform(0, 1)
            self.p_yx[0][y][1] = 1.0 - self.p_yx[0][y][0]
        for i in range(self.L1, self.L2):
            for y in range(nc):
                self.p_yx[i][y][0] = self.r.nextUniform(0, 1)
                self.p_yx[i][y][1] = 1.0 - self.p_yx[i][y][0]
        for y in range(nc):
            for x1 in range(nvals):
                self.p_yxx[1][y][x1][0] = self.r.nextUniform(0, 1)
                self.p_yxx[1][y][x1][1] = 1 - self.p_yxx[1][y][x1][0]
        for i in range(self.L2, self.L3):
            for y in range(nc):
                for x1 in range(nvals):
                    self.p_yxx[i][y][x1][0] = self.r.nextUniform(0, 1)
                    self.p_yxx[i][y][x1][1] = 1 - self.p_yxx[i][y][x1][0]
        for i in range(self.L3, self.L4):
            for y in range(nc):
                for x1 in range(nvals):
                    for x2 in range(nvals):
                        self.p_yxxx[i][y][x1][x2][0] = self.r.nextUniform(0, 1)
                        self.p_yxxx[i][y][x1][x2][1] = 1 - self.p_yxxx[i][y][x1][x2][0]
        for i in range(self.L1, self.L2):
            for y in range(nc):
                self.d_yx[i][y] = self.r.nextInt(0, 1)
                self.d_yx[i][y] = self.r.nextInt(0, 1)
        for i in range(self.L2, self.L3):
            for y in range(nc):
                for x in range(nvals):
                    self.d_yxx[i][y][x] = self.r.nextInt(0, 1)
                    self.d_yxx[i][y][x] = self.r.nextInt(0, 1)
        for i in range(self.L3, self.L4):
            for y in range(nc):
                for x1 in range(nvals):
                    for x2 in range(nvals):
                        self.d_yxxx[i][y][x1][x2] = self.r.nextInt(0, 1)
                        self.d_yxxx[i][y][x1][x2] = self.r.nextInt(0, 1)


