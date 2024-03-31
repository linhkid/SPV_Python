import random
from moa.core import FastVector, InstanceExample, ObjectRepository
from moa.options import AbstractOptionHandler, FlagOption, FloatOption, IntOption
from moa.streams import InstanceStream
from moa.tasks import TaskMonitor
from yahoo.labs.samoa.instances import Attribute, DenseInstance, Instance, Instances, InstancesHeader
from utils import DriftGenerator

class SimpleTANGeneratorWithDriftBinaryValues(DriftGenerator):
    def __init__(self):
        self.driftLength = IntOption("driftLength", 'l', "The number of instances the concept drift will be applied acrros", 0, 0, 10000000)
        self.frequency = IntOption("frequency", 'F', "The number of instances after which to change an entry in pygxbd", 0, 0, 100000)
        self.streamHeader = None
        self.p_y = None
        self.p_yx = None
        self.p_yxx = None
        self.d_yxx = None
        self.d_yx = None
        self.parents = None
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
        if self.nInstancesGeneratedSoFar % Globals.getDriftMagnitude2() == 0:
            self.randAttributes = [None] * numDriftAttributes
            size = 0
            while size < numDriftAttributes:
                p = random.randint(1, n-1)
                if p not in self.randAttributes:
                    self.randAttributes[size] = p
                    size += 1
        for i in range(numDriftAttributes):
            p = self.randAttributes[i]
            if p == 0:
                pass
            else:
                self.intoducePosteriorDrift(self.p_yxx[p], self.delta)
        inst = DenseInstance(self.streamHeader.numAttributes())
        inst.setDataset(self.streamHeader)
        nc = Globals.getNumClasses()
        y = random.randint(0, nc - 1)
        inst.setClassValue(y)
        x = [None] * n
        x[0] = SUtils.sampleFromNonUniformDistribution(self.p_yx[y], self.r)
        inst.setValue(0, x[0])
        for i in range(1, n):
            p = self.parents[i]
            xp = x[p]
            x[i] = SUtils.sampleFromNonUniformDistribution(self.p_yxx[i][y][xp], self.r)
            inst.setValue(i, x[i])
        self.nInstancesGeneratedSoFar += 1
        return InstanceExample(inst)

    def intoducePosteriorDrift(self, p_yxx, delta):
        for p in range(2):
            for y in range(2):
                if p_yxx[y][p][0] < delta:
                    self.d_yxx[y][p] = 1
                    p_yxx[y][p][0] += delta
                    p_yxx[y][p][1] = 1.0 - p_yxx[y][p][0]
                elif p_yxx[y][p][0] > 1.0 - delta:
                    self.d_yxx[y][p] = 0
                    p_yxx[y][p][0] -= delta
                    p_yxx[y][p][1] = 1.0 - p_yxx[y][p][0]
                elif self.d_yxx[y][p] == 1:
                    p_yxx[y][p][0] += delta
                    p_yxx[y][p][1] = 1.0 - p_yxx[y][p][0]
                else:
                    p_yxx[y][p][0] -= delta
                    p_yxx[y][p][1] = 1.0 - p_yxx[y][p][0]

    def intoducePosteriorDrift(self, p_yx, delta):
        for y in range(2):
            if p_yx[y][0] < delta:
                self.d_yx[y] = 1
                p_yx[y][0] += delta
                p_yx[y][1] = 1.0 - p_yx[y][0]
            elif p_yx[y][0] > 1.0 - delta:
                self.d_yx[y] = 0
                p_yx[y][0] -= delta
                p_yx[y][1] = 1.0 - p_yx[y][0]
            elif self.d_yx[y] == 1:
                p_yx[y][0] += delta
                p_yx[y][1] = 1.0 - p_yx[y][0]
            else:
                p_yx[y][0] -= delta
                p_yx[y][1] = 1.0 - p_yx[y][0]

    def intoducePosteriorDrift8(self, p_yxx: List[List[List[float]]], delta: float):
        for p in range(2):
            for y in range(2):
                if p_yxx[y][p][0] < delta:
                    p_yxx[y][p][0] += delta
                    p_yxx[y][p][1] = 1.0 - p_yxx[y][p][0]
                elif p_yxx[y][p][0] > 1.0 - delta:
                    p_yxx[y][p][0] -= delta
                    p_yxx[y][p][1] = 1.0 - p_yxx[y][p][0]
                elif r.nextInt(0, 1) == 0:
                    p_yxx[y][p][0] += delta
                    p_yxx[y][p][1] = 1.0 - p_yxx[y][p][0]
                else:
                    p_yxx[y][p][0] -= delta
                    p_yxx[y][p][1] = 1.0 - p_yxx[y][p][0]

    def intoducePosteriorDrift8(self, p_yx: List[List[float]], delta: float):
        for y in range(2):
            if p_yx[y][0] < delta:
                p_yx[y][0] += delta
                p_yx[y][1] = 1.0 - p_yx[y][0]
            elif p_yx[y][0] > 1.0 - delta:
                p_yx[y][0] -= delta
                p_yx[y][1] = 1.0 - p_yx[y][0]
            elif r.nextInt(0, 1) == 0:
                p_yx[y][0] += delta
                p_yx[y][1] = 1.0 - p_yx[y][0]
            else:
                p_yx[y][0] -= delta
                p_yx[y][1] = 1.0 - p_yx[y][0]

    def intoducePosteriorDrift7(self, p_yxx: List[List[List[float]]], delta: float):
        if p_yxx[0][0][0] < delta:
            assert(p_yxx[0][0][1] >= delta)
            p_yxx[0][0][0] += delta
            p_yxx[1][0][0] = 1.0 - p_yxx[0][0][0]
            p_yxx[0][0][1] -= delta
            p_yxx[1][0][1] = 1.0 - p_yxx[0][0][1]
        elif p_yxx[0][0][0] > 1.0 - delta:
            assert(p_yxx[0][0][1] <= 1.0 - delta)
            p_yxx[0][0][0] -= delta
            p_yxx[1][0][0] = 1.0 - p_yxx[0][0][0]
            p_yxx[0][0][1] += delta
            p_yxx[1][0][1] = 1.0 - p_yxx[0][0][1]
        elif r.nextInt(0, 1) == 0:
            assert(p_yxx[0][0][1] >= delta)
            p_yxx[0][0][0] += delta
            p_yxx[1][0][0] = 1.0 - p_yxx[0][0][0]
            p_yxx[0][0][1] -= delta
            p_yxx[1][0][1] = 1.0 - p_yxx[0][0][1]
        else:
            assert(p_yxx[0][0][1] <= 1.0 - delta)
            p_yxx[0][0][0] -= delta
            p_yxx[1][0][0] = 1.0 - p_yxx[0][0][0]
            p_yxx[0][0][1] += delta
            p_yxx[1][0][1] = 1.0 - p_yxx[0][0][1]

        if p_yxx[0][1][0] < delta:
            assert(p_yxx[0][1][1] >= delta)
            p_yxx[0][1][0] += delta
            p_yxx[1][1][0] = 1.0 - p_yxx[0][1][0]
            p_yxx[0][1][1] -= delta
            p_yxx[1][1][1] = 1.0 - p_yxx[0][1][1]
        elif p_yxx[0][1][0] > 1.0 - delta:
            assert(p_yxx[0][1][1] <= 1.0 - delta)
            p_yxx[0][1][0] -= delta
            p_yxx[1][1][0] = 1.0 - p_yxx[0][1][0]
            p_yxx[0][1][1] += delta
            p_yxx[1][1][1] = 1.0 - p_yxx[0][1][1]
        elif r.nextInt(0, 1) == 0:
            assert(p_yxx[0][1][1] >= delta)
            p_yxx[0][1][0] += delta
            p_yxx[1][1][0] = 1.0 - p_yxx[0][1][0]
            p_yxx[0][1][1] -= delta
            p_yxx[1][1][1] = 1.0 - p_yxx[0][1][1]
        else:
            assert(p_yxx[0][0][1] <= 1.0 - delta)
            p_yxx[0][1][0] -= delta
            p_yxx[1][1][0] = 1.0 - p_yxx[0][1][0]
            p_yxx[0][1][1] += delta
            p_yxx[1][1][1] = 1.0 - p_yxx[0][1][1]

    def intoducePosteriorDrift7(self, p_yx: List[List[float]], delta: float):
        if p_yx[0][0] < delta:
            assert(p_yx[0][1] >= delta)
            p_yx[0][0] += delta
            p_yx[1][0] = 1.0 - p_yx[0][0]
            p_yx[0][1] -= delta
            p_yx[1][1] = 1.0 - p_yx[0][1]
        elif p_yx[0][0] > 1.0 - delta:
            assert(p_yx[0][1] <= 1.0 - delta)
            p_yx[0][0] -= delta
            p_yx[1][0] = 1.0 - p_yx[0][0]
            p_yx[0][1] += delta
            p_yx[1][1] = 1.0 - p_yx[0][1]
        elif r.nextInt(0, 1) == 0:
            assert(p_yx[0][1] >= delta)
            p_yx[0][0] += delta
            p_yx[1][0] = 1.0 - p_yx[0][0]
            p_yx[0][1] -= delta
            p_yx[1][1] = 1.0 - p_yx[0][1]
        else:
            assert(p_yx[0][1] <= 1.0 - delta)
            p_yx[0][0] -= delta
            p_yx[1][0] = 1.0 - p_yx[0][0]
            p_yx[0][1] += delta
            p_yx[1][1] = 1.0 - p_yx[0][1]

    def intoducePosteriorDrift6(self, p_yxx: List[List[List[float]]], delta: float, gamma: float, sign: float):
        if r.nextInt(0, 1) <= 0.5:
            p_yxx[0][0][0] += delta
            if p_yxx[0][0][0] > 1.0:
                p_yxx[0][0][0] = 1.0
            p_yxx[1][0][0] = 1.0 - p_yxx[0][0][0]
            p_yxx[0][0][1] -= delta
            if p_yxx[0][0][1] < 0.0:
                p_yxx[0][0][1] = 0.0
            p_yxx[1][0][1] = 1.0 - p_yxx[0][0][1]
        else:
            p_yxx[0][0][0] -= delta
            if p_yxx[0][0][0] < 0.0:
                p_yxx[0][0][0] = 0.0
            p_yxx[1][0][0] = 1.0 - p_yxx[0][0][0]
            p_yxx[0][0][1] += delta
            if p_yxx[0][0][1] > 1.0:
                p_yxx[0][0][1] = 1.0
            p_yxx[1][0][1] = 1.0 - p_yxx[0][0][1]

        if r.nextInt(0, 1) <= 0.5:
            p_yxx[0][1][0] += delta
            if p_yxx[0][1][0] > 1.0:
                p_yxx[0][1][0] = 1.0
            p_yxx[1][1][0] = 1.0 - p_yxx[0][1][0]
            p_yxx[0][1][1] -= delta
            if p_yxx[0][1][1] < 0.0:
                p_yxx[0][1][1] = 0.0
            p_yxx[1][1][1] = 1.0 - p_yxx[0][1][1]
        else:
            p_yxx[0][1][0] -= delta
            if p_yxx[0][1][0] < 0.0:
                p_yxx[0][1][0] = 0.0
            p_yxx[1][1][0] = 1.0 - p_yxx[0][1][0]
            p_yxx[0][1][1] += delta
            if p_yxx[0][1][1] > 1.0:
                p_yxx[0][1][1] = 1.0
            p_yxx[1][1][1] = 1.0 - p_yxx[0][1][1]
    def intoducePosteriorDrift5(self, p_yxx, delta, gamma, sign):
        if sign == 1:
            if p_yxx[0][0][0] <= delta:
                p_yxx[0][0][0] += delta
            if p_yxx[1][0][0] >= (1 - delta):
                p_yxx[1][0][0] -= delta
            if p_yxx[0][0][1] >= (1 - delta):
                p_yxx[0][0][1] -= delta
            if p_yxx[1][0][1] <= delta:
                p_yxx[1][0][1] += delta
            if p_yxx[0][1][0] <= delta:
                p_yxx[0][1][0] += gamma
            if p_yxx[1][1][0] >= (1 - delta):
                p_yxx[1][1][0] -= gamma
            if p_yxx[0][1][1] >= (1 - delta):
                p_yxx[0][1][1] -= gamma
            if p_yxx[1][1][1] <= delta:
                p_yxx[1][1][1] += gamma
        elif sign == -1:
            if p_yxx[0][0][0] >= (1 - delta):
                p_yxx[0][0][0] -= delta
            if p_yxx[1][0][0] <= delta:
                p_yxx[1][0][0] += delta
            if p_yxx[0][0][1] <= delta:
                p_yxx[0][0][1] += delta
            if p_yxx[1][0][1] >= (1 - delta):
                p_yxx[1][0][1] -= delta
            if p_yxx[0][1][0] >= (1 - delta):
                p_yxx[0][1][0] -= gamma
            if p_yxx[1][1][0] <= delta:
                p_yxx[1][1][0] += gamma
            if p_yxx[0][1][1] <= delta:
                p_yxx[0][1][1] += gamma
            if p_yxx[1][1][1] >= (1 - delta):
                p_yxx[1][1][1] -= gamma
    
    def intoducePosteriorDrift4(self, p_yxx, delta, gamma, sign):
        newval = p_yxx[0][0][0] + (sign * delta)
        if newval > 1:
            p_yxx[0][0][0] = 1
        elif newval < 0:
            p_yxx[0][0][0] = 0
        newval = p_yxx[1][0][0] - (sign * delta)
        if newval > 1:
            p_yxx[1][0][0] = 1
        elif newval < 0:
            p_yxx[1][0][0] = 0
        newval = p_yxx[0][0][1] - (sign * delta)
        if newval > 1:
            p_yxx[0][0][1] = 1
        elif newval < 0:
            p_yxx[0][0][1] = 0
        newval = p_yxx[1][0][1] + (sign * delta)
        if newval > 1:
            p_yxx[1][0][1] = 1
        elif newval < 0:
            p_yxx[1][0][1] = 0
        newval = p_yxx[0][1][0] + (sign * gamma)
        if newval > 1:
            p_yxx[0][1][0] = 1
        elif newval < 0:
            p_yxx[0][1][0] = 0
        newval = p_yxx[1][1][0] - (sign * gamma)
        if newval > 1:
            p_yxx[1][1][0] = 1
        elif newval < 0:
            p_yxx[1][1][0] = 0
        newval = p_yxx[0][1][1] - (sign * gamma)
        if newval > 1:
            p_yxx[0][1][1] = 1
        elif newval < 0:
            p_yxx[0][1][1] = 0
        newval = p_yxx[1][1][1] + (sign * gamma)
        if newval > 1:
            p_yxx[1][1][1] = 1
        elif newval < 0:
            p_yxx[1][1][1] = 0
    
    def intoducePosteriorDrift2(self, p_yxx, delta, gamma, sign):
        nvals = 2
        nc = 2
        for y in range(nc):
            for x1 in range(nvals):
                sum = 0
                for x2 in range(nvals):
                    p_yxx[y][x1][x2] = random.uniform(0, 1)
                    sum += p_yxx[y][x1][x2]
                for x2 in range(nvals):
                    p_yxx[y][x1][x2] /= sum
    
    def intoducePosteriorDrift2(self, p_yx, delta, gamma, sign):
        nvals = 2
        nc = 2
        for y in range(nc):
            sum = 0
            for x1 in range(nvals):
                p_yx[y][x1] = random.uniform(0, 1)
                sum += p_yx[y][x1]
            for x1 in range(nvals):
                p_yx[y][x1] /= sum
    
    def intoducePosteriorDrift3(self, p_yxx, delta, gamma, sign):
        p_yxx[0][0][0] += (sign * delta)
        p_yxx[1][0][0] -= (sign * delta)
        p_yxx[0][0][1] -= (sign * delta)
        p_yxx[1][0][1] += (sign * delta)
        p_yxx[0][1][0] += (sign * gamma)
        p_yxx[1][1][0] -= (sign * gamma)
        p_yxx[0][1][1] -= (sign * gamma)
        p_yxx[1][1][1] += (sign * gamma)
    
    def intoducePosteriorDrift3(self, p_yx, delta, sign):
        p_yx[0][0] += (sign * delta)
        p_yx[1][0] -= (sign * delta)
        p_yx[0][1] -= (sign * delta)
        p_yx[1][1] += (sign * delta)
    
    def prepareForUseImpl(self, monitor, repository):
        rg = JDKRandomGenerator()
        rg.setSeed(seed.getValue())
        self.r = RandomDataGenerator(rg)
        self.generateHeader()
        nc = Globals.getNumClasses()
        n = self.streamHeader.numAttributes() - 1
        nvals = 2
        self.parents = [None] * n
        self.parents[1] = 0
        for i in range(2, n):
            p = 0
            self.parents[i] = p
        print("Seed = " + seed.getValue())
        print(self.parents)
        self.p_y = [None] * nc
        self.p_yx = [[None] * nvals for _ in range(nc)]
        for y in range(nc):
            self.p_yx[y] = [None] * nvals
        self.p_yxx = [[[None] * nvals for _ in range(nvals)] for _ in range(n)]
        for i in range(1, n):
            for y in range(nc):
                self.p_yxx[i][y] = [[None] * nvals for _ in range(nvals)]
        self.d_yx = [None] * nc
        self.d_yxx = [[None] * nvals for _ in range(nc)]
        for y in range(nc):
            self.d_yxx[y] = [None] * nvals
        self.intializeCPTBinary()
        self.nInstancesGeneratedSoFar = 0L
    
    def intializeCPT_Constrained(self):
        nc = Globals.getNumClasses()
        n = self.streamHeader.numAttributes() - 1
        nvals = 2
        self.p_y[0] = random.uniform(self.delta, 1-self.delta)
        self.p_y[1] = 1 - self.p_y[0]
        for y in range(nc):
            self.p_yx[y][0] = random.uniform(self.delta, 1 - self.delta)
            self.p_yx[y][1] = 1 - self.p_yx[y][0]
        for i in range(1, n):
            for y in range(nc):
                for x1 in range(nvals):
                    self.p_yxx[i][y][x1][0] = random.uniform(self.delta, 1 - self.delta)
                    self.p_yxx[i][y][x1][1] = 1 - self.p_yxx[i][y][x1][0]
    
    def intializeCPT(self):
        nc = Globals.getNumClasses()
        n = self.streamHeader.numAttributes() - 1
        nvals = 2
        sum = 0
        for y in range(nc):
            self.p_y[y] = random.uniform(0, 1)
            sum += self.p_y[y]
        for y in range(nc):
            self.p_y[y] /= sum
        for y in range(nc):
            sum = 0
            for x1 in range(nvals):
                self.p_yx[y][x1] = random.uniform(0, 1)
                sum += self.p_yx[y][x1]
            for x1 in range(nvals):
                self.p_yx[y][x1] /= sum
        for i in range(1, n):
            for y in range(nc):
                for x1 in range(nvals):
                    sum = 0
                    for x2 in range(nvals):
                        self.p_yxx[i][y][x1][x2] = random.uniform(0, 1)
                        sum += self.p_yxx[i][y][x1][x2]
                    for x2 in range(nvals):
                        self.p_yxx[i][y][x1][x2] /= sum
    
    def intializeCPTBinary(self):
        nc = Globals.getNumClasses()
        n = self.streamHeader.numAttributes() - 1
        nvals = 2
        self.p_y[0] = random.uniform(0, 1)
        self.p_y[1] = 1.0 - self.p_y[0]
        for y in range(nc):
            self.p_yx[y][0] = random.uniform(0, 1)
            self.p_yx[y][1] = 1.0 - self.p_yx[y][0]
        for i in range(1, n):
            for y in range(nc):
                for x1 in range(nvals):
                    self.p_yxx[i][y][x1][0] = random.uniform(0, 1)
                    self.p_yxx[i][y][x1][1] = 1 - self.p_yxx[i][y][x1][0]
        self.d_yx[0] = random.randint(0, 1)
        self.d_yx[1] = random.randint(0, 1)
        self.d_yxx[0][1] = random.randint(0, 1)
        self.d_yxx[0][0] = random.randint(0, 1)
        self.d_yxx[1][1] = random.randint(0, 1)
        self.d_yxx[1][0] = random.randint(0, 1)
