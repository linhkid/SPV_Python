import moa.core.FastVector
import moa.core.InstanceExample
import moa.core.ObjectRepository
import moa.options.AbstractOptionHandler
import moa.streams.InstanceStream
import moa.tasks.TaskMonitor
import random
import numpy as np
from sklearn.utils import check_random_state

class SimpleTANGeneratorWithDriftBinaryValuesMod(DriftGenerator):
    def __init__(self):
        self.driftLength = IntOption("driftLength", 'l', "The number of instances the concept drift will be applied acrros", 0, 0, 10000000)
        self.frequency = IntOption("frequency", 'F', "The number of instances after which to change an entry in pygxbd", 0, 0, 100000)
        self.streamHeader = None
        self.p_y = None
        self.p_yx = None
        self.p_yxx = None
        self.d_yx = None
        self.d_yxx = None
        self.parents = None
        self.randAttributes = None
        self.r = None
        self.nInstancesGeneratedSoFar = None
        self.m_Q = None
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
        self.m_Q = int(Globals.getDriftMagnitude3() * (n-1))
        print("Out of " + str(n) + " attributes, " + str(self.m_Q + 1) + " of them will be order 1 and " + str(n - 1 - self.m_Q)  + " will be order 2")
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
                    p = random.randint(1, n-1)
                    if p not in self.randAttributes:
                        self.randAttributes[size] = p
                        size += 1
        for i in range(numDriftAttributes):
            p = self.randAttributes[i]
            if p == 0:
                pass
            else:
                if p < self.m_Q:
                    self.intoducePosteriorDrift(self.p_yx[p], self.d_yx[p], self.delta)
                else:
                    self.intoducePosteriorDrift(self.p_yxx[p], self.d_yxx[p], self.delta)
        inst = DenseInstance(self.streamHeader.numAttributes())
        inst.setDataset(self.streamHeader)
        nc = Globals.getNumClasses()
        y = random.randint(0, nc - 1)
        inst.setClassValue(y)
        x = [-1] * n
        x[0] = SUtils.sampleFromNonUniformDistribution(self.p_yx[0][y], self.r)
        inst.setValue(0, x[0])
        for i in range(1, n):
            p = self.parents[i]
            xp = x[p]
            if i < self.m_Q:
                x[i] = SUtils.sampleFromNonUniformDistribution(self.p_yx[i][y], self.r)
            else:
                x[i] = SUtils.sampleFromNonUniformDistribution(self.p_yxx[i][y][xp], self.r)
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
    
    def intoducePosteriorDrift8(self, p_yxx, delta):
        for p in range(2):
            for y in range(2):
                if p_yxx[y][p][0] < delta:
                    p_yxx[y][p][0] += delta
                    p_yxx[y][p][1] = 1.0 - p_yxx[y][p][0]
                elif p_yxx[y][p][0] > 1.0 - delta:
                    p_yxx[y][p][0] -= delta
                    p_yxx[y][p][1] = 1.0 - p_yxx[y][p][0]
                elif random.randint(0, 1) == 0:
                    p_yxx[y][p][0] += delta
                    p_yxx[y][p][1] = 1.0 - p_yxx[y][p][0]
                else:
                    p_yxx[y][p][0] -= delta
                    p_yxx[y][p][1] = 1.0 - p_yxx[y][p][0]
    
    def intoducePosteriorDrift8(self, p_yx, delta):
        for y in range(2):
            if p_yx[y][0] < delta:
                p_yx[y][0] += delta
                p_yx[y][1] = 1.0 - p_yx[y][0]
            elif p_yx[y][0] > 1.0 - delta:
                p_yx[y][0] -= delta
                p_yx[y][1] = 1.0 - p_yx[y][0]
            elif random.randint(0, 1) == 0:
                p_yx[y][0] += delta
                p_yx[y][1] = 1.0 - p_yx[y][0]
            else:
                p_yx[y][0] -= delta
                p_yx[y][1] = 1.0 - p_yx[y][0]
    
    def intoducePosteriorDrift7(self, p_yxx, delta):
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
        elif random.randint(0, 1) == 0:
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
        elif random.randint(0, 1) == 0:
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
    
    def intoducePosteriorDrift7(self, p_yx, delta):
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
        elif random.randint(0, 1) == 0:
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
    
    def intoducePosteriorDrift6(self, p_yxx, delta, gamma, sign):
        if random.randint(0, 1) <= 0.5:
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
        
        if random.randint(0, 1) <= 0.5:
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
                    p_yxx[y][x1][x2] = r.nextUniform(0, 1)
                    sum += p_yxx[y][x1][x2]
                for x2 in range(nvals):
                    p_yxx[y][x1][x2] /= sum

    def intoducePosteriorDrift2(self, p_yx, delta, gamma, sign):
        nvals = 2
        nc = 2
        for y in range(nc):
            sum = 0
            for x1 in range(nvals):
                p_yx[y][x1] = r.nextUniform(0, 1)
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
        r = RandomDataGenerator(rg)
        generateHeader()
        nc = Globals.getNumClasses()
        n = streamHeader.numAttributes() - 1
        nvals = 2
        parents = [0] * n
        parents[1] = 0
        for i in range(2, n):
            p = 0
            parents[i] = p
        print("Seed = " + seed.getValue())
        print(parents)
        p_y = [0] * nc
        p_yx = [[0] * nc for _ in range(nvals)]
        for y in range(nc):
            p_yx[0][y] = [0] * nvals
        for i in range(1, m_Q):
            p_yx[i] = [[0] * nc for _ in range(nvals)]
        p_yxx = [[[0] * nvals for _ in range(nvals)] for _ in range(n)]
        d_yx = [[0] * nc for _ in range(n)]
        for i in range(m_Q):
            d_yx[i] = [r.nextInt(0, 1) for _ in range(nc)]
        d_yxx = [[[r.nextInt(0, 1) for _ in range(nvals)] for _ in range(nc)] for _ in range(n)]
        intializeCPTBinary()
        nInstancesGeneratedSoFar = 0

    def intializeCPTBinary(self):
        nc = Globals.getNumClasses()
        n = streamHeader.numAttributes() - 1
        nvals = 2
        p_y[0] = r.nextUniform(0, 1)
        p_y[1] = 1.0 - p_y[0]
        for y in range(nc):
            p_yx[0][y][0] = r.nextUniform(0, 1)
            p_yx[0][y][1] = 1.0 - p_yx[0][y][0]
        for i in range(1, m_Q):
            for y in range(nc):
                p_yx[i][y][0] = r.nextUniform(0, 1)
                p_yx[i][y][1] = 1.0 - p_yx[i][y][0]
        for i in range(m_Q, n):
            for y in range(nc):
                for x1 in range(nvals):
                    p_yxx[i][y][x1][0] = r.nextUniform(0, 1)
                    p_yxx[i][y][x1][1] = 1 - p_yxx[i][y][x1][0]
        for i in range(m_Q):
            for y in range(nc):
                d_yx[i][y] = r.nextInt(0, 1)
                d_yx[i][y] = r.nextInt(0, 1)
        for i in range(m_Q, n):
            for y in range(nc):
                for x1 in range(nvals):
                    d_yxx[i][y][x1] = r.nextInt(0, 1)
                    d_yxx[i][y][x1] = r.nextInt(0, 1)


