class ande(Model):
    def __init__(self):
        self.m_NumTuples = 0
        self.dParameters_ = None
        self.structure = None
        self.sourceFile = None

    def buildClassifier(self):
        self.sourceFile = Globals.getSOURCEFILE()
        self.m_NumTuples = Globals.getLevel()
        if self.m_NumTuples < 0 or self.m_NumTuples > 2:
            print("AnDE is not implemented with level" + str(self.m_NumTuples))
            sys.exit(-1)
        print("[----- AnDE -----]: Level = " + str(self.m_NumTuples) + " Reading structure -- " + str(self.sourceFile))
        if Globals.getExperimentType().equalsIgnoreCase("prequential") or Globals.getExperimentType().equalsIgnoreCase("flowMachines") or Globals.getExperimentType().equalsIgnoreCase("drift"):
            val = Globals.getDataStructureParameter()
            if val.equalsIgnoreCase("Flat"):
                self.dParameters_ = wdAnDEParametersFlat()
            else:
                print("Prequential training of AnDE requires only Flat Parameter Structure")
                sys.exit(-1)
        else:
            val = Globals.getDataStructureParameter()
            if val.equalsIgnoreCase("Flat"):
                self.dParameters_ = wdAnDEParametersFlat()
            elif val.equalsIgnoreCase("IndexedBig"):
                self.dParameters_ = wdAnDEParametersIndexedBig()
            elif val.equalsIgnoreCase("BitMap"):
                pass
            elif val.equalsIgnoreCase("Hash"):
                pass
            reader = ArffReader(BufferedReader(FileReader(self.sourceFile), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE())
            self.structure = reader.getStructure()
            self.structure.setClassIndex(self.structure.numAttributes() - 1)
            row = reader.readInstance(self.structure)
            while row is not None:
                self.dParameters_.updateFirstPass(row)
                row = reader.readInstance(self.structure)
            if Globals.isVerbose():
                print("Finished first pass.")
            self.dParameters_.finishedFirstPass()
            if self.dParameters_.needSecondPass():
                reader = ArffReader(BufferedReader(FileReader(self.sourceFile), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE())
                self.structure = reader.getStructure()
                self.structure.setClassIndex(self.structure.numAttributes() - 1)
                row = reader.readInstance(self.structure)
                while row is not None:
                    self.dParameters_.updateAfterFirstPass(row)
                    row = reader.readInstance(self.structure)
                if Globals.isVerbose():
                    print("Finished second pass.")
            print("Finish training")

    def update(self, row):
        self.dParameters_.updateFirstPass(row)

    def predictA0DE(self, inst):
        nc = self.dParameters_.getNC()
        N = self.dParameters_.getN()
        n = self.dParameters_.getn()
        paramsPerAtt = self.dParameters_.getParamsPerAtt()
        probs = [0.0] * nc
        for c in range(len(probs)):
            probs[c] = math.log(SUtils.MEsti(self.dParameters_.getCountAtFullIndex(c), N, nc))
            for att1 in range(n):
                att1val = int(inst.value(att1))
                index = self.dParameters_.getAttributeIndex(att1, att1val, c)
                probs[c] += math.log(SUtils.MEsti(self.dParameters_.getCountAtFullIndex(int(index)), self.dParameters_.getCountAtFullIndex(c), paramsPerAtt[att1]))
        return probs

    def predictA1DE(self, inst):
        nc = self.dParameters_.getNC()
        N = self.dParameters_.getN()
        n = self.dParameters_.getn()
        paramsPerAtt = self.dParameters_.getParamsPerAtt()
        probs = [0.0] * nc
        probInitializerA1DE = sys.float_info.max / (n + 1)
        spodeProbs = [[0.0] * nc for _ in range(n)]
        parentCount = 0
        for up in range(n):
            x_up = int(inst.value(up))
            index = 0
            countOfX1AndY = 0
            for c in range(nc):
                index = self.dParameters_.getAttributeIndex(up, x_up, c)
                countOfX1AndY += self.dParameters_.getCountAtFullIndex(int(index))
            if countOfX1AndY > SUtils.m_Limit:
                parentCount += 1
                for c in range(nc):
                    index = self.dParameters_.getAttributeIndex(up, x_up, c)
                    spodeProbs[up][c] = probInitializerA1DE * SUtils.MEsti(self.dParameters_.getCountAtFullIndex(int(index)), N, paramsPerAtt[up] * nc)
        for up in range(1, n):
            x_up = int(inst.value(up))
            for uc in range(up):
                x_uc = int(inst.value(uc))
                for c in range(nc):
                    index1 = self.dParameters_.getAttributeIndex(up, x_up, uc, x_uc, c)
                    index2 = self.dParameters_.getAttributeIndex(uc, x_uc, c)
                    index3 = self.dParameters_.getAttributeIndex(up, x_up, c)
                    spodeProbs[uc][c] *= SUtils.MEsti(self.dParameters_.getCountAtFullIndex(int(index1)), self.dParameters_.getCountAtFullIndex(int(index2)), paramsPerAtt[up])
                    spodeProbs[up][c] *= SUtils.MEsti(self.dParameters_.getCountAtFullIndex(int(index1)), self.dParameters_.getCountAtFullIndex(int(index3)), paramsPerAtt[uc])
        for c in range(nc):
            for u in range(n):
                probs[c] += spodeProbs[u][c]
        SUtils.log(probs)
        return probs

    def predictA2DE(self, inst):
        nc = self.dParameters_.getNC()
        N = self.dParameters_.getN()
        n = self.dParameters_.getn()
        paramsPerAtt = self.dParameters_.getParamsPerAtt()
        probs = [0.0] * nc
        probInitializerA2DE = sys.float_info.max / ((n + 1) * (n + 1))
        spodeProbs = [[[0.0] * nc for _ in range(up2)] for up1 in range(1, n)]
        parentCount = 0
        for up1 in range(1, n):
            up2size = 0
            for up2 in range(up1):
                up2size += 1
            spodeProbs[up1] = [[0.0] * nc for _ in range(up2size)]
        for up1 in range(1, n):
            x_up1 = int(inst.value(up1))
            for up2 in range(up1):
                x_up2 = int(inst.value(up2))
                index = 0
                countOfX1AndX2AndY = 0
                for c in range(nc):
                    index = self.dParameters_.getAttributeIndex(up1, x_up1, up2, x_up2, c)
                    countOfX1AndX2AndY += self.dParameters_.getCountAtFullIndex(int(index))
                if countOfX1AndX2AndY >= SUtils.m_Limit:
                    parentCount += 1
                    for c in range(nc):
                        index = self.dParameters_.getAttributeIndex(up1, x_up1, up2, x_up2, c)
                        spodeProbs[up1][up2][c] = probInitializerA2DE * SUtils.MEsti(self.dParameters_.getCountAtFullIndex(int(index)), N, paramsPerAtt[up1] * paramsPerAtt[up2] * nc)
        for up1 in range(2, n):
            x_up1 = int(inst.value(up1))
            for up2 in range(1, up1):
                x_up2 = int(inst.value(up2))
                for uc in range(up2):
                    x_uc = int(inst.value(uc))
                    for c in range(nc):
                        index2 = self.dParameters_.getAttributeIndex(up1, x_up1, up2, x_up2, c)
                        index = self.dParameters_.getAttributeIndex(up1, x_up1, up2, x_up2, uc, x_uc, c)
                        parentFreq = self.dParameters_.getCountAtFullIndex(int(index))
                        index3 = self.dParameters_.getAttributeIndex(up2, x_up2, uc, x_uc, c)
                        index4 = self.dParameters_.getAttributeIndex(up1, x_up1, uc, x_uc, c)
                        spodeProbs[up1][up2][c] *= SUtils.MEsti(parentFreq, self.dParameters_.getCountAtFullIndex(int(index2)), paramsPerAtt[uc])
                        spodeProbs[up2][uc][c] *= SUtils.MEsti(parentFreq, self.dParameters_.getCountAtFullIndex(int(index3)), paramsPerAtt[up1])
                        spodeProbs[up1][uc][c] *= SUtils.MEsti(parentFreq, self.dParameters_.getCountAtFullIndex(int(index4)), paramsPerAtt[up2])
        for c in range(nc):
            for up1 in range(1, n):
                for up2 in range(up1):
                    probs[c] += spodeProbs[up1][up2][c]
        SUtils.log(probs)
        return probs

    def distributionForInstance(self, inst):
        probs = None
        if self.m_NumTuples == 0:
            probs = self.predictA0DE(inst)
        elif self.m_NumTuples == 1:
            probs = self.predictA1DE(inst)
        elif self.m_NumTuples == 2:
            probs = self.predictA2DE(inst)
        SUtils.normalizeInLogDomain(probs)
        SUtils.exp(probs)
        return probs

    def evaluateFunction(self, sourceFile):
        return 0

    def predict(self, inst):
        return None

    def computeGrad(self, inst, probs, x_C):
        pass

    def computeGradAndUpdateParameters(self, instance, probs, x_C):
        pass

    def evaluateFunction(self, cvInstances):
        return None
