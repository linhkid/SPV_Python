import os
import numpy as np
from sklearn.metrics import mean_squared_error

class evaluationExternal:
    def __init__(self):
        self.df = "{:.####}"

    def learn(self):
        option = "generateAddNoiseSplit_Run"
        if option.lower() == "runlibfm":
            pass
        elif option.lower() == "generateaddnoisesplit_run":
            sourceFile = None
            sourceFileTrain = None
            sourceFileTest = None
            m_RMSE = 0
            m_Error = 0
            NTest = 0
            learner = None
            result = np.zeros(Globals.getNumExp())
            for exp in range(Globals.getNumExp()):
                if Globals.getDriftType().lower() == "simplest":
                    print("Calling TAN Drift generator")
                    sourceFile = Sampler.generateTANDrift(exp, 0)
                elif Globals.getDriftType().lower() == "simplestkdb":
                    print("Calling KDB (K=2) Drift generator")
                    sourceFile = Sampler.generateKDBDrift(exp, 0.0)
                numNoiseColumns = Globals.getNumRandAttributes()
                sourceFile = SUtils.addNoise(numNoiseColumns, sourceFile)
                Globals.setExperimentType("preProcess")
                Globals.setPreProcessParameter("Dice")
                Globals.setDicedPercentage(50)
                Globals.setDicedStratified(False)
                Globals.setTrainFile(sourceFile.getAbsolutePath())
                Globals.setDataSetName("temp"+str(exp))
                evaluationPreprocess.learn()
                sourceFileTrain = os.path.join(Globals.getTempDirectory(), "temp"+str(exp)+"_Train.arff")
                sourceFileTest = os.path.join(Globals.getTempDirectory(), "temp"+str(exp)+"_Test.arff")
                val = Globals.getModel()
                if val.lower() == "ande":
                    learner = ande()
                Globals.setSOURCEFILE(sourceFileTrain)
                if not Globals.isNumInstancesKnown():
                    Globals.setNumberInstances(SUtils.determineNumData())
                structure = SUtils.setStructure()
                N = int(Globals.getNumberInstances())
                nc = Globals.getNumClasses()
                learner.buildClassifier()
                reader = ArffReader(open(sourceFileTest, "r"), Globals.getBUFFER_SIZE(), Globals.getARFF_BUFFER_SIZE())
                current = None
                for current in reader:
                    probs = learner.distributionForInstance(current)
                    x_C = int(current.classValue())
                    pred = -1
                    bestProb = float("-inf")
                    for y in range(nc):
                        if not np.isnan(probs[y]):
                            if probs[y] > bestProb:
                                pred = y
                                bestProb = probs[y]
                            m_RMSE += (1 / nc * (probs[y] - (1 if y == x_C else 0)) ** 2)
                        else:
                            print("probs[{}] is NaN! oh no!".format(y))
                    if pred != x_C:
                        m_Error += 1
                    NTest += 1
                result[exp] = m_Error / NTest
            print("\nTrain Test Experimentation")
            print("\nClassifier       : {} (K = {})".format(Globals.getModel(), Globals.getLevel()))
            print("\nSourceFile Train : {}".format(sourceFileTrain))
            print("\nSourceFile Test  : {}".format(sourceFileTest))
            print("\nError            : {}".format(self.df.format(m_Error / NTest)))
            print("\nRMSE             : {}".format(self.df.format(np.sqrt(m_RMSE / NTest))))
            print("\n\n\n")