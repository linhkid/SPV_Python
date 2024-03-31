import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

class evaluationTrainTest:
    @staticmethod
    def learn():
        data = Globals.getTrainFile()
        if data == "":
            sys.stderr.write("evaluation: No Training File given\n")
            sys.exit(-1)
        sourceFileTrain = data
        if not os.path.exists(sourceFileTrain):
            sys.stderr.write("Train evaluation: File " + data + " not found!\n")
            sys.exit(-1)
        if Globals.isVerbose():
            print("Training Source File is at: " + os.path.abspath(sourceFileTrain))
        data = Globals.getTestFile()
        if data == "":
            sys.stderr.write("evaluation: No Testing File given\n")
            sys.exit(-1)
        sourceFileTest = data
        if not os.path.exists(sourceFileTest):
            sys.stderr.write("Test evaluation: File " + data + " not found!\n")
            sys.exit(-1)
        if Globals.isVerbose():
            print("Testing Source File is at: " + os.path.abspath(sourceFileTest))
        data = Globals.getCvFile()
        sourceFileCV = data
        if not os.path.exists(sourceFileCV):
            print("CV evaluation: File " + data + " not found!")
        else:
            Globals.setCvFilePresent(True)
        if Globals.isVerbose():
            print("CV Source File is at: " + os.path.abspath(sourceFileCV))
        Globals.setCVFILE(sourceFileCV)
        Globals.setSOURCEFILE(sourceFileTrain)
        if Globals.getDiscretization().lower() != "none" or Globals.isNormalizeNumeric() or Globals.isDoCrossValidateTuningParameter():
            sourceFileTrain = PreprocessData.preProcessData()
            Globals.setSOURCEFILE(sourceFileTrain)
        if not Globals.isNumInstancesKnown():
            Globals.setNumberInstances(SUtils.determineNumData())
        structure = SUtils.setStructure()
        N = int(Globals.getNumberInstances())
        nc = Globals.getNumClasses()
        print("<num data points, num classes> = <" + str(N) + ", " + str(nc) + ">")
        m_RMSE = 0
        m_Error = 0
        m_LogLoss = 0
        NTest = 0
        instanceProbs = np.zeros((N, nc))
        trainTime = 0
        testTime = 0
        trainStart = 0
        testStart = 0
        learner = None
        val = Globals.getModel()
        if val.lower() == "ande":
            learner = ande()
        trainStart = time.time()
        learner.buildClassifier()
        trainTime += (time.time() - trainStart)
        testStart = time.time()
        lineNo = 0
        reader = ArffReader(open(sourceFileTest, "r"), Globals.getBUFFER_SIZE())
        current = None
        while True:
            current = reader.readInstance(structure)
            if current is None:
                break
            probs = learner.distributionForInstance(current)
            pred = np.argmax(probs)
            x_C = int(current.classValue())
            m_RMSE += (1 / nc * np.power((probs[x_C] - ((x_C == pred) ? 1 : 0)), 2))
            if pred != x_C:
                m_Error += 1
            m_LogLoss += np.log(probs[x_C])
            NTest += 1
            lineNo += 1
        testTime += (time.time() - testStart)
        print("\nTrain Test Experimentation")
        print("\nClassifier\t: " + Globals.getModel() + " (K = " + Globals.getLevel() + ")")
        print("\nData File\t: " + data)
        print("\nError\t\t: " + str(m_Error / NTest))
        print("\nRMSE\t\t: " + str(np.sqrt(m_RMSE / NTest)))
        print("\nLogLoss\t\t: " + str(m_LogLoss / (nc * NTest)))
        print("\nTraining Time\t: " + str(trainTime))
        print("\nTesting Time\t: " + str(testTime))
        print("\n\n\n")