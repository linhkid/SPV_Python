import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, log_loss

class evaluationCV:
    @staticmethod
    def learn():
        data = Globals.getTrainFile()
        if data == "":
            sys.stderr.write("evaluation: No Training File given")
            sys.exit(-1)
        
        sourceFile = os.path.abspath(data)
        if not os.path.exists(sourceFile):
            sys.stderr.write("evaluation: File " + data + " not found!")
            sys.exit(-1)
        
        if Globals.isVerbose():
            print("Initial Source File is at: " + sourceFile)
        
        Globals.setSOURCEFILE(sourceFile)
        
        if (Globals.getDiscretization() != "None") or Globals.isNormalizeNumeric() or Globals.isDoCrossValidateTuningParameter():
            sourceFile = PreprocessData.preProcessData()
            Globals.setSOURCEFILE(sourceFile)
        
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
        seed = 3071980
        instanceProbs = np.zeros((N, nc))
        trainTime = 0
        testTime = 0
        trainStart = 0
        testStart = 0
        learner = None
        
        for exp in range(Globals.getNumExp()):
            if Globals.isVerbose():
                print("Experiment No. " + str(exp))
            
            seed += 1
            indexes = [np.bitwise.BitArray() for _ in range(Globals.getNumFolds())]
            SUtils.getIndexes(indexes)
            
            for fold in range(Globals.getNumFolds()):
                if Globals.isVerbose():
                    print("Fold No. " + str(fold))
                
                trainIndexes = SUtils.combineIndexes(indexes, fold)
                trainFile = SUtils.createTrainTmpFile(structure, trainIndexes)
                print("Train file generated")
                
                if Globals.isVerbose():
                    print("Training fold " + str(fold) + ": trainFile is '" + trainFile.getAbsolutePath() + "'")
                
                val = Globals.getModel()
                if val.equalsIgnoreCase("AnDE"):
                    learner = ande()
                
                trainStart = time.time()
                Globals.setSOURCEFILE(trainFile)
                learner.buildClassifier()
                Globals.setSOURCEFILE(sourceFile)
                trainTime += (time.time() - trainStart)
                
                if Globals.isVerbose():
                    print("Testing fold 0 started")
                
                testStart = time.time()
                thisNTest = 0
                lineNo = 0
                reader = ArffReader(open(Globals.getSOURCEFILE()), Globals.getBUFFER_SIZE())
                current = None
                
                for current in reader:
                    if not trainIndexes[lineNo]:
                        probs = learner.distributionForInstance(current)
                        pred = -1
                        bestProb = float("-inf")
                        
                        for y in range(nc):
                            if not np.isnan(probs[y]):
                                if probs[y] > bestProb:
                                    pred = y
                                    bestProb = probs[y]
                                
                                m_RMSE += (1 / nc * ((probs[y] - 1) if y == current.classValue() else probs[y]) ** 2)
                            else:
                                sys.stderr.write("probs[ " + str(y) + "] is NaN! oh no!")
                        
                        if pred != current.classValue():
                            m_Error += 1
                        
                        m_LogLoss += np.log(probs[current.classValue()])
                        thisNTest += 1
                        NTest += 1
                        instanceProbs[lineNo][pred] += 1
                    
                    lineNo += 1
                
                testTime += time.time() - testStart
                
                if Globals.isVerbose():
                    print("Testing fold " + str(fold) + " finished - 0-1=" + str(m_Error / NTest) + "\trmse=" + str(np.sqrt(m_RMSE / NTest)) + "\tlogloss=" + str(m_LogLoss / (nc * NTest)))
        
        m_Bias = 0
        m_Sigma = 0
        m_Variance = 0
        lineNo = 0
        reader = ArffReader(open(Globals.getSOURCEFILE()), Globals.getBUFFER_SIZE())
        
        for current in reader:
            predProbs = instanceProbs[lineNo]
            pActual, pPred = 0, 0
            bsum, vsum, ssum = 0, 0, 0
            
            for j in range(nc):
                pActual = 1 if current.classValue() == j else 0
                pPred = predProbs[j] / Globals.getNumExp()
                bsum += (pActual - pPred) * (pActual - pPred) - pPred * (1 - pPred) / (Globals.getNumExp() - 1)
                vsum += (pPred * pPred)
                ssum += pActual * pActual
            
            m_Bias += bsum
            m_Variance += (1 - vsum)
            m_Sigma += (1 - ssum)
            lineNo += 1
        
        m_Bias = m_Bias / (2 * lineNo)
        m_Variance = (m_Error / NTest) - m_Bias
        
        print("\nBias-Variance Decomposition")
        print("\nClassifier       : " + Globals.getModel() + " (K = " + Globals.getLevel() + ")")
        print("\nData File   : " + data)
        print("\nError                 : " + str(m_Error / NTest))
        print("\nBias^2              : " + str(m_Bias))
        print("\nVariance           : " + str(m_Variance))
        print("\nRMSE               : " + str(np.sqrt(m_RMSE / NTest)))
        print("\nLogLoss           : " + str(m_LogLoss / (nc * NTest)))
        print("\nTraining Time   : " + str(trainTime / 1000))
        print("\nTesting Time    : " + str(testTime / 1000))
        print("\n\n\n")