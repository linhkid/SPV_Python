import sys
import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class EvaluationFlowMachines:
    @staticmethod
    def learn():
        flowVal = SanctityCheck.determineFlowVal()
        if flowVal.lower() not in ["adaptivecontrolparameter", "sgdtuningparameter", "lambda"]:
            print("Can only create flows with {adaptiveControlParameter, sgdTuningParameter, lambda} parameters.")
            sys.exit(-1)
        flowValues = SanctityCheck.getFlowValues()
        numFlows = len(flowValues)
        if flowValues is None:
            print("Specify a proper range for flow")
            sys.exit(-1)
        else:
            print("FlowVal = " + flowVal + ", Values = " + str(flowValues))
        
        data = Globals.getTrainFile()
        if data == "":
            print("evaluation: No Training File given")
            sys.exit(-1)
        sourceFile = None
        N = 0
        structure = None
        
        sourceFile = data
        if not os.path.exists(sourceFile):
            print("evaluation: File " + data + " not found!")
            sys.exit(-1)
        if Globals.isVerbose():
            print("Initial Source File is at: " + sourceFile)
        Globals.setSOURCEFILE(sourceFile)
        if not Globals.isNumInstancesKnown():
            Globals.setNumberInstances(SUtils.determineNumData())
        structure = SUtils.setStructure()
        N = int(Globals.getNumberInstances())
        nc = Globals.getNumClasses()
        print("<num data points, num classes> = <" + str(N) + ", " + str(nc) + ">")
        rmseExpFlowResults = []
        learners = None
        numDataEvaluated = 0
        
        for exp in range(Globals.getNumExp()):
            rmseExpResults = []
            sourceFile = SUtils.randomizeTrainingFile()
            Globals.setSOURCEFILE(sourceFile)
            if Globals.isVerbose():
                print("Experiment No. " + str(exp))
            val = Globals.getModel()
            if val.lower() == "ande":
                learners = [ande() for _ in range(numFlows)]
                for f in range(numFlows):
                    learners[f].buildClassifier()
            elif val.lower() == "alr":
                pass
            elif val.lower() == "kdb":
                pass
            elif val.lower() == "fm":
                pass
            elif val.lower() == "ann":
                pass
            reader = ArffReader(open(Globals.getSOURCEFILE()), Globals.getBUFFER_SIZE())
            row = None
            lineNo = 0
            N = 0
            numDataEvaluated = 0
            while True:
                row = reader.readInstance(structure)
                if row is None:
                    break
                if lineNo % Globals.getPrequentialOutputResolution() == 0:
                    rmseResults = []
                    for f in range(numFlows):
                        probs = learners[f].distributionForInstance(row)
                        results = SUtils.getResults(probs, int(row.classValue()), nc)
                        rmseResults.append(results[0])
                    rmseExpResults.append(rmseResults)
                    numDataEvaluated += 1
                for f in range(numFlows):
                    if flowVal.lower() == "adaptivecontrolparameter":
                        Globals.setAdaptiveControlParameter(flowValues[f])
                    elif flowVal.lower() == "sgdtuningparameter":
                        Globals.setSgdTuningParameter(flowValues[f])
                    elif flowVal.lower() == "lambda":
                        Globals.setLambda(flowValues[f])
                    learners[f].update(row)
                lineNo += 1
                N += 1
                Globals.setNumberInstances(N)
            print("\nExp: " + str(exp) + ". Read: " + str(lineNo) + " data points, out of which learner was evaluated on: " + str(numDataEvaluated))
            rmseExpFlowResults.append(rmseExpResults)
        
        averageResults = np.zeros((numDataEvaluated, numFlows))
        for exp in range(Globals.getNumExp()):
            for i in range(numDataEvaluated):
                for f in range(numFlows):
                    averageResults[i][f] += (1/Globals.getNumExp() * rmseExpFlowResults[exp][i][f])
        
        if Globals.isDoMovingAverage():
            pass
        else:
            buffAverageResults = []
            bufferSize = Globals.getPrequentialBufferOutputResolution()
            for i in range(numDataEvaluated):
                flowAverageResults = []
                for f in range(numFlows):
                    if i < bufferSize:
                        pass
                    else:
                        tempSum = 0
                        for j in range(i, i + bufferSize):
                            if j >= numDataEvaluated:
                                break
                            tempSum += averageResults[j][f]
                        flowAverageResults.append((1/bufferSize * tempSum))
                        i += bufferSize
                buffAverageResults.append(flowAverageResults)
            identifier = Globals.getModel() + "_" + Globals.getLevel()
            outputFile = Globals.getOuputResultsDirectory() + identifier + "_LC.m"
            with open(outputFile, "w") as output:
                for f in range(numFlows):
                    output.write("fx_" + identifier + "_f_" + str(flowValues[f]) + " = [")
                    for i in range(len(buffAverageResults) - 1):
                        output.write(str(buffAverageResults[i][f]) + ", ")
                    output.write("];\n")