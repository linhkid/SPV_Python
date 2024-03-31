import sys
import os
import random
import numpy as np
import pandas as pd

class evaluationPrequential:
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
        
        if Globals.isDoCrossValidateTuningParameter():
            sourceFile = PreprocessData.preProcessData()
            Globals.setSOURCEFILE(sourceFile)
        
        if not Globals.isNumInstancesKnown():
            Globals.setNumberInstances(SUtils.determineNumData())
        
        structure = SUtils.setStructure()
        N = int(Globals.getNumberInstances())
        nc = Globals.getNumClasses()
        print("<num data points, num classes> = <" + str(N) + ", " + str(nc) + ">")
        learner = None
        val = Globals.getModel()
        
        rmseResults = []
        numDataEvaluated = 0
        seed = 3071980
        learner = None
        for exp in range(Globals.getNumExp()):
            sourceFile = SUtils.randomizeTrainingFile()
            Globals.setSOURCEFILE(sourceFile)
            rmseExpResults = []
            if Globals.isVerbose():
                print("Experiment No. " + str(exp))
            seed += 1
            val = Globals.getModel()
            if val.equalsIgnoreCase("AnDE"):
                learner = ande()
                learner.buildClassifier()
            
            reader = ArffReader(open(Globals.getSOURCEFILE()), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE())
            row = None
            lineNo = 0
            numDataEvaluated = 0
            N = 0
            while True:
                row = reader.readInstance(structure)
                if row is None:
                    break
                if lineNo % Globals.getPrequentialOutputResolution() == 0:
                    probs = learner.distributionForInstance(row)
                    results = SUtils.getResults(probs, int(row.classValue()), nc)
                    rmseExpResults.append(results[0])
                    numDataEvaluated += 1
                learner.update(row)
                lineNo += 1
                N += 1
                Globals.setNumberInstances(N)
            
            print("\nExp: " + str(exp) + ". Read: " + str(lineNo) + " data points, out of which learner was evaluated on: " + str(numDataEvaluated))
            rmseResults.append(rmseExpResults)
        
        averageResults = np.zeros(numDataEvaluated)
        for exp in range(Globals.getNumExp()):
            for i in range(numDataEvaluated):
                averageResults[i] += (1/Globals.getNumExp() * rmseResults[exp][i])
        
        buffAverageResults = []
        bufferSize = Globals.getPrequentialBufferOutputResolution()
        if Globals.isDoMovingAverage():
            for i in range(numDataEvaluated):
                if i < bufferSize:
                    pass
                else:
                    tempSum = 0
                    for j in range(i, i + bufferSize):
                        if j >= numDataEvaluated:
                            break
                        tempSum += averageResults[j]
                    buffAverageResults.append((1/bufferSize * tempSum))
                    i += 1
        else:
            for i in range(numDataEvaluated):
                if i < bufferSize:
                    pass
                else:
                    tempSum = 0
                    for j in range(i, i + bufferSize):
                        if j >= numDataEvaluated:
                            break
                        tempSum += averageResults[j]
                    buffAverageResults.append((1/bufferSize * tempSum))
                    i += bufferSize
        
        identifier = Globals.getModel() + "_" + Globals.getLevel()
        if Globals.getModel().equalsIgnoreCase("ALR"):
            if Globals.isDoWanbiac():
                identifier += "w"
            if Globals.isDoDiscriminative():
                identifier += "d"
        
        outputFile = Globals.getOuputResultsDirectory() + identifier + "_LC.m"
        file = open(outputFile, "w")
        file.write("fx_" + identifier + " = [")
        for i in range(len(buffAverageResults) - 1):
            file.write(str(buffAverageResults[i]) + ", ")
        file.write("];")
        file.close()