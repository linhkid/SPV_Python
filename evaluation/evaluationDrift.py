import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

def learn():
    flowVal = determineFlowVal()
    if flowVal.lower() == "level":
        print("FlowVal used is: " + flowVal)
    else:
        print("Can only create flows with {adaptiveControlParameter} parameters.")
        exit(-1)
    flowValues = getFlowValues()
    if flowValues is None:
        print("Specify a proper range for flow")
        exit(-1)
    else:
        print("FlowVal = " + flowVal + ", Values = " + str(flowValues) + "\n")
    numFlows = len(flowValues)
    
    decayWindow_Rate = [20, 50, 500]
    
    learner = None
    structure = None
    sourceFile = None
    
    res = None
    resError = None
    nSamplesForChart = 0
    data = ""
    dw_name = ""
    driftMagnitude = 0
    numCycles = 1
    if isGenerateDriftData():
        nSamplesForChart = (getTotalNInstancesBeforeDrift() + getTotalNInstancesDuringDrift() + getTotalNInstancesAfterDrift()) / getPrequentialBufferOutputResolution()
        if (getTotalNInstancesBeforeDrift() + getTotalNInstancesDuringDrift() + getTotalNInstancesAfterDrift()) % getPrequentialBufferOutputResolution() != 0:
            nSamplesForChart += 1
        res = np.zeros((getNumExp(), len(decayWindow_Rate), numFlows, nSamplesForChart))
        resError = np.zeros((getNumExp(), len(decayWindow_Rate), numFlows, nSamplesForChart))
    else:
        data = getTrainFile()
        if data == "":
            print("evaluation: No Training File given")
            exit(-1)
        sourceFile = pd.read_csv(data)
        if sourceFile.empty:
            print("Train evaluation: File " + data + " not found!")
            exit(-1)
        if not isNumInstancesKnown():
            setNumberInstances(determineNumData())
        structure = setStructure()
        N = getNumberInstances()
        N *= numCycles
        nSamplesForChart = N / getPrequentialBufferOutputResolution()
        if N % getPrequentialBufferOutputResolution() != 0:
            nSamplesForChart += 1
        res = np.zeros((getNumExp(), len(decayWindow_Rate), numFlows, nSamplesForChart))
        resError = np.zeros((getNumExp(), len(decayWindow_Rate), numFlows, nSamplesForChart))
    
    for exp in range(getNumExp()):
        if isVerbose():
            print("-------------------------------------------------------------")
            print("Experiment No. " + str(exp))
            print("-------------------------------------------------------------")
        for d in range(len(decayWindow_Rate)):
            if decayWindow_Rate[d] == -1:
                setAdaptiveControl("None")
            else:
                setAdaptiveControl("window")
                dw_name = "window="
                setAdaptiveControlParameter(decayWindow_Rate[d])
            if isVerbose():
                print(" --------------> Decay. " + str(decayWindow_Rate[d]) + " <-------------- ")
            
            if isGenerateDriftData():
                print("Drift Type = " + getDriftType())
                driftMagnitude = getDriftMagnitude()
                print("Drfit Magnitude = " + str(driftMagnitude))
                if getDriftType().lower() == "simplest":
                    print("Calling TAN Drift generator")
                    sourceFile = generateTANDrift(exp, 0)
                elif getDriftType().lower() == "simplestkdb":
                    print("Calling KDB (K=2) Drift generator")
                    sourceFile = generateKDBDrift(exp, 0.0)
                elif getDriftType().lower() == "nodrift":
                    sourceFile = generateNoDrift(exp, 0.0)
                elif getDriftType().lower() == "gbayesian":
                    sourceFile = generateDriftGradualBayesian(exp, driftMagnitude)
                elif getDriftType().lower() == "glr":
                    sourceFile = generateDriftGradual(exp, driftMagnitude)
                elif getDriftType().lower() == "abrupt":
                    sourceFile = generateDriftData(exp, driftMagnitude)
                elif getDriftType().lower() == "withbayeserror":
                    sourceFile = generateDriftGradualSwappingGenerator(exp, driftMagnitude)
                numNoiseColumns = getNumRandAttributes()
                sourceFile = addNoise(numNoiseColumns, sourceFile)
            else:
                sourceFile = pd.read_csv(data)
                if sourceFile.empty:
                    print("Train evaluation: File " + data + " not found!")
                    exit(-1)
                setSOURCEFILE(sourceFile)
                if not isNumInstancesKnown():
                    setNumberInstances(determineNumData())
                sourceFile = generateSimpleDriftFromData(exp, numCycles)
            
            setSOURCEFILE(sourceFile)
            if not isNumInstancesKnown():
                setNumberInstances(determineNumData())
            structure = setStructure()
            N = getNumberInstances()
            nc = getNumClasses()
            print("<num data points, num classes> = <" + str(N) + ", " + str(nc) + ">")
            
            for f in range(numFlows):
                if isVerbose():
                    print("Level (" + getModel() + "): " + str(f))
                if flowVal.lower() == "adaptivecontrolparameter":
                    setAdaptiveControlParameter(flowValues[f])
                elif flowVal.lower() == "sgdtuningparameter":
                    setSgdTuningParameter(flowValues[f])
                elif flowVal.lower() == "lambda":
                    setLambda(flowValues[f])
                elif flowVal.lower() == "level":
                    setLevel(int(flowValues[f]))
                val = getModel()
                if val.lower() == "ande":
                    learner = ande()
                    learner.buildClassifier()
                reader = pd.read_csv(getSOURCEFILE(), chunksize=getBUFFER_SIZE())
                nErrors = 0
                totalTested = 0
                indexPlot = 0
                lineNo = 0
                rmse = 0
                for chunk in reader:
                    for i in range(len(chunk)):
                        row = chunk.iloc[i]
                        probs = learner.distributionForInstance(row)
                        results = getResults(probs, int(row['class']), len(probs))
                        if results[1] == 1:
                            nErrors += 1
                        totalTested += 1
                        rmse += results[0]
                        if lineNo % getPrequentialBufferOutputResolution() == 0:
                            res[exp][d][f][indexPlot] = 1.0 * rmse / totalTested
                            resError[exp][d][f][indexPlot] = 1.0 * nErrors / totalTested
                            indexPlot += 1
                            nErrors = 0
                            rmse = 0
                            totalTested = 0
                        learner.update(row)
                        lineNo += 1
                print("\nExp: " + str(exp) + ". Read: " + str(lineNo) + " data points, out of which learner was evaluated on: " + str(indexPlot))
                print()
            deleteSourceFile()
    
    minMean = np.full(numFlows, np.inf)
    meanErrorPlot = np.zeros((numFlows, len(decayWindow_Rate), nSamplesForChart))
    stdDevErrorPlot = np.zeros((numFlows, len(decayWindow_Rate), nSamplesForChart))
    meanErrorPlotError = np.zeros((numFlows, len(decayWindow_Rate), nSamplesForChart))
    stdDevErrorPlotError = np.zeros((numFlows, len(decayWindow_Rate), nSamplesForChart))
    for m in range(len(decayWindow_Rate)):
        for c in range(numFlows):
            for indexPlot in range(nSamplesForChart):
                for exp in range(getNumExp()):
                    meanErrorPlot[c][m][indexPlot] += res[exp][m][c][indexPlot]
                    meanErrorPlotError[c][m][indexPlot] += resError[exp][m][c][indexPlot]
                meanErrorPlot[c][m][indexPlot] /= getNumExp()
                meanErrorPlotError[c][m][indexPlot] /= getNumExp()
                for exp in range(getNumExp()):
                    diff1 = (res[exp][m][c][indexPlot] - meanErrorPlot[c][m][indexPlot])
                    diff2 = (resError[exp][m][c][indexPlot] - meanErrorPlotError[c][m][indexPlot])
                    stdDevErrorPlot[c][m][indexPlot] += diff1 * diff1
                    stdDevErrorPlotError[c][m][indexPlot] += diff2 * diff2
                stdDevErrorPlot[c][m][indexPlot] /= getNumExp()
                stdDevErrorPlotError[c][m][indexPlot] /= getNumExp()
                stdDevErrorPlot[c][m][indexPlot] = np.sqrt(stdDevErrorPlot[c][m][indexPlot])
                stdDevErrorPlotError[c][m][indexPlot] = np.sqrt(stdDevErrorPlotError[c][m][indexPlot])
                minMean[c] = min(minMean[c], meanErrorPlot[c][m][indexPlot])
    
    fileName = getOuputResultsDirectory()
    if not os.path.exists(fileName):
        os.makedirs(fileName)
    
    fileName += "-nExp=" + str(getNumExp())
    
    fileName += "-drfitMagnitude=" + str(getDriftMagnitude())
    fileName += "-drfitDelta=" + str(getDriftDelta())
    fileName += "-drfitMagnitude2=" + str(getDriftMagnitude2())
    fileName += "-drfitMagnitude3=" + str(getDriftMagnitude3())
    fileName += "-drfitType=" + str(getDriftType())
    
    fileName += "-" + str(int(time.time()))
    for c in range(numFlows):
        f1 = fileName + "-RMSE" + "_Level" + str(flowValues[c]) + ".results"
        f2 = fileName + "-01Loss" + "_Level" + str(flowValues[c]) + ".results"
        np.savetxt(f1, meanErrorPlot[c], delimiter=", ")
        np.savetxt(f2, meanErrorPlotError[c], delimiter=", ")
    
    res = None
    resError = None
    
    plot(fileName, numFlows, flowValues, driftMagnitude, decayWindow_Rate, dw_name, nSamplesForChart, meanErrorPlotError, stdDevErrorPlot, minMean)
    plot2(fileName, numFlows, flowValues, driftMagnitude, decayWindow_Rate, dw_name, nSamplesForChart, meanErrorPlotError, stdDevErrorPlot, minMean)


