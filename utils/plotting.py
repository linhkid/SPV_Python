import matplotlib.pyplot as plt
import numpy as np

def plot(fileName, numFlows, level, driftMagnitude, decayWindow_Rate, dw_name, nSamplesForChart, meanErrorPlot, stdDevErrorPlot, minMean):
    for c in range(numFlows):
        learnername = ""
        learnername = Globals.getModel() + " - " + str(level[c]) + " - Drift (" + str(Globals.getDriftMagnitude()) + ", " + str(Globals.getDriftMagnitude2()) + "," + str(Globals.getDriftMagnitude3()) + ")"
        
        dataSet = []
        dataSetErrors = []
        for m in range(len(decayWindow_Rate)):
            series = []
            seriesError = []
            for indexPlot in range(nSamplesForChart):
                trueNInstancesForIndex = 1 + indexPlot * Globals.getPrequentialBufferOutputResolution()
                mean = meanErrorPlot[c][m][indexPlot]
                low = mean - stdDevErrorPlot[c][m][indexPlot]
                high = mean + stdDevErrorPlot[c][m][indexPlot]
                seriesError.append([trueNInstancesForIndex, trueNInstancesForIndex, trueNInstancesForIndex, mean, low, high])
                series.append([trueNInstancesForIndex, mean])
            
            dataSet.append(series)
            dataSetErrors.append(seriesError)
        
        fig, ax = plt.subplots()
        ax.errorbar(x=np.array(dataSetErrors), y=np.array(dataSet), fmt='o', capsize=3)
        ax.set_xlabel('error-rate')
        ax.set_ylabel('nInstances')
        ax.set_title(learnername)
        ax.grid(False)
        
        filename = fileName + "_Level_" + str(level[c]) + ".plot"
        plt.savefig(filename + "-mean.pdf")
        plt.close()
        
def plot2(fileName, numFlows, level, driftMagnitude, decayWindow_Rate, dw_name, nSamplesForChart, meanErrorPlot, stdDevErrorPlot, minMean):
    learnername = ""
    learnername = "Drift (" + str(Globals.getDriftMagnitude()) + ", " + str(Globals.getDriftMagnitude2()) + "," + str(Globals.getDriftMagnitude3()) + ")"
    fileName += ".plot"
    dataSet = []
    for c in range(numFlows):
        for m in range(len(decayWindow_Rate)):
            series = []
            for indexPlot in range(nSamplesForChart):
                trueNInstancesForIndex = 1 + indexPlot * Globals.getPrequentialBufferOutputResolution()
                mean = meanErrorPlot[c][m][indexPlot]
                series.append([trueNInstancesForIndex, mean])
            
            dataSet.append(series)
        
        fig, ax = plt.subplots()
        ax.plot(np.array(dataSet))
        ax.set_xlabel('nInstances')
        ax.set_ylabel('error-rate')
        ax.set_title(learnername)
        ax.grid(False)
        
        plt.savefig(fileName + "-mean.pdf")
        plt.close()


