import os
import sys
import math
import random
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def learn():
    datasets = ["powersupply.arff", "elecNormNew.arff", "airlines.arff", "sensor.arff"]
    for dataset in datasets:
        dataName = dataset.split(".")[0]
        decays = [0.5, 0.1, 0.01, 0.001, 0.0001]
        numDecays = len(decays)
        windows = [10, 20, 50, 100, 500, 1000]
        numWindows = len(windows)
        learnersDecayNB = [None] * numDecays
        learnersDecayA1DE = [None] * numDecays
        learnersDecayA2DE = [None] * numDecays
        resDecay = np.zeros((len(datasets), numDecays, 3))
        learnersWindowNB = [None] * numWindows
        learnersWindowA1DE = [None] * numWindows
        learnersWindowA2DE = [None] * numWindows
        
        for i in range(numDecays):
            stream = pd.read_csv(dataName)
            filter = IDADiscretizer(5, 1000, IDAType.IDAW)
            filter.setInputStream(stream)
            filter.init()
            filter.prepareForUse()
            Globals.setAdaptiveControl("decay")
            Globals.setAdaptiveControlParameter(decays[i])
            level = 0
            Globals.setLevel(level)
            learnersDecayNB[i] = ande()
            error = evaluateLearner(learnersDecayNB[i], filter)
            resDecay[d][i][level] = error
            level = 1
            Globals.setLevel(level)
            learnersDecayA1DE[i] = ande()
            error = evaluateLearner(learnersDecayA1DE[i], filter)
            resDecay[d][i][level] = error
            level = 2
            Globals.setLevel(level)
            learnersDecayA2DE[i] = ande()
            error = evaluateLearner(learnersDecayA2DE[i], filter)
            resDecay[d][i][level] = error
        
        for i in range(numDecays):
            stream = pd.read_csv(dataName)
            filter = IDADiscretizer(5, 1000, IDAType.IDAW)
            filter.setInputStream(stream)
            filter.init()
            filter.prepareForUse()
            Globals.setAdaptiveControl("window")
            Globals.setAdaptiveControlParameter(windows[i])
            level = 0
            Globals.setLevel(level)
            learnersWindowNB[i] = ande()
            error = evaluateLearner(learnersWindowNB[i], filter)
            resDecay[d][i][level] = error
            level = 1
            Globals.setLevel(level)
            learnersWindowA1DE[i] = ande()
            error = evaluateLearner(learnersWindowA1DE[i], filter)
            resDecay[d][i][level] = error
            level = 2
            Globals.setLevel(level)
            learnersWindowA2DE[i] = ande()
            error = evaluateLearner(learnersWindowA2DE[i], filter)
            resDecay[d][i][level] = error

def evaluateLearner(learner, filter):
    error = 0
    nErrors = 0
    lineNo = 0
    while filter.hasMoreInstances():
        row = filter.nextInstance().getData()
        probs = learner.distributionForInstance(row)
        results = SUtils.getResults(probs, row.classValue(), len(probs))
        if results[1] == 1:
            nErrors += 1
        learner.update(row)
        lineNo += 1
    error = nErrors / lineNo
    return error


