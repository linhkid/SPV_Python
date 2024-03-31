import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class PreprocessData:
    out = None
    res = None
    CVInstances = None
    
    @staticmethod
    def preProcessData():
        if Globals.isCvFilePresent():
            print("Loading CV File in memory")
            CVInstances = SUtils.getTrainTestInstances(Globals.getCVFILE())
            Globals.setCVInstances(CVInstances)
            print("CV file load successfully in memory")
            out = Globals.getSOURCEFILE()
        else:
            print("Start: Getting Stratified Indices")
            res = SUtils.getStratifiedIndices()
            print("Finish: Getting Stratified Indices")
            print("Start: Creating CV Instances in memory")
            CVInstances = SUtils.getTrainTestInstances(res)
            print("Finish: Creating CV Instances in memory")
            Globals.setCVInstances(CVInstances)
            out = os.path.join(tempfile.gettempdir(), "trainCV-.arff")
            PreprocessData.excludeCVInstances()
        
        if Globals.getDiscretization() != "None":
            PreprocessData.discreteNumeric()
        elif Globals.isNormalizeNumeric():
            PreprocessData.normalizeNumeric()
        
        if Globals.isVerbose():
            print("New Source File after Pre-processing is: " + out)
        
        return out
    
    @staticmethod
    def excludeCVInstances():
        print("Starting Exclusion of CV Instances")
        fileSaver = ArffSaver()
        fileSaver.setFile(out)
        fileSaver.setRetrieval(Saver.INCREMENTAL)
        reader = ArffReader(open(Globals.getSOURCEFILE(), "r"), Globals.getBUFFER_SIZE())
        structure = reader.getStructure()
        structure.setClassIndex(structure.numAttributes() - 1)
        fileSaver.setStructure(structure)
        i = 0
        lineNo = 0
        for row in reader:
            if not res.get(lineNo):
                fileSaver.writeIncremental(row)
                i += 1
            lineNo += 1
        fileSaver.writeIncremental(None)
        print("New Training set size =", i)
    
    @staticmethod
    def normalizeNumeric():
        print("Starting Normalization")
        m_NormalizedInstances = None
        m_Norm = StandardScaler()
        m_NormalizedInstances = m_Norm.fit_transform(Globals.getCVInstances())
        fileSaver = ArffSaver()
        fileSaver.setFile(out)
        fileSaver.setRetrieval(Saver.INCREMENTAL)
        fileSaver.setStructure(m_NormalizedInstances)
        reader = ArffReader(open(Globals.getSOURCEFILE(), "r"), Globals.getBUFFER_SIZE())
        structure = reader.getStructure()
        structure.setClassIndex(structure.numAttributes() - 1)
        i = 0
        lineNo = 0
        for row in reader:
            if not res.get(lineNo):
                row = m_Norm.transform(row)
                fileSaver.writeIncremental(row)
                i += 1
            lineNo += 1
        fileSaver.writeIncremental(None)
        print("New Training size =", i)
        Globals.setCVInstances(m_NormalizedInstances)
    
    @staticmethod
    def discreteNumeric():
        val = Globals.getDiscretization()
        s = Globals.getDiscretizationParameter()
        if val == "mdl":
            PreprocessData.discretizeMDL()
        elif val == "ef":
            PreprocessData.discretizeEF(s)
        elif val == "ew":
            PreprocessData.discretizeEW(s)
    
    @staticmethod
    def discretizeEW(s):
        pass
    
    @staticmethod
    def discretizeEF(s):
        numBins = s
        print("Starting Equal Frequency Discretization with", numBins, "bins.")
        m_Disc = KBinsDiscretizer(n_bins=numBins, encode='ordinal', strategy='quantile')
        m_DiscreteInstances = m_Disc.fit_transform(Globals.getCVInstances())
        fileSaver = ArffSaver()
        fileSaver.setFile(out)
        fileSaver.setRetrieval(Saver.INCREMENTAL)
        fileSaver.setStructure(m_DiscreteInstances)
        reader = ArffReader(open(Globals.getSOURCEFILE(), "r"), Globals.getBUFFER_SIZE())
        structure = reader.getStructure()
        structure.setClassIndex(structure.numAttributes() - 1)
        i = 0
        lineNo = 0
        for row in reader:
            if not res.get(lineNo):
                row = m_Disc.transform(row)
                fileSaver.writeIncremental(row)
                i += 1
            lineNo += 1
        fileSaver.writeIncremental(None)
        print("New Training size =", i)
        Globals.setCVInstances(m_DiscreteInstances)
    
    @staticmethod
    def discretizeMDL():
        print("Starting MDL Discretization")
        m_Disc = MDLPDiscretizer()
        m_DiscreteInstances = m_Disc.fit_transform(Globals.getCVInstances())
        fileSaver = ArffSaver()
        fileSaver.setFile(out)
        fileSaver.setRetrieval(Saver.INCREMENTAL)
        fileSaver.setStructure(m_DiscreteInstances)
        reader = ArffReader(open(Globals.getSOURCEFILE(), "r"), Globals.getBUFFER_SIZE())
        structure = reader.getStructure()
        structure.setClassIndex(structure.numAttributes() - 1)
        i = 0
        lineNo = 0
        for row in reader:
            if not res.get(lineNo):
                row = m_Disc.transform(row)
                fileSaver.writeIncremental(row)
                i += 1
            lineNo += 1
        fileSaver.writeIncremental(None)
        print("New Training size =", i)
        Globals.setCVInstances(m_DiscreteInstances)