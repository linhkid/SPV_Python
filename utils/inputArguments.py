import argparse

class InputArguments:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-help", "--help", help="print this message", action="store_true")
        self.parser.add_argument("-version", "--version", help="print the version information and exit", action="store_true")
        self.parser.add_argument("-verbose", "--verbose", help="be extra verbose", action="store_true")
        self.parser.add_argument("-experimentType", "--experimentType", help="Experiment Type: <cv, traintest, prequential, recommender>")
        self.parser.add_argument("-trainFile", "--trainFile", help="Specify valid ARFF File for Training")
        self.parser.add_argument("-testFile", "--testFile", help="Specify valid ARFF File for Testing")
        self.parser.add_argument("-cvFile", "--cvFile", help="Specify valid ARFF File for Testing")
        self.parser.add_argument("-arffBufferSize", "--arffBufferSize", help="Arff Buffer Size")
        self.parser.add_argument("-bufferSize", "--bufferSize", help="Buffer Size")
        self.parser.add_argument("-numExp", "--numExp", help="Number of Experiments to Run")
        self.parser.add_argument("-numFolds", "--numFolds", help="Number of Folds to train in case of cv Experiments")
        self.parser.add_argument("-discretization", "--discretization", help="Discretize Data: <none, mdl, ef, ew>")
        self.parser.add_argument("-discretizationParameter", "--discretizationParameter", help="Input parameter for Discretization")
        self.parser.add_argument("-normalizeNumeric", "--normalizeNumeric", help="Normalize Numeric Attributes", action="store_true")
        self.parser.add_argument("-holdoutsetPrecentage", "--holdoutsetPrecentage", help="Percentage of Hold-out data, default is 5%")
        self.parser.add_argument("-model", "--model", help="Model: <ALR, AnJE, KDB, FM, ANN, AnDE>")
        self.parser.add_argument("-level", "--level", help="Level of Model")
        self.parser.add_argument("-doWanbiac", "--doWanbiac", help="Use WANBIA-C trick", action="store_true")
        self.parser.add_argument("-doDiscriminative", "--doDiscriminative", help="Do Discrimnative Learning (SGD or MCMC)", action="store_true")
        self.parser.add_argument("-doSelectiveKDB", "--doSelectiveKDB", help="Do Selective KDB", action="store_true")
        self.parser.add_argument("-doRegularization", "--doRegularization", help="Do Regularization", action="store_true")
        self.parser.add_argument("-lambda", "--lambda", help="Lambda value for regularization")
        self.parser.add_argument("-adaptiveRegularization", "--adaptiveRegularization", help="Adaptive Regularization: <None, Rendle, Provost, Zaidi1, Zaidi2>")
        self.parser.add_argument("-regularizationType", "--regularizationType", help="Regularization type: <L2>")
        self.parser.add_argument("-regularizationTowards", "--regularizationTowards", help="Regularization towards a value")
        self.parser.add_argument("-optimizer", "--optimizer", help="Model: <SGD, MCMC>")
        self.parser.add_argument("-numIterations", "--numIterations", help="No. of Iterations")
        self.parser.add_argument("-sgdType", "--sgdType", help="SGD type: <plainsgd, adagrad, adadelta, nplr>")
        self.parser.add_argument("-sgdTuningParameter", "--sgdTuningParameter", help="Specify ONE Tuning Parameter for SGD")
        self.parser.add_argument("-doCrossValidateTuningParameter", "--doCrossValidateTuningParameter", help="Cross-validate SGD tuning Parameter", action="store_true")
        self.parser.add_argument("-mcmcType", "--mcmcType", help="MCMC type: <ALS>")
        self.parser.add_argument("-featureSelection", "--featureSelection", help="Feature Selection: <None, MI, Count, ChiSqTest, GTest, FisherExactTest, AnJETest, AnJETestLOOCV, ALRTest>")
        self.parser.add_argument("-featureSelectionParameter", "--featureSelectionParameter", help="Specify ONE Parameter for Feature Selection")
        self.parser.add_argument("-objectiveFunction", "--objectiveFunction", help="Objective Function: <CLL, MSE, HL>")
        self.parser.add_argument("-doBinaryClassification", "--doBinaryClassification", help="Do 1vsAll Binary Classification", action="store_true")
        self.parser.add_argument("-dataStructureParameter", "--dataStructureParameter", help="Parameter Structure: <Flat, IndexedBig, BitMap, Hash>")
        self.parser.add_argument("-hashParameter", "--hashParameter", help="Hash Parameter (length of hashed vector)")
        self.parser.add_argument("-adaptiveControl", "--adaptiveControl", help="Adaptive Control: <None, Decay, Window>")
        self.parser.add_argument("-adaptiveControlParameter", "--adaptiveControlParameter", help="Parameter of Adaptive Control")
        self.parser.add_argument("-prequentialOutputResolution", "--prequentialOutputResolution", help="Parameter Controlling output resolution of Prequential Plots")
        self.parser.add_argument("-prequentialBufferOutputResolution", "--prequentialBufferOutputResolution", help="Parameter Controlling output buffer resolution of Prequential Plots")
        self.parser.add_argument("-doMovingAverage", "--doMovingAverage", help="Plot Prequential Curves with Moving averages", action="store_true")
        self.parser.add_argument("-tempDirectory", "--tempDirectory", help="Directory to store tmp files. Default: /tmp/")
        self.parser.add_argument("-ouputResultsDirectory", "--ouputResultsDirectory", help="Directory to store Result files. Default: /tmp/")
        self.parser.add_argument("-driftType", "--driftType", help="Drift Type. E.g., None, Bayesian, LR, Abrupt")
        self.parser.add_argument("-flowParameter", "--flowParameter", help="Flow Parameters. E.g. adaptiveControl:Decay: {0.2,0.4} or adaptiveControl:Window: {10,20}")
        self.parser.add_argument("-driftMagnitude", "--driftMagnitude", help="Drfit Magnitude")
        self.parser.add_argument("-driftMagnitude2", "--driftMagnitude2", help="Drfit Magnitude2")
        self.parser.add_argument("-driftMagnitude3", "--driftMagnitude3", help="Drfit Magnitude2")
        self.parser.add_argument("-driftDelta", "--driftDelta", help="driftDelta")
        self.parser.add_argument("-driftNAttributes", "--driftNAttributes", help="Drfit Number of Attributes")
        self.parser.add_argument("-driftNAttributesValues", "--driftNAttributesValues", help="Drfit Number of Attributes Values")
        self.parser.add_argument("-totalNInstancesDuringDrift", "--totalNInstancesDuringDrift", help="Drift duration in terms of no. of data points")
        self.parser.add_argument("-discretizeOutOfCore", "--discretizeOutOfCore", help="Discretize Out of core", action="store_true")
        self.parser.add_argument("-preProcessParameter", "--preProcessParameter", help="preProcessParameter: <Explore, CreateHeader, Discretize, Slice, Normalize, MissingImputate>")
        self.parser.add_argument("-ignoreAttributes", "--ignoreAttributes", help="ignoreAttributes. {3,5,10}")
        self.parser.add_argument("-classAttribute", "--classAttribute", help="classAttribute. E.g 2 (default is the last parameter")
        self.parser.add_argument("-attributeType", "--attributeType", help="attributeType. E.g {1,0,0,0,0,1,1}")
        self.parser.add_argument("-isGenerateDriftData", "--isGenerateDriftData", help="Generate Drift Data or work on provided file", action="store_true")
        self.parser.add_argument("-numClasses", "--numClasses", help="Number of Classes")
        self.parser.add_argument("-numRandAttributes", "--numRandAttributes", help="Number of Random Attributes")
        self.parser.add_argument("-dicedStratified", "--dicedStratified", help="Do a stratified sampling (Dicing)", action="store_true")
        self.parser.add_argument("-dicedPercentage", "--dicedPercentage", help="Drfit Percentage (0.2, 0.5, etc.)")
        self.parser.add_argument("-dicedAt", "--dicedAt", help="1000, 2000, etc")
        self.parser.add_argument("-latentK", "--latentK", help="latentK. E.g {10, 100, 500}")
        self.parser.add_argument("-plotRMSEResuts", "--plotRMSEResuts", help="Plot RMSE Results", action="store_true")

    def setOptions(self, args):
        args = self.parser.parse_args(args)
        if args.help:
            self.printHelp()
            exit(0)
        if args.verbose:
            Globals.setVerbose(True)
        if args.normalizeNumeric:
            Globals.setNormalizeNumeric(True)
        if args.discretization:
            Globals.setDiscretization(args.discretization)
        if args.discretizationParameter:
            Globals.setDiscretizationParameter(int(args.discretizationParameter))
        if args.holdoutsetPrecentage:
            Globals.setHoldoutsetPrecentage(int(args.holdoutsetPrecentage))
        if args.experimentType:
            Globals.setExperimentType(args.experimentType)
        if args.trainFile:
            Globals.setTrainFile(args.trainFile)
        if args.testFile:
            Globals.setTestFile(args.testFile)
        if args.cvFile:
            Globals.setCvFile(args.cvFile)
        if args.numExp:
            Globals.setNumExp(int(args.numExp))
        if args.arffBufferSize:
            Globals.setARFF_BUFFER_SIZE(int(args.arffBufferSize))
        if args.bufferSize:
            Globals.setBUFFER_SIZE(int(args.bufferSize))
        if args.numFolds:
            Globals.setNumFolds(int(args.numFolds))
        if args.model:
            Globals.setModel(args.model)
        if args.level:
            Globals.setLevel(int(args.level))
        if args.doWanbiac:
            Globals.setDoWanbiac(True)
        if args.doSelectiveKDB:
            Globals.setDoSelectiveKDB(True)
        if args.doRegularization:
            Globals.setDoRegularization(True)
        if args.lambda:
            Globals.setLambda(float(args.lambda))
        if args.adaptiveRegularization:
            Globals.setAdaptiveRegularization(args.adaptiveRegularization)
        if args.regularizationType:
            Globals.setRegularizationType(args.regularizationType)
        if args.regularizationTowards:
            Globals.setRegularizationTowards(float(args.regularizationTowards))
        if args.optimizer:
            Globals.setOptimizer(args.optimizer)
        if args.numIterations:
            Globals.setNumIterations(int(args.numIterations))
        if args.sgdType:
            Globals.setSgdType(args.sgdType)
        if args.sgdTuningParameter:
            Globals.setSgdTuningParameter(float(args.sgdTuningParameter))
        if args.doCrossValidateTuningParameter:
            Globals.setDoCrossValidateTuningParameter(True)
        if args.mcmcType:
            Globals.setMcmcType(args.mcmcType)
        if args.featureSelection:
            Globals.setFeatureSelection(args.featureSelection)
        if args.featureSelectionParameter:
            Globals.setFeatureSelectionParameter(float(args.featureSelectionParameter))
        if args.doBinaryClassification:
            Globals.setDoBinaryClassification(True)
        if args.objectiveFunction:
            Globals.setObjectiveFunction(args.objectiveFunction)
        if args.dataStructureParameter:
            Globals.setDataStructureParameter(args.dataStructureParameter)
        if args.hashParameter:
            Globals.setHashParameter(int(args.hashParameter))
        if args.doDiscriminative:
            Globals.setDoDiscriminative(True)
        if args.adaptiveControl:
            Globals.setAdaptiveControl(args.adaptiveControl)
        if args.adaptiveControlParameter:
            Globals.setAdaptiveControlParameter(float(args.adaptiveControlParameter))
        if args.prequentialOutputResolution:
            Globals.setPrequentialOutputResolution(int(args.prequentialOutputResolution))
        if args.doMovingAverage:
            Globals.setDoMovingAverage(True)
        if args.prequentialBufferOutputResolution:
            Globals.setPrequentialBufferOutputResolution(int(args.prequentialBufferOutputResolution))
        if args.tempDirectory:
            Globals.setTempDirectory(args.tempDirectory)
        if args.ouputResultsDirectory:
            Globals.setOuputResultsDirectory(args.ouputResultsDirectory)
        if args.flowParameter:
            Globals.setFlowParameter(args.flowParameter)
        if args.driftType:
            Globals.setDriftType(args.driftType)
        if args.isGenerateDriftData:
            Globals.setGenerateDriftData(True)
        if args.driftMagnitude:
            Globals.setDriftMagnitude(args.driftMagnitude)
        if args.driftMagnitude2:
            Globals.setDriftMagnitude2(args.driftMagnitude2)
        if args.driftMagnitude3:
            Globals.setDriftMagnitude3(args.driftMagnitude3)
        if args.driftDelta:
            Globals.setDriftDelta(args.driftDelta)
        if args.driftNAttributes:
            Globals.setDriftNAttributes(args.driftNAttributes)
        if args.driftNAttributesValues:
            Globals.setDriftNAttributesValues(args.driftNAttributesValues)
        if args.totalNInstancesDuringDrift:
            Globals.setTotalNInstancesDuringDrift(args.totalNInstancesDuringDrift)
        if args.discretizeOutOfCore:
            Globals.setDiscretizeOutOfCore(True)
        if args.preProcessParameter:
            Globals.setPreProcessParameter(args.preProcessParameter)
        if args.ignoreAttributes:
            Globals.setIgnoreAttributes(args.ignoreAttributes)
        if args.classAttribute:
            Globals.setClassAttribute(args.classAttribute)
        if args.attributeType:
            Globals.setAttributeType(args.attributeType)
        if args.dicedStratified:
            Globals.setDicedStratified(True)
        if args.dicedAt:
            Globals.setDicedAt(args.dicedAt)
        if args.dicedPercentage:
            Globals.setDicedPercentage(args.dicedPercentage)
        if args.numClasses:
            Globals.setNumClasses(args.numClasses)
        if args.numRandAttributes:
            Globals.setNumRandAttributes(args.numRandAttributes)
        if args.latentK:
            Globals.setLatentK(args.latentK)
        if args.plotRMSEResuts:
            Globals.setPlotRMSEResuts(True)

    def getOptions(self):
        pass


