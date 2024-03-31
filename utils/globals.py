class Globals:
    dataSetName = ""
    BUFFER_SIZE = 10*1024*1024
    ARFF_BUFFER_SIZE = 100000
    SOURCEFILE = None
    CVFILE = None
    cvFilePresent = False
    version = "v0.2"
    verbose = False
    experimentType = ""
    trainFile = ""
    testFile = ""
    cvFile = ""
    numExp = 5
    numFolds = 2
    model = "AnDE"
    level = 0
    doWanbiac = False
    doDiscriminative = False
    doSelectiveKDB = False
    doRegularization = False
    lambda = 0.01
    adaptiveRegularization = "None"
    doBinaryClassification = False
    objectiveFunction = "CLL"
    dataStructureParameter = "Flat"
    hashParameter = 60000000
    regularizationType = "L2"
    regularizationTowards = 0
    optimizer = "sgd"
    numIterations = 10
    sgdType = "Adagrad"
    sgdTuningParameter = 0.01
    doCrossValidateTuningParameter = False
    discretization = "None"
    normalizeNumeric = False
    discretizationParameter = 10
    holdoutsetPrecentage = 5
    mcmcType = "ALS"
    featureSelection = "None"
    featureSelectionParameter = 1.0
    numberInstances = 0
    numClasses = 2
    numAttributes = 0
    numRandAttributes = 3
    paramsPerAtt = None
    isNumericTrue = None
    CVInstances = None
    structure = None
    numInstancesKnown = False
    adaptiveControl = "None"
    adaptiveControlParameter = 0.1
    prequentialOutputResolution = 1
    doMovingAverage = False
    prequentialBufferOutputResolution = 100
    computeProbabilitiesFromCount = False
    tempDirectory = "/Users/nayyar/Desktop/AA/temp"
    ouputResultsDirectory = "/Users/nayyar/Desktop/AA/output"
    datasetRepository = "/Users/nayyar/WData/datasets_Decay/"
    flowParameter = "adaptiveControlParameter:{0,0.1,0.01,0.001}"
    driftType = "noDrift"
    generateDriftData = False
    totalNInstancesBeforeDrift = 0
    totalNInstancesDuringDrift = 10000
    totalNInstancesAfterDrift = 0
    driftNAttributes = 5
    driftNAttributesValues = 3
    driftMagnitude = 0.4
    driftMagnitude2 = 10
    driftMagnitude3 = 0.1
    discretizeOutOfCore = False
    preProcessParameter = "discretize"
    ignoreAttributes = ""
    classAttribute = -1
    attributeType = ""
    dicedPercentage = 10
    dicedStratified = False
    dicedAt = 0
    fsScore = None
    latentK = 10
    driftDelta = 0.1
    plotRMSEResuts = False

    @staticmethod
    def getVersion():
        return Globals.version

    @staticmethod
    def setVersion(version):
        Globals.version = version

    @staticmethod
    def isVerbose():
        return Globals.verbose

    @staticmethod
    def setVerbose(verbose):
        Globals.verbose = verbose

    @staticmethod
    def getExperimentType():
        return Globals.experimentType

    @staticmethod
    def setExperimentType(experimentType):
        Globals.experimentType = experimentType

    @staticmethod
    def getTrainFile():
        return Globals.trainFile

    @staticmethod
    def setTrainFile(trainFile):
        Globals.trainFile = trainFile

    @staticmethod
    def getTestFile():
        return Globals.testFile

    @staticmethod
    def setTestFile(testFile):
        Globals.testFile = testFile

    @staticmethod
    def getNumExp():
        return Globals.numExp

    @staticmethod
    def setNumExp(numExp):
        Globals.numExp = numExp

    @staticmethod
    def getNumFolds():
        return Globals.numFolds

    @staticmethod
    def setNumFolds(numFolds):
        Globals.numFolds = numFolds

    @staticmethod
    def getModel():
        return Globals.model

    @staticmethod
    def setModel(model):
        Globals.model = model

    @staticmethod
    def getLevel():
        return Globals.level

    @staticmethod
    def setLevel(level):
        Globals.level = level

    @staticmethod
    def isDoWanbiac():
        return Globals.doWanbiac

    @staticmethod
    def setDoWanbiac(doWanbiac):
        Globals.doWanbiac = doWanbiac

    @staticmethod
    def isDoSelectiveKDB():
        return Globals.doSelectiveKDB

    @staticmethod
    def setDoSelectiveKDB(doSelectiveKDB):
        Globals.doSelectiveKDB = doSelectiveKDB

    @staticmethod
    def isDoRegularization():
        return Globals.doRegularization

    @staticmethod
    def setDoRegularization(doRegularization):
        Globals.doRegularization = doRegularization

    @staticmethod
    def getLambda():
        return Globals.lambda

    @staticmethod
    def setLambda(lambda_):
        Globals.lambda = lambda_

    @staticmethod
    def getAdaptiveRegularization():
        return Globals.adaptiveRegularization

    @staticmethod
    def setAdaptiveRegularization(adaptiveRegularization):
        Globals.adaptiveRegularization = adaptiveRegularization

    @staticmethod
    def getOptimizer():
        return Globals.optimizer

    @staticmethod
    def setOptimizer(optimizer):
        Globals.optimizer = optimizer

    @staticmethod
    def getNumIterations():
        return Globals.numIterations

    @staticmethod
    def setNumIterations(numIterations):
        Globals.numIterations = numIterations

    @staticmethod
    def getSgdType():
        return Globals.sgdType

    @staticmethod
    def setSgdType(sgdType):
        Globals.sgdType = sgdType

    @staticmethod
    def getSgdTuningParameter():
        return Globals.sgdTuningParameter

    @staticmethod
    def setSgdTuningParameter(sgdTuningParameter):
        Globals.sgdTuningParameter = sgdTuningParameter

    @staticmethod
    def isDoCrossValidateTuningParameter():
        return Globals.doCrossValidateTuningParameter

    @staticmethod
    def setDoCrossValidateTuningParameter(doCrossValidateTuningParameter):
        Globals.doCrossValidateTuningParameter = doCrossValidateTuningParameter

    @staticmethod
    def getRegularizationType():
        return Globals.regularizationType

    @staticmethod
    def setRegularizationType(regularizationType):
        Globals.regularizationType = regularizationType

    @staticmethod
    def getRegularizationTowards():
        return Globals.regularizationTowards

    @staticmethod
    def setRegularizationTowards(regularizationTowards):
        Globals.regularizationTowards = regularizationTowards

    @staticmethod
    def getMcmcType():
        return Globals.mcmcType

    @staticmethod
    def setMcmcType(mcmcType):
        Globals.mcmcType = mcmcType

    @staticmethod
    def setDiscretization(discretization):
        Globals.discretization = discretization

    @staticmethod
    def isNormalizeNumeric():
        return Globals.normalizeNumeric

    @staticmethod
    def setNormalizeNumeric(normalizeNumeric):
        Globals.normalizeNumeric = normalizeNumeric

    @staticmethod
    def getDiscretizationParameter():
        return Globals.discretizationParameter

    @staticmethod
    def getDiscretization():
        return Globals.discretization

    @staticmethod
    def setDiscretizationParameter(discretizationParameter):
        Globals.discretizationParameter = discretizationParameter

    @staticmethod
    def getBUFFER_SIZE():
        return Globals.BUFFER_SIZE

    @staticmethod
    def setBUFFER_SIZE(BUFFER_SIZE):
        Globals.BUFFER_SIZE = BUFFER_SIZE

    @staticmethod
    def getARFF_BUFFER_SIZE():
        return Globals.ARFF_BUFFER_SIZE

    @staticmethod
    def setARFF_BUFFER_SIZE(ARFF_BUFFER_SIZE):
        Globals.ARFF_BUFFER_SIZE = ARFF_BUFFER_SIZE

    @staticmethod
    def getSOURCEFILE():
        return Globals.SOURCEFILE

    @staticmethod
    def setSOURCEFILE(SOURCEFILE):
        Globals.SOURCEFILE = SOURCEFILE

    @staticmethod
    def getHoldoutsetPrecentage():
        return Globals.holdoutsetPrecentage

    @staticmethod
    def setHoldoutsetPrecentage(holdoutsetPrecentage):
        Globals.holdoutsetPrecentage = holdoutsetPrecentage

    @staticmethod
    def getNumberInstances():
        return Globals.numberInstances

    @staticmethod
    def setNumberInstances(numberInstances):
        Globals.numberInstances = numberInstances

    @staticmethod
    def getNumClasses():
        return Globals.numClasses

    @staticmethod
    def setNumClasses(numClasses):
        Globals.numClasses = numClasses

    @staticmethod
    def getNumAttributes():
        return Globals.numAttributes

    @staticmethod
    def setNumAttributes(numAttributes):
        Globals.numAttributes = numAttributes

    @staticmethod
    def isDoBinaryClassification():
        return Globals.doBinaryClassification

    @staticmethod
    def setDoBinaryClassification(doBinaryClassification):
        Globals.doBinaryClassification = doBinaryClassification

    @staticmethod
    def getObjectiveFunction():
        return Globals.objectiveFunction

    @staticmethod
    def setObjectiveFunction(objectiveFunction):
        Globals.objectiveFunction = objectiveFunction

    @staticmethod
    def getDataStructureParameter():
        return Globals.dataStructureParameter

    @staticmethod
    def setDataStructureParameter(dataStructureParameter):
        Globals.dataStructureParameter = dataStructureParameter

    @staticmethod
    def getHashParameter():
        return Globals.hashParameter

    @staticmethod
    def setHashParameter(hashParameter):
        Globals.hashParameter = hashParameter

    @staticmethod
    def getParamsPerAtt():
        return Globals.paramsPerAtt

    @staticmethod
    def setParamsPerAtt(paramsPerAtt):
        Globals.paramsPerAtt = paramsPerAtt

    @staticmethod
    def getIsNumericTrue():
        return Globals.isNumericTrue

    @staticmethod
    def setIsNumericTrue(isNumericTrue):
        Globals.isNumericTrue = isNumericTrue

    @staticmethod
    def isDoDiscriminative():
        return Globals.doDiscriminative

    @staticmethod
    def setDoDiscriminative(doDiscriminative):
        Globals.doDiscriminative = doDiscriminative

    @staticmethod
    def getCVInstances():
        return Globals.CVInstances

    @staticmethod
    def setCVInstances(CVInstances):
        Globals.CVInstances = CVInstances

    @staticmethod
    def getStructure():
        return Globals.structure

    @staticmethod
    def setStructure(structure):
        Globals.structure = structure

    @staticmethod
    def getAdaptiveControl():
        return Globals.adaptiveControl

    @staticmethod
    def setAdaptiveControl(adaptiveControl):
        Globals.adaptiveControl = adaptiveControl

    @staticmethod
    def getAdaptiveControlParameter():
        return Globals.adaptiveControlParameter

    @staticmethod
    def setAdaptiveControlParameter(adaptiveControlParameter):
        Globals.adaptiveControlParameter = adaptiveControlParameter

    @staticmethod
    def printWelcomeMessage():
        msg = ""
        msg += "------------------------------------------------------------------------- \n"
        msg += " Welcome to Aquila Audax \n"
        msg += " Version: " + Globals.getVersion() + "\n\n"
        msg += " Library for learning from extremely large quantities of data in minimal  \n"
        msg += " number of passes through the data. \n"
        msg += " Salient features: \n"
        msg += "         1) Superior Feature Engineering Capability \n"
        msg += "         2) Fast Optimization \n"
        msg += "         3) Out-of-core data processing \n"
        msg += "\n"
        msg += " Type -help for information how to use the library \n\n"
        msg += " Copyrights DataSmelly Pvt Ltd \n"
        msg += "------------------------------------------------------------------------- \n"
        print(msg)

    @staticmethod
    def printWelcomeMessageWranglerini():
        msg = ""
        msg += "------------------------------------------------------------------------- \n"
        msg += " Invoking [Wranglerini] -- Data Wranglining Engine \n"
        msg += " Version: " + Globals.getVersion() + "\n\n"
        msg += " Copyrights DataSmelly Pvt Ltd \n"
        msg += "------------------------------------------------------------------------- \n"
        print(msg)

    @staticmethod
    def printWelcomeMessageRecommendica():
        msg = ""
        msg += "------------------------------------------------------------------------- \n"
        msg += " Invoking [Recommendica] -- Recommender Systems Engine \n"
        msg += " Version: " + Globals.getVersion() + "\n\n"
        msg += " Copyrights DataSmelly Pvt Ltd \n"
        msg += "------------------------------------------------------------------------- \n"
        print(msg)

    @staticmethod
    def printHelp():
        msg = ""
        msg += "------------------------------------------------------------------------- \n"
        msg += " java -jar AquilaAudax.jar  \n"
        msg += "         -typeofexperiment <traintest>  \n"
        msg += "               --trainfile  \n"
        msg += "               --testfile  \n"
        msg += "         -typeofexperiment <cv>  \n"
        msg += "               --trainfile  \n"
        msg += "         -typeofexperiment <prequential>  \n"
        msg += "         -typeofexperiment <recommender>  \n\n"
        msg += "         -numrounds  \n"
        msg += "         -numfolds  \n"
        msg += "------------------------------------------------------------------------- \n"
        print(msg)

    @staticmethod
    def isNumInstancesKnown():
        return Globals.numInstancesKnown

    @staticmethod
    def setNumInstancesKnown(numInstancesKnown):
        Globals.numInstancesKnown = numInstancesKnown

    @staticmethod
    def getPrequentialOutputResolution():
        return Globals.prequentialOutputResolution

    @staticmethod
    def setPrequentialOutputResolution(prequentialOutputResolution):
        Globals.prequentialOutputResolution = prequentialOutputResolution

    @staticmethod
    def isDoMovingAverage():
        return Globals.doMovingAverage

    @staticmethod
    def setDoMovingAverage(doMovingAverage):
        Globals.doMovingAverage = doMovingAverage

    @staticmethod
    def getPrequentialBufferOutputResolution():
        return Globals.prequentialBufferOutputResolution

    @staticmethod
    def setPrequentialBufferOutputResolution(prequentialBufferOutputResolution):
        Globals.prequentialBufferOutputResolution = prequentialBufferOutputResolution

    @staticmethod
    def isComputeProbabilitiesFromCount():
        return Globals.computeProbabilitiesFromCount

    @staticmethod
    def setComputeProbabilitiesFromCount(computeProbabilitiesFromCount):
        Globals.computeProbabilitiesFromCount = computeProbabilitiesFromCount

    @staticmethod
    def getTempDirectory():
        return Globals.tempDirectory

    @staticmethod
    def setTempDirectory(tempDirectory):
        Globals.tempDirectory = tempDirectory

    @staticmethod
    def getOuputResultsDirectory():
        return Globals.ouputResultsDirectory

    @staticmethod
    def setOuputResultsDirectory(ouputResultsDirectory):
        Globals.ouputResultsDirectory = ouputResultsDirectory

    @staticmethod
    def getFlowParameter():
        return Globals.flowParameter

    @staticmethod
    def setFlowParameter(flowParameter):
        Globals.flowParameter = flowParameter

    @staticmethod
    def isGenerateDriftData():
        return Globals.generateDriftData

    @staticmethod
    def setGenerateDriftData(generateDriftData):
        Globals.generateDriftData = generateDriftData

    @staticmethod
    def getTotalNInstancesBeforeDrift():
        return Globals.totalNInstancesBeforeDrift

    @staticmethod
    def setTotalNInstancesBeforeDrift(totalNInstancesBeforeDrift):
        Globals.totalNInstancesBeforeDrift = totalNInstancesBeforeDrift

    @staticmethod
    def getTotalNInstancesDuringDrift():
        return Globals.totalNInstancesDuringDrift

    @staticmethod
    def setTotalNInstancesDuringDrift(totalNInstancesDuringDrift):
        Globals.totalNInstancesDuringDrift = totalNInstancesDuringDrift

    @staticmethod
    def getTotalNInstancesAfterDrift():
        return Globals.totalNInstancesAfterDrift

    @staticmethod
    def setTotalNInstancesAfterDrift(totalNInstancesAfterDrift):
        Globals.totalNInstancesAfterDrift = totalNInstancesAfterDrift

    @staticmethod
    def getDriftNAttributes():
        return Globals.driftNAttributes

    @staticmethod
    def setDriftNAttributes(driftNAttributes):
        Globals.driftNAttributes = driftNAttributes

    @staticmethod
    def getDriftNAttributesValues():
        return Globals.driftNAttributesValues

    @staticmethod
    def setDriftNAttributesValues(driftNAttributesValues):
        Globals.driftNAttributesValues = driftNAttributesValues

    @staticmethod
    def getDriftMagnitude():
        return Globals.driftMagnitude

    @staticmethod
    def setDriftMagnitude(driftMagnitude):
        Globals.driftMagnitude = driftMagnitude

    @staticmethod
    def getFeatureSelection():
        return Globals.featureSelection

    @staticmethod
    def setFeatureSelection(featureSelection):
        Globals.featureSelection = featureSelection

    @staticmethod
    def getFeatureSelectionParameter():
        return Globals.featureSelectionParameter

    @staticmethod
    def setFeatureSelectionParameter(featureSelectionParameter):
        Globals.featureSelectionParameter = featureSelectionParameter

    @staticmethod
    def isDiscretizeOutOfCore():
        return Globals.discretizeOutOfCore

    @staticmethod
    def setDiscretizeOutOfCore(discretizeOutOfCore):
        Globals.discretizeOutOfCore = discretizeOutOfCore

    @staticmethod
    def getDataSetName():
        return Globals.dataSetName

    @staticmethod
    def setDataSetName(dataSetName):
        Globals.dataSetName = dataSetName

    @staticmethod
    def getPreProcessParameter():
        return Globals.preProcessParameter

    @staticmethod
    def setPreProcessParameter(preProcessParameter):
        Globals.preProcessParameter = preProcessParameter

    @staticmethod
    def getIgnoreAttributes():
        return Globals.ignoreAttributes

    @staticmethod
    def setIgnoreAttributes(ignoreAttributes):
        Globals.ignoreAttributes = ignoreAttributes

    @staticmethod
    def getClassAttribute():
        return Globals.classAttribute

    @staticmethod
    def setClassAttribute(classAttribute):
        Globals.classAttribute = classAttribute

    @staticmethod
    def getAttributeType():
        return Globals.attributeType

    @staticmethod
    def setAttributeType(attributeType):
        Globals.attributeType = attributeType

    @staticmethod
    def get_diced_percentage():
        return diced_percentage

    @staticmethod
    def set_diced_percentage(diced_percentage):
        Globals.diced_percentage = diced_percentage

    @staticmethod
    def is_diced_stratified():
        return diced_stratified

    @staticmethod
    def set_diced_stratified(diced_stratified):
        Globals.diced_stratified = diced_stratified

    @staticmethod
    def get_diced_at():
        return diced_at

    @staticmethod
    def set_diced_at(diced_at):
        Globals.diced_at = diced_at

    @staticmethod
    def get_fs_score():
        return fs_score

    @staticmethod
    def set_fs_score(fs_score):
        Globals.fs_score = fs_score

    @staticmethod
    def get_cv_file():
        return cv_file

    @staticmethod
    def set_cv_file(cv_file):
        Globals.cv_file = cv_file

    @staticmethod
    def get_CVFILE():
        return CVFILE

    @staticmethod
    def set_CVFILE(cVFILE):
        CVFILE = cVFILE

    @staticmethod
    def is_cv_file_present():
        return cv_file_present

    @staticmethod
    def set_cv_file_present(cv_file_present):
        Globals.cv_file_present = cv_file_present

    @staticmethod
    def get_drift_type():
        return drift_type

    @staticmethod
    def set_drift_type(drift_type):
        Globals.drift_type = drift_type

    @staticmethod
    def get_num_rand_attributes():
        return num_rand_attributes

    @staticmethod
    def set_num_rand_attributes(num_rand_attributes):
        Globals.num_rand_attributes = num_rand_attributes

    @staticmethod
    def get_latent_k():
        return latent_k

    @staticmethod
    def set_latent_k(latent_k):
        Globals.latent_k = latent_k

    @staticmethod
    def get_drift_delta():
        return drift_delta

    @staticmethod
    def set_drift_delta(drift_delta):
        Globals.drift_delta = drift_delta

    @staticmethod
    def get_drift_magnitude2():
        return drift_magnitude2

    @staticmethod
    def set_drift_magnitude2(drift_magnitude2):
        Globals.drift_magnitude2 = drift_magnitude2

    @staticmethod
    def get_drift_magnitude3():
        return drift_magnitude3

    @staticmethod
    def set_drift_magnitude3(drift_magnitude3):
        Globals.drift_magnitude3 = drift_magnitude3

    @staticmethod
    def is_plot_rmse_results():
        return plot_rmse_results

    @staticmethod
    def set_plot_rmse_results(plot_rmse_results):
        Globals.plot_rmse_results = plot_rmse_results

    @staticmethod
    def get_dataset_repository():
        return dataset_repository

    @staticmethod
    def set_dataset_repository(dataset_repository):
        Globals.dataset_repository = dataset_repository
