def printExperimentInformation():
    val = Globals.getExperimentType()
    msg = ""
    msg += "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ \n"
    msg += "Experiment type: " + val + "\n"
    msg += "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ \n"
    if val.lower() == "cv":
        msg += getCVExperimentinfo()
    elif val.lower() == "traintest":
        pass
    elif val.lower() == "recommender":
        pass
    elif val.lower() == "prequential":
        msg += getPrequentialExperimentinfo()
    elif val.lower() == "flowmachines":
        pass
    elif val.lower() == "drift":
        pass
    elif val.lower() == "preprocess":
        pass
    elif val.lower() == "external":
        pass
    elif val.lower() == "moa":
        pass
    msg += "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ \n"
    msg += "That is all what AquilaAudax needs to train this model -- Ignoring all other flags \n"
    msg += "Sanctity Check of input arguments is now complete. \n\n"
    print(msg)

def getPrequentialExperimentinfo():
    msg = ""
    msg += "Num Exp = " + Globals.getNumExp() + "\n"
    val = Globals.getModel()
    if val.lower() == "ande":
        msg += getAnDEModeinfo()
    elif val.lower() == "alr":
        msg += getALRModeinfo()
    elif val.lower() == "halr":
        msg += getHALRModeinfo()
    elif val.lower() == "kdb":
        pass
    elif val.lower() == "fm":
        pass
    elif val.lower() == "ann":
        pass
    msg += "Adaptive Control = " + Globals.getAdaptiveControl() + " \n"
    msg += "Adaptive Control parameter = " + Globals.getAdaptiveControlParameter()  + " \n"
    return msg

def getCVExperimentinfo():
    msg = ""
    msg += "Num Exp = " + Globals.getNumExp() + ", Num of Folds = " + Globals.getNumFolds() + "\n"
    val = Globals.getModel()
    if val.lower() == "ande":
        msg += getAnDEModeinfo()
    elif val.lower() == "alr":
        msg += getALRModeinfo()
    elif val.lower() == "halr":
        msg += getHALRModeinfo()
    elif val.lower() == "fealr":
        msg += getFEALRModeinfo()
    elif val.lower() == "kdb":
        pass
    elif val.lower() == "fm":
        pass
    elif val.lower() == "ann":
        pass
    return msg

def getALRModeinfo():
    msg = ""
    msg += "ALR -- Level (n) = " + Globals.getLevel() + "\n"
    msg += "Parameter Structure = " + Globals.getDataStructureParameter() + "\n"
    val = Globals.getOptimizer()
    if val.lower() == "none":
        msg += "Optimizer = None: Training a Generative Classifier (also known as AnJE) \n"
    elif val.lower() == "sgd":
        msg += "Do WANBIA-C trick = " + Globals.isDoWanbiac() + "\n"
        msg += getSGDOptimizereinfo()
    elif val.lower() == "mcmc":
        msg += getMCMCOptimizereinfo()
    return msg

def getHALRModeinfo():
    msg = ""
    msg += "hALR -- Level (n) = " + Globals.getLevel() + "\n"
    msg += "Parameter Structure = " + Globals.getDataStructureParameter() + "\n"
    val = Globals.getOptimizer()
    if val.lower() == "none":
        msg += "Optimizer = None: Training a Generative Classifier (also known as AnJE) \n"
    elif val.lower() == "sgd":
        msg += "Do WANBIA-C trick = " + Globals.isDoWanbiac() + "\n"
        msg += getSGDOptimizereinfo()
    elif val.lower() == "mcmc":
        msg += getMCMCOptimizereinfo()
    return msg

def getFEALRModeinfo():
    msg = ""
    msg += "feALR -- Level to (n) = " + Globals.getLevel() + "\n"
    msg += "Parameter Structure = " + Globals.getDataStructureParameter() + "\n"
    val = Globals.getOptimizer()
    if val.lower() == "none":
        msg += "Optimizer = None: Training a Generative Classifier (also known as AnJE) \n"
    elif val.lower() == "sgd":
        msg += "Do WANBIA-C trick = " + Globals.isDoWanbiac() + "\n"
        msg += getSGDOptimizereinfo()
    elif val.lower() == "mcmc":
        msg += getMCMCOptimizereinfo()
    return msg

def getAnDEModeinfo():
    msg = ""
    msg += "AnDE -- Level (n) = " + Globals.getLevel() + "\n"
    msg += "Parameter Structure = " + Globals.getDataStructureParameter() + "\n"
    return msg

def getMCMCOptimizereinfo():
    msg = ""
    msg += "MCMC optimizer type = " + Globals.getMcmcType() + "\n"
    return msg

def getSGDOptimizereinfo():
    msg = ""
    if Globals.isDoDiscriminative():
        msg += "SGD type = " + Globals.getSgdType() + "\n"
        msg += "Regularization = " + Globals.isDoRegularization() + "\n"
        if Globals.isDoRegularization(): 
            msg += "Regularization Type = " +  0 + "\n"
            msg += "Regularization Towards = " +  0 + "\n"
            msg += "Lambda = " + Globals.getLambda() + "\n"
            msg += "Adaptive Regularization = " + Globals.getAdaptiveRegularization() + "\n"
        msg += "Tuning Parameter = " + Globals.getSgdTuningParameter() + "\n"
        msg += "Cross-validate Tuning Parameter = " + Globals.isDoCrossValidateTuningParameter() + "\n"
    return msg

def checkStringArgs():
    if checkExperimentType() and checkModel() and checkOptimizer() and checkSGDType() and checkAdaptiveRegularization() and checkMCMCType() and checkRegularizationType() and checkDiscretizationType() and checkObjectiveFunctionType() and checkStructureParameter() and checkAdaptiveControl() and checkFeatureSelection() and checkPreProcessParameterType():
        return True
    else:
        return False

def checkFeatureSelection():
    flag = False
    val = Globals.getFeatureSelection()
    if val.lower() == "none" or val.lower() == "count" or val.lower() == "mi" or val.lower() == "chisqtest" or val.lower() == "gtest" or val.lower() == "fisherexacttest" or val.lower() == "anjetest" or val.lower() == "anjetestloocv" or val.lower() == "alrtest":
        flag = True
    else:
        flag = False
        print("-featureSelection takes values from {None, Count, MI, ChiSqTest, GTest, FisherExactTest, AnJETest, AnJETestLOOCV, ALRTest}")
    return flag

def checkAdaptiveControl():
    flag = False
    val = Globals.getAdaptiveControl()
    if val.lower() == "none" or val.lower() == "decay" or val.lower() == "window":
        flag = True
    else:
        flag = False
        print("-adaptivecontrol takes values from {None, Decay, Window}")
    return flag

def checkObjectiveFunctionType():
    flag = False
    val = Globals.getObjectiveFunction()
    if val.lower() == "cll" or val.lower() == "mse" or val.lower() == "hl":
        flag = True
    else:
        flag = False
        print("-objectiveFunction takes values from {CLL, MSE, HLL}")
    return flag

def checkStructureParameter():
    flag = False
    val = Globals.getDataStructureParameter()
    if val.lower() == "flat" or val.lower() == "indexedbig" or val.lower() == "bitmap" or val.lower() == "hash":
        flag = True
    else:
        flag = False
        print("-structureParameter takes values from {Flat, IndexedBig, BitMap, Hash}")
    return flag

def checkDiscretizationType():
    flag = False
    val = Globals.getDiscretization()
    if val.lower() == "none":
        flag = True
    elif val.lower() == "mdl":
        flag = True
    elif val.lower() == "ef":
        flag = True
    elif val.lower() == "ew":
        flag = True
    else:
        flag = False
        print("-DiscretizationType type takes values from {None, mdl, ef, ew}")
    if not val.lower() == "none" and Globals.isNormalizeNumeric():
        print("Can't discretize and normalize at the same time")
        flag = False
    return flag

def checkMCMCType():
    flag = False
    val = Globals.getMcmcType()
    if val.lower() == "als":
        flag = True
    else:
        flag = False
        print("-MCMC type takes values from {ALS}")
    return flag

def checkExperimentType():
    flag = False
    val = Globals.getExperimentType()
    if val.lower() == "drift":
        flag = True
    else:
        flag = False
        print("-experimentType takes values from {drift}")
    return flag

def checkModel():
    flag = False
    val = Globals.getModel()
    if val.lower() == "ande" or val.lower() == "alr" or val.lower() == "kdb" or val.lower() == "fm" or val.lower() == "ann" or val.lower() == "halr" or val.lower() == "fealr" or val.lower() == "feealr":
        flag = True
    else:
        flag = False
        print("-Model takes values from {AnDE, ALR, KDB, FM, ANN, hALR, feALR, feeALR}")
    return flag

def checkOptimizer():
    flag = False
    val = Globals.getOptimizer()
    if val.lower() == "none" or val.lower() == "sgd" or val.lower() == "mcmc":
        flag = True
    else:
        flag = False
        print("-optimizer takes values from {None, SGD, MCMC}")
    return flag

def checkSGDType():
    flag = False
    val = Globals.getSgdType()
    if val.lower() == "plainsgd" or val.lower() == "adagrad" or val.lower() == "adadelta" or val.lower() == "nplr":
        flag = True
    else:
        flag = False
        print("-sgdType takes values from {plainsgd, adagrad, adadelta, nplr}")
    return flag

def checkAdaptiveRegularization():
    flag = False
    val = Globals.getAdaptiveRegularization()
    if val.lower() == "none" or val.lower() == "rendal" or val.lower() == "provost" or val.lower() == "zaidi1" or val.lower() == "zaidi2":
        flag = True
    else:
        flag = False
        print("-adaptiveRegularization takes values from {none, rendal, provost, zaidi1, zaidi2}")
    return flag

def checkRegularizationType():
    flag = False
    val = Globals.getRegularizationType()
    if val.lower() == "l2":
        flag = True
    else:
        flag = False
        print("-Regularization Type takes values from {L2}")
    return flag

def checkPreProcessParameterType():
    flag = False
    val = Globals.getPreProcessParameter()
    if val.lower() == "explore" or val.lower() == "createheader" or val.lower() == "discretize" or val.lower() == "slice" or val.lower() == "normalize" or val.lower() == "missingimputate" or val.lower() == "headeralignment" or val.lower() == "dice" or val.lower() == "binarize" or val.lower() == "binarizeclass" or val.lower() == "compactdiscretizedfile":
        flag = True
    else:
        flag = False
        print("-preProcessParameter takes values from <Explore, CreateHeader, Discretize, Slice, Normalize, MissingImputate, HeaderAlignment, Dice, Binarize, BinarizeClass, compactDiscretizedFile}")
    return flag

def determineFlowVal():
    flowVal = ""
    val = Globals.getFlowParameter()
    parseVal = val.split(":")
    if len(parseVal) != 2:
        flowVal = ""
    else:
        flowVal = parseVal[0]
    return flowVal

def getFlowValues():
    flowValues = ""
    val = Globals.getFlowParameter()
    parseVal = val.split(":")
    if len(parseVal) != 2:
        flowValues = ""
    else:
        flowValues = parseVal[1]
    flowValues = flowValues.replace("{", "").replace("}", "").replace("(", "").replace(")", "")
    parseFlowValues = flowValues.split(",")
    numValues = len(parseFlowValues)
    flowValuesDouble = None
    if numValues != 0:
        flowValuesDouble = [float(x) for x in parseFlowValues]
    return flowValuesDouble

def determineDSValue():
    val = Globals.getTrainFile()
    loclast = val.rfind("/")
    locDot = val.rfind(".")
    ds = val[loclast+1:locDot]
    return ds


