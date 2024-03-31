import evaluation.utils.InputArguments
import evaluation.utils.SanctityCheck
import os
import evaluation.utils.Globals

if __name__ == "__main__":
    options = InputArguments()
    options.setOptions(args)
    if not SanctityCheck.checkStringArgs():
        print("Please correct your input arguments, ... Exiting()")
        os._exit(-1)
    else:
        SanctityCheck.printExperimentInformation()
    
    val = Globals.getExperimentType()
    if val.lower() == "traintest":
        ds = SanctityCheck.determineDSValue()
        Globals.setDataSetName(ds)
        evaluationTrainTest.learn()
    elif val.lower() == "prequential":
        ds = SanctityCheck.determineDSValue()
        Globals.setDataSetName(ds)
        evaluationPrequential.learn()
    elif val.lower() == "flowmachines":
        evaluationFlowMachines.learn()
    elif val.lower() == "drift":
        evaluationDrift.learn()