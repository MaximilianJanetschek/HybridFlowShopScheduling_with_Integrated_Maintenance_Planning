import gurobipy as gp


def setTwoStageInitialSolution(scenarioModels, FirstStage, TwoStageModel, parameters):
    """
    Combine individual scenario solution into new incubment solution for two-stage model.
    """
    # scheduled repair
    repairFirstStage = FirstStage['scheduling_repair']
    # set first stage variables
    for (stage, shift) in TwoStageModel._scheduling_repair.keys():
        if repairFirstStage[stage, shift] > 0:
            TwoStageModel._scheduling_repair[stage, shift].Start = repairFirstStage[stage, shift]
        else:
            TwoStageModel._scheduling_repair[stage, shift].Start = 0

    # set all other variables
    for scenario in parameters.scenarios:
        # get all variables in Scenario
        for varSce in scenarioModels[scenario].getVars():
            # get var from TwoStage
            nameTwo = f'{scenario}{varSce.VarName}'
            varTwo = TwoStageModel.getVarByName(nameTwo)
            varTwo.Start = varSce.X