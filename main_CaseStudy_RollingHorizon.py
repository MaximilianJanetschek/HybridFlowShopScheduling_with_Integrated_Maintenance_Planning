# import general modules
import gurobipy as gb
import matplotlib.pyplot as plt


# import own modules
from Data import load_Data_from_excel
from model import *
from utilities import *

# load configuration file - specifies horizon length and amount of jobs
config = load_config("./config/config.yml")

# load Input data from Excel file to retrieve machine parameters, maintenance task and layout
data, layout, shifts = load_Data_from_excel('discrete', config)

# define parameters class for all input date (Excel and config) to allow for easier access
parameters = ParametersOfModel(data, layout, config, shifts)

# Initialize Relaxed Two-Stage Stochastic Program, including Pricing Routine to generate relevant Production Assignments
TwoStageRelaxed = TwoStageModel(parameters, layout, shifts, config)

# Transform relaxed model into its MILP formulation
fristStage_GreedyStart, patternSelected, batchDependeny, batchProcessingTime = TwoStageRelaxed.generate_IP_model()

# Generate Scenario Problems, required to solve them independantly in construction procedure
HFSS = HFSS_ColumnGeneration(parameters=parameters, first_stage_decision=fristStage_GreedyStart)
HFSS.generate_SolutionProcedureForScenario(parameters, scenarios=parameters.scenarios, afterPricingDependency=batchDependeny, AfterPricingProcessTime=batchProcessingTime, selectedPattern = patternSelected)

# tune individual scenarios by optimizing them with the given first stage decision
newFirstStage = initialSolutionTuning(scenarioProblem=HFSS, firstStageGreedySolution=fristStage_GreedyStart, parameters=parameters)

# set tuned solution of all individal scenarios as incumbent solution in two Stage model
setTwoStageInitialSolution(HFSS.IntegerScenarioModels, newFirstStage, TwoStageRelaxed.IP_model, parameters)

# optimize full MILP Stochastic Program
TwoStageRelaxed.optimizeTwoStageModel(parameters)


# Get detials on objective
print('Objective function cost')
# First Stage Decision
firstStage_Cost =gp.quicksum(TwoStageRelaxed.scheduling_cost[
    (TwoStageRelaxed.schedulingCategory.repair, stage)]*TwoStageRelaxed.IP_model._scheduling_repair[stage, shift].X for (stage, shift) in TwoStageRelaxed.IP_model._scheduling_repair.keys())
print(f'planned {firstStage_Cost}')

# get objective details for all scenarios
for scenario in parameters.scenarios:
    obj_summedEnergyConsumption = 0
    obj_rescheduling_cost_repair = 0
    obj_rescheduling_cost_repair += gp.quicksum(parameters.rescheduling_data_cost[
                                                   (parameters.schedulingCategory.repair, stage)] *
                                               TwoStageRelaxed.IP_model._TwoStage_rescheduling_repair[scenario][(stage, shift)].X for
                                               (stage, shift) in TwoStageRelaxed.IP_model._TwoStage_rescheduling_repair[scenario].keys())

    # get energy consumption by summing up over all machines
    for machine in parameters.machines:
        var = TwoStageRelaxed.IP_model.getVarByName(f'{scenario}_energyConsumptionMachine[{machine}]')
        obj_summedEnergyConsumption += var.X * parameters.input_data[config.data.generalPar]['EnergyCost'][
            'ParameterValue']
    # output final objectives
    print(f'energy {obj_summedEnergyConsumption} in scenario {scenario}, contributing {TwoStageRelaxed.scenarios_prob[scenario]*obj_summedEnergyConsumption} to total objective')
    print(f'rescheduling {obj_rescheduling_cost_repair} in scenario {scenario}, contributing {TwoStageRelaxed.scenarios_prob[scenario]*obj_rescheduling_cost_repair} to total objective')



