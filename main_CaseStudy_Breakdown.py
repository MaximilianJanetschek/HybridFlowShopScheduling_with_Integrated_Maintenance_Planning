# import own modules
from Data import load_Data_from_excel
from model import *
from utilities import *

# load configuration file
config = load_config("./config/config.yml")

# adjust input Excel to corresponding file
config.data.path_excel_discrete = "Data/ProcessParamter_Data_Discrete_Breakdown.xlsx"

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

# Determine used shifts and repair assignments
for (stage, shift) in TwoStageRelaxed.IP_model._scheduling_repair.keys():
    if TwoStageRelaxed.IP_model._scheduling_repair[stage, shift].X > 0.1:
        print(f'{stage}, {shift}, {TwoStageRelaxed.IP_model._scheduling_repair[stage, shift].X}')
        # fix all planned repairs by enforcing lb
        TwoStageRelaxed.IP_model._scheduling_repair[stage, shift].lb = TwoStageRelaxed.IP_model._scheduling_repair[stage, shift].X

# Retrieve current planned and adhoc repairs
for scenario in parameters.scenarios:
    # retrieve planned repairs
    for (stage, shift) in TwoStageRelaxed.IP_model._TwoStage_rescheduling_repair[scenario].keys():
        if TwoStageRelaxed.IP_model._TwoStage_rescheduling_repair[scenario][stage, shift].X >= 0.1:
            print(f'{scenario},{stage}, {shift} with rescheduling  {TwoStageRelaxed.IP_model._TwoStage_rescheduling_repair[scenario][stage, shift].X}')
    # retrieve ad hoc repairs
    for (repair, batch, machine) in TwoStageRelaxed.IP_model._scenario_repair_batch_machine[scenario].keys():
        if TwoStageRelaxed.IP_model._TwoStage_rescheduling_repair[scenario][stage, shift].X >= 0.1:
            print(f'{scenario},{repair},{batch},{machine} with rescheduling  {TwoStageRelaxed.IP_model._scenario_repair_batch_machine[scenario][scenario][(repair, batch, machine) ].X}')



# get obj attributes
print('Objective function cost')
firstStage_Cost =gp.quicksum(TwoStageRelaxed.scheduling_cost[
    (TwoStageRelaxed.schedulingCategory.repair, stage)]*TwoStageRelaxed.IP_model._scheduling_repair[stage, shift].X for (stage, shift) in TwoStageRelaxed.IP_model._scheduling_repair.keys())
obj_summedEnergyConsumption = 0
obj_rescheduling_cost_repair = 0
# collect objective in scenarios
for scenario in parameters.scenarios:
    obj_rescheduling_cost_repair += TwoStageRelaxed.scenarios_prob[scenario]*gp.quicksum(parameters.rescheduling_data_cost[
                                                   (parameters.schedulingCategory.repair, stage)] *
                                               TwoStageRelaxed.IP_model._TwoStage_rescheduling_repair[scenario][(stage, shift)].X for
                                               (stage, shift) in TwoStageRelaxed.IP_model._TwoStage_rescheduling_repair[scenario].keys())

    # energy cost are calculated based on machine cost
    for machine in parameters.machines:
        var = TwoStageRelaxed.IP_model.getVarByName(f'{scenario}_energyConsumptionMachine[{machine}]')
        obj_summedEnergyConsumption += TwoStageRelaxed.scenarios_prob[scenario]*var.X * parameters.input_data[config.data.generalPar]['EnergyCost'][
            'ParameterValue']
print(f'planned {firstStage_Cost}')
print(f'energy {obj_summedEnergyConsumption}')
print(f'rescheduling {obj_rescheduling_cost_repair}')

# reset initial state for machine due to breakdown in all scenarios, coater&dryer_2 is considered
for scenario in parameters.scenarios:
    # get constr
    constr = TwoStageRelaxed.IP_model.getConstrByName(f'{scenario}_repair_time_tracker[0,Coater&Dryer_2,RoutineRepair_R]')
    constr.RHS=0
TwoStageRelaxed.IP_model.update()

# reoptimize model
print("###"*10)
print('reoptimize')
TwoStageRelaxed.optimizeTwoStageModel(parameters, "reopt")

# get obj attributes
print('Objective function cost after reoptimization')

print('Objective function cost')
firstStage_Cost =gp.quicksum(TwoStageRelaxed.scheduling_cost[
    (TwoStageRelaxed.schedulingCategory.repair, stage)]*TwoStageRelaxed.IP_model._scheduling_repair[stage, shift].X for (stage, shift) in TwoStageRelaxed.IP_model._scheduling_repair.keys())
obj_summedEnergyConsumption = 0
obj_rescheduling_cost_repair = 0
# collect objective in scenarios
for scenario in parameters.scenarios:
    obj_rescheduling_cost_repair += TwoStageRelaxed.scenarios_prob[scenario] * gp.quicksum(parameters.rescheduling_data_cost[
                                                   (parameters.schedulingCategory.repair, stage)] *
                                               TwoStageRelaxed.IP_model._TwoStage_rescheduling_repair[scenario][(stage, shift)].X for
                                               (stage, shift) in TwoStageRelaxed.IP_model._TwoStage_rescheduling_repair[scenario].keys())

    # energy cost are calculated based on machine cost
    for machine in parameters.machines:
        var = TwoStageRelaxed.IP_model.getVarByName(f'{scenario}_energyConsumptionMachine[{machine}]')
        obj_summedEnergyConsumption += TwoStageRelaxed.scenarios_prob[scenario] * var.X * parameters.input_data[config.data.generalPar]['EnergyCost'][
            'ParameterValue']
print(f'planned {firstStage_Cost}')
print(f'energy {obj_summedEnergyConsumption}')
print(f'rescheduling {obj_rescheduling_cost_repair}')


