# import general modules
import gurobipy as gb
import matplotlib.pyplot as plt


# import own modules
from Data import load_Data_from_excel
from model import *
from utilities import *

# load configuration file - specifies horizon length and amount of jobs
config = load_config("./config/config.yml")
config.data.path_excel_discrete = "Data/ProcessParamter_Data_Discrete_IndpendentScenarios.xlsx"

# load Input data from Excel file to retrieve machine parameters, maintenance task and layout
data, layout, shifts = load_Data_from_excel('discrete', config)

# define parameters class for all input date (Excel and config) to allow for easier access
parameters = ParametersOfModel(data, layout, config, shifts)

# Initialize Relaxed Two-Stage Stochastic Program, including Pricing Routine to generate relevant Production Assignments
TwoStageRelaxed = TwoStageModel(parameters, layout, shifts, config)

# Transform relaxed model into its MILP formulation
fristStage_GreedyStart, patternSelected, batchDependeny, batchProcessingTime = TwoStageRelaxed.generate_IP_model()
print('\nFinished Two Stage Master \n')

# set first stage decision on repairs to be 0 for stages and shifts
firstStage = {'scheduling_repair':{}}
for stage in parameters.set_stages:
    for shift in parameters.set_shifts:
        firstStage['scheduling_repair'][stage, shift] = 0

# add class for scenario problems
HFSS_P = HFSS_ColumnGeneration(parameters=parameters, first_stage_decision=firstStage)
# generate individal scenario problems and solution procedure for the identified produciton assignments during pricign = batchProcssingTime, pattern selected
HFSS_P.generate_SolutionProcedureForScenario(parameters, scenarios=parameters.scenarios, afterPricingDependency=batchDependeny, AfterPricingProcessTime=batchProcessingTime, selectedPattern = patternSelected)

# initialse plotting
plt.rcParams['font.size']=14
plt.rcParams.update({
    "font.family": "Helvetica"
})
fig, (ax1, ax2,ax3) = plt.subplots(3, 1,sharex=True, figsize=(12,20))
plotByScenario = {'Pessimistic': ax1, 'Average': ax2, 'Optimistic': ax3}
fig.suptitle('Bound convergence for independent optimisation of scenarios', fontsize = 18)


for scenario in parameters.scenarios:
    print(f'Test indiviudal scenario {scenario}')
    # initialise required model attributes for tracking the solution procedure
    IP_model = HFSS_P.IntegerScenarioModels[scenario]
    IP_model._counter = 0
    IP_model._start = time.time()
    IP_model._obj = np.inf
    IP_model._bd = 0
    IP_model._dataUB = []
    IP_model._dataLB = []
    IP_model.Params.TimeLimit = 3600


    # set True to determine performance in Vanilla B&B setup
    if False:
        IP_model.Params.Cuts = 0
        IP_model.Params.TimeLimit = 3600
        #IP_model.Params.Heuristics=0
        #IP_model.Params.RINS=0
        #IP_model.Params.Presolve=0
        #IP_model.Params.Aggregate=0
        #IP_model.Params.Symmetry = 0
        #IP_model.Params.Disconnected = 0
    IP_model.update()

    # optimise model with defined problem cust
    IP_model.optimize(independantScenario_cb)

    # markers_on defines all points where a new incumbent solution was found
    markers_on = [counter for counter in range(0,len(IP_model._dataUB))]
    (ltime, lbound) = IP_model._dataLB[-1]
    (utime, ubound) = IP_model._dataUB[-1]
    IP_model._dataUB.append((ltime, ubound))
    horizon = parameters.horizon

    # save tracked upper and lower bound
    with open(f"Results/independent_dataUB_{horizon}.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "UpperBound"])
        writer.writerows(IP_model._dataUB)
    with open(f"Results/independent_dataLB_{horizon}.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "LowerBound"])
        writer.writerows(IP_model._dataLB)

    import pandas as pd
    from matplotlib import pyplot as plt

    # transform collected bounds into dataframe format for easier plotting
    columns = ["Time", "UpperBound"]
    dfUB = pd.read_csv(f"Results/independent_dataUB_{horizon}.csv", usecols=columns)

    columns = ["Time", "LowerBound"]
    dfLB = pd.read_csv(f"Results/independent_dataLB_{horizon}.csv", usecols=columns)

    print("Contents in csv file:", dfLB)
    print("Contents in csv file:", dfUB)

    # plot upper and lower bound
    test = plt.figure(figsize=(6, 4))
    plotByScenario[scenario].plot(dfUB.Time, dfUB.UpperBound ,markevery=markers_on,label='UpperBound', linestyle='--', marker='o', color='#0000C4')
    plotByScenario[scenario].plot(dfLB.Time, dfLB.LowerBound, label='LowerBound', color='#4E9DF8')
    plotByScenario[scenario].set(title=f'{scenario}')
    plotByScenario[scenario].legend(['Upper Bound', 'Lower Bound'], fontsize=10)


    print(f'It took {IP_model.runtime} to solve {scenario} with an objective of {IP_model.ObjVal}')

# plot all scenario solution in stacked format
current_values = fig.gca().get_yticks()
ax1.set_yticklabels(['{:,.0f}'.format(x/1000) for x in current_values])
ax2.set_yticklabels(['{:,.0f}'.format(x/1000) for x in current_values])
ax3.set_yticklabels(['{:,.0f}'.format(x/1000) for x in current_values])
fig.supxlabel('CPU times [in seconds]')
fig.supylabel('Objective Function Value\n[in thousand Euro]')
#for ax in plotByScenario.values():
    #ax.label_outer()
fig.tight_layout()
plotName = "figures/StackedIndependenScenario_Instance_"+ '_' + str(parameters.config.model_discrete.set.time_slots.horizon_in_days)
fig.savefig(plotName, dpi=800)
fig.show()