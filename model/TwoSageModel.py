from utilities import *
from .HFSS_RestrictedMaster import *
import gurobipy as gp
from gurobipy import *
import numpy as np
from gurobipy import GRB
from .Pricing_Problem import *
import os
import csv




class TwoStageModel():
    def __init__(self, parameters, layout, shifts, config):
        # define scenarios
        self.scenarios = [scenario for scenario in config.data.maintenance_name.scenarios_prob.keys()]
        self.scenarios_prob = {scenario: prob for scenario, prob in config.data.maintenance_name.scenarios_prob.items()}

        # define scheduling aspects
        self.scheduling_cost = parameters.input_data[config.data.scheduling]  # format {(maintenance, stage): cost, ..., (repair,stage): cost, ...}
        self.schedulingCategory = config.data.scheduling_cost.category_name

        # safe production characteristic for easier access
        self.layout = layout
        self.shifts = shifts
        self.config = config
        self.parameters = parameters


    def generate_IP_model(self):

        # initialize model
        parameters = self.parameters
        set_scenarios = parameters.config.data.maintenance_name.scenarios.keys()
        self.BigM = parameters.max_number_slots
        model = gp.Model('Full Two Stage Model')

        # initialize team TOP as pricing
        Pricing_Problem = self.define_pricing_problem(parameters)

        # add a feasible production assignment
        self.add_initial_generated_pattern(model)
        
        # add first stage decision variables for repair and production assignment selection
        model._scheduling_repair = model.addVars(self.layout, self.shifts, vtype=GRB.CONTINUOUS,lb=0 , name='firstStage_scheduling_repair')
        # initialise tuple list, will be extended during pricign
        model._generated_pattern = gp.tupledict()
        model._generated_pattern[1] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name='patterns[1]')

        # add model constraints
        model._const_sum_patterns = model.addConstr(model._generated_pattern[1]==1, name='select_pattern')
        model.update()

        # cost for scheduled repair
        firstStage_Cost = gp.LinExpr()
        firstStage_Cost += gp.quicksum(model._scheduling_repair.sum(stage,'*') * self.scheduling_cost[
            (self.schedulingCategory.repair, stage)] for stage in self.layout )

        # first_stage_decision: format {scheduled_repair: {(stage, shift): number_of_repair}}
        first_stage_decision = {'scheduling_repair':gp.tupledict()}
        for stage in self.layout:
            for shift in self.shifts:
                first_stage_decision['scheduling_repair'][stage, shift] = model._scheduling_repair[stage, shift]

        # initialize required class for scenario problems, e.g. class that prepares all information on maitenance, energy consumption and so in desire format
        HFSS_Scenario = HFSS_ColumnGeneration(parameters=self.parameters, first_stage_decision=first_stage_decision)
        HFSS_Scenario.defining_HFSS_model(parameters,model)
        ScenarioCost = gp.LinExpr()

        # add the second stage formulation by iterating through all scenarios and adjusting the model initialization accordingly, e.g. use the corresponding upper maintenance limits
        for scenario in set_scenarios:
            model._scenario = scenario
            # add decision variables
            HFSS_Scenario.add_decision_variables_to_model(model,False,scenario, parameters, scenarioKey=scenario)
            # add constraints
            HFSS_Scenario.add_constraints_to_model(model, scenario,  parameters, model._generated_pattern)
            # add objective
            obj_rescheduling_cost_repair, obj_peakEnergyConsumption = HFSS_Scenario.modelSetObjective(model, parameters)
            ScenarioCost += self.scenarios_prob[scenario] * (obj_rescheduling_cost_repair + obj_peakEnergyConsumption)

            model.update()

        # add combined objective
        model.setObjective(firstStage_Cost + ScenarioCost, GRB.MINIMIZE)
        model.update()
        
        # add full pricing routine
        start_pricing = time.time()
        self.perform_pricing_routing(model, Pricing_Problem)
        print(f'Pricing took {time.time()-start_pricing}')
        model.update()

        # make formulation more compact and reinitialize
        modelCompact = self.compactModelFormulation(parameters, set_scenarios, model)
        modelCompact.update()

        # retrieve improvement of model formulation
        variableReduction = round((model.NumVars - modelCompact.NumVars)*100 / model.NumVars,2)
        constReduction = round((model.NumConstrs - modelCompact.NumConstrs) * 100 / model.NumConstrs, 2)
        print("\n Model reduction based on Forward-Backward-Scheduling")
        print(f"Reduced Number of variables by {variableReduction} %")
        print(f"Reduced Number of constraints by {constReduction} %")

        modelCompact.optimize()
        modelCompact.update()



        # generate IP model
        IP_model = modelCompact.copy()

        # change frist stage variables
        IP_model._scheduling_repair = gp.tupledict()

        # adapt first stage decision variables fo production assignment selection and repair planning to firt IP formulatino
        for (stage, shift) in modelCompact._scheduling_repair.keys():
            var = IP_model.getVarByName(f'firstStage_scheduling_repair[{stage},{shift}]')
            var.setAttr(GRB.Attr.VType, GRB.INTEGER)
            IP_model._scheduling_repair[(stage, shift)] = var

        IP_model._generated_pattern = gp.tupledict()
        for pattern in modelCompact._generated_pattern.keys():
            var = IP_model.getVarByName(f'patterns[{pattern}]')
            var.setAttr(GRB.Attr.VType, GRB.BINARY)
            IP_model._generated_pattern[pattern] = var

        # set first stage decision to 0
        for (stage, shift) in modelCompact._scheduling_repair.keys():
            HFSS_Scenario.firstStage_repair[(stage, shift)] = 0
        IP_model._TwoStage_rescheduling_repair = gp.tupledict()

        IP_model._scenario_repair_batch_machine = {scenario: gp.tupledict() for scenario in parameters.scenarios}
        # change all scenario variables
        initialSolution = 0
        for scenario in parameters.scenarios:
            HFSS_Scenario.changeVariableTypes(IP_model, modelCompact, scenario)

            # update also rescheduling variable
            self.changeAddVars(IP_model, modelCompact, scenario)

            # update continous variables
            HFSS_Scenario.update_varPointer_toModel(IP_model, modelCompact, scenario)

            # copy attributes
            HFSS_Scenario.copy_ModelAttrributes(IP_model, modelCompact)

            # change reschedling attribut
            self.changeAddVars(IP_model,modelCompact, scenario)

            # set variables to initial solution
            scenarioGreedy = time.time()
            create_initial_solution(HFSS_Scenario, IP_model, scenario)
            initialSolution += time.time() - scenarioGreedy
        IP_model.update()


        print(f"Number of integer and binaryy variables {IP_model.NumIntVars}")
        print(f"Number of variables {IP_model.NumVars}")
        print(f"Number of constraint {IP_model.NumConstrs}")
        addTime = time.time()
        patternSelected = {}
        for pattern in model._batch_dependency.keys():
            if pattern == 1:
                model._generated_pattern[pattern].Start = 1
                patternSelected[pattern]=1
            else:
                model._generated_pattern[pattern].Start = 0
                patternSelected[pattern] = 0


        first_stage_key = parameters.config.model_discrete.decision_variables
        fristStage_GreedyStart = gp.tupledict()
        fristStage_GreedyStart[first_stage_key.scheduled_repair] = gp.tupledict()
        # set starting solution for now to 0 for all repairs and shifts
        for (stage, shift) in IP_model._TwoStage_rescheduling_repair[scenario].keys():
            fristStage_GreedyStart[first_stage_key.scheduled_repair][(stage, shift)] = 0
        for varKey in IP_model._scheduling_repair.keys():
            IP_model._scheduling_repair[varKey].Start = 0


        model.update()
        IP_model.update()
        averageScenario = 'Average'
        # identify all used repair-shift assignments in average solution, to swap adhoc repairs for planned
        for (stage, shift) in IP_model._TwoStage_rescheduling_repair[averageScenario].keys():
            if IP_model._TwoStage_rescheduling_repair[averageScenario][stage, shift].Start >= 0.8:
                fristStage_GreedyStart[first_stage_key.scheduled_repair][stage, shift] = \
                IP_model._TwoStage_rescheduling_repair[averageScenario][stage, shift].Start
                # set first stage repair to ad hoc repair
                IP_model._scheduling_repair[stage, shift].Start = IP_model._TwoStage_rescheduling_repair[averageScenario][
                    stage, shift].Start
                # set ad hoc repairs to 0
                IP_model._TwoStage_rescheduling_repair[averageScenario][stage, shift].Start = 0
        IP_model.update()
        # reduce ad hoc repairs in all subsequent
        for scenario in parameters.scenarios:
            # skip average scenario as already swapped
            if scenario == averageScenario:
                next
            for (stage, shift) in IP_model._TwoStage_rescheduling_repair[scenario].keys():
                    IP_model._TwoStage_rescheduling_repair[scenario][stage, shift].Start = max(0,IP_model._TwoStage_rescheduling_repair[scenario][stage, shift].Start-IP_model._scheduling_repair[stage, shift].Start)

        initialSolution += time.time() - addTime

        print(f'It took {initialSolution} seconds to construct inital hfss solution')

        # retrieve MIP Start sol value by initializing model quickly, thus time limit of 10 seconds
        self.IP_model = IP_model
        self.IP_model.Params.OutputFlag = 1
        IP_model.Params.TimeLimit = 10
        self.IP_model._counter = 0
        self.IP_model._start = time.time()
        self.IP_model._obj = np.inf
        self.IP_model._bd = 5500
        self.IP_model._dataUB = []
        self.IP_model._dataLB = []
        self.IP_model._gap = []
        IP_model.update()
        self.IP_model.optimize(twoStage_cb)
        print(f'MIP construction solution {IP_model._startobjval}')

        # set model to be ready for optimisation
        self.IP_model.Params.OutputFlag = 1
        IP_model.Params.TimeLimit = 3600
        IP_model._fileDir = 'solution_master'+ '/'
        IP_model._filePath = IP_model._fileDir + '/leaf'
        if not os.path.exists(IP_model._fileDir):
            # if the corresponding scenario directory is not present then create it.
            os.makedirs(IP_model._fileDir)

        IP_model.Params.SolFiles = IP_model._filePath
        print('\nFinished Two Stage Master \n')

        return fristStage_GreedyStart, patternSelected, model._batch_dependency, model._batch_processing_time

    # required batch dependency, and greedy first stage decision for

    def compactModelFormulation(self, parameters, set_scenarios, model):
        modelCompact = gp.Model('Full Two Stage Compact modelCompact')

        # just set first stage variables to 0
        modelCompact._scheduling_repair = modelCompact.addVars(self.layout, self.shifts, vtype=GRB.CONTINUOUS, lb=0,
                                                 name='firstStage_scheduling_repair')
        modelCompact._generated_pattern = gp.tupledict()
        modelCompact._generated_pattern = modelCompact.addVars(list(model._batch_dependency.keys()),vtype=GRB.CONTINUOUS, lb=0, ub=1, name='patterns')

        # add modelCompact constraints
        modelCompact._const_sum_patterns = modelCompact.addConstr(modelCompact._generated_pattern.sum('*') == 1, name='select_pattern')
        modelCompact.update()
        # cost for scheduled repair
        firstStage_Cost = gp.LinExpr()
        firstStage_Cost += gp.quicksum(modelCompact._scheduling_repair.sum(stage, '*') * self.scheduling_cost[
            (self.schedulingCategory.repair, stage)] for stage in self.layout)

        # Start sub scenario

        # first_stage_decision: format {scheduled_repair: {(stage, shift): number_of_repair}}
        first_stage_decision = {'scheduling_repair': gp.tupledict()}
        for stage in self.layout:
            for shift in self.shifts:
                first_stage_decision['scheduling_repair'][stage, shift] = modelCompact._scheduling_repair[stage, shift]

        # add all variables in scenario
        HFSS_Scenario = HFSS_ColumnGeneration(parameters=self.parameters, first_stage_decision=first_stage_decision)
        HFSS_Scenario.defining_HFSS_model(parameters, modelCompact)
        ScenarioCost = gp.LinExpr()
        startTightening = time.time()
        HFSS_Scenario.removeRedundantVariables(parameters, HFSS_Scenario.gurobi_model, model._batch_dependency,
                                               model._batch_processing_time)

        print(f'It took {time.time()-startTightening} seconds to determine tightened formulation')
        # copy required attrbitues
        modelCompact._batch_dependency = model._batch_dependency
        modelCompact._batch_processing_time = model._batch_processing_time

        for scenario in set_scenarios:
            print(f'Add all variables and constraints for {scenario}')
            modelCompact._scenario = scenario
            # add decision variables
            HFSS_Scenario.add_decision_variables_to_model(modelCompact, False, scenario, parameters, scenarioKey=scenario)
            # add constraints
            HFSS_Scenario.add_constraints_to_model(modelCompact, scenario, parameters, modelCompact._generated_pattern, modelCompact._batch_dependency, modelCompact._batch_processing_time)
            # add objective
            obj_rescheduling_cost_repair, obj_peakEnergyConsumption = HFSS_Scenario.modelSetObjective(modelCompact, parameters)
            ScenarioCost += self.scenarios_prob[scenario] * (obj_rescheduling_cost_repair + obj_peakEnergyConsumption)

            modelCompact.update()
        # add combined objective
        modelCompact.setObjective(firstStage_Cost + ScenarioCost, GRB.MINIMIZE)
        modelCompact.update()


        return modelCompact

    def optimizeTwoStageModel(self, parameters, nameFig = ''):
        """
        This functions optimises the two stage stochastic programm and track the gaps convergence during optimisation. Finally, the collected upper and lower bounds are visualized with matplotlib
        :param parameters:
        :param nameFig:
        :return:
        """
        # initialize required model attributes
        self.IP_model._start = time.time()
        self.IP_model._obj = np.inf
        self.IP_model._bd = 5000
        self.IP_model._dataUB = []
        self.IP_model._dataLB = []
        self.IP_model._gap = []
        self.IP_model.Params.MIPFocus = 3
        self.IP_model.update()
        #self.IP_model.tune()
        #self.IP_model.tune()
        self.IP_model.optimize(twoStage_cb)

        # set markers to visualize where incumbent solution are identified
        markers_on = [counter for counter in range(0, len(self.IP_model._dataUB))]
        (ltime, lbound) = self.IP_model._dataLB[-1]
        (utime, ubound) = self.IP_model._dataUB[-1]
        horizon = parameters.horizon
        self.IP_model._dataUB.append((ltime, ubound))

        # safe identified lower bound, upper bound and gap
        with open(f'Results/dataUB_{horizon}.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["Time", "UpperBound"])
            writer.writerows(self.IP_model._dataUB)
        with open(f'Results/dataLB_{horizon}.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["Time", "LowerBound"])
            writer.writerows(self.IP_model._dataLB)
        with open(f'Results/dataGAP_{horizon}.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["Time", "GAP"])
            writer.writerows(self.IP_model._gap)

        # load matplotlib and pandas to draw the bounds
        import pandas as pd
        from matplotlib import pyplot as plt, font_manager as fm
        plt.rcParams['font.size'] = 14

        # use pandas to get lower and upper bound into dataframe format
        columns = ["Time", "UpperBound"]
        dfUB = pd.read_csv(f"Results/dataUB_{horizon}.csv", usecols=columns)

        columns = ["Time", "LowerBound"]
        dfLB = pd.read_csv(f"Results/dataLB_{horizon}.csv", usecols=columns)

        print("Contents in csv file:", dfLB)
        print("Contents in csv file:", dfUB)

        # plot identified bounds during optimisation
        test = plt.figure(figsize=(10,5))
        plt.plot(dfUB.Time, dfUB.UpperBound, label='UpperBound', linestyle='--',marker='o', color='#0000C4')
        plt.plot(dfLB.Time, dfLB.LowerBound, label='LowerBound', color='#4E9DF8')
        current_values = plt.gca().get_yticks()
        plt.gca().set_yticklabels(['{:,.1f}'.format(x/1000) for x in current_values])
        plt.grid(axis='y')
        plt.ylabel('Objective Function Value\n[in thousand Euro]')
        plt.xlabel('CPU Time [in seconds]')
        plt.suptitle('Convergence upper and lower bound', fontsize=18)

        # save and show figure
        plotName = "figures/Test_Instance" +nameFig + str(self.parameters.config.model_discrete.set.time_slots.horizon_in_days)
        plt.legend(['Upper Bound', 'Lower Bound'])
        plt.tight_layout()
        plt.savefig(plotName, dpi=600)
        plt.show()

        # plot used planned and ad hoc repairs for identified solution
        for (stage, shift) in self.IP_model._scheduling_repair.keys():
            if self.IP_model._scheduling_repair[stage, shift].X > 0.1:
                print(f'{stage}, {shift}, {self.IP_model._scheduling_repair[stage, shift].X}')
        for scenario in parameters.scenarios:
            for (stage, shift) in self.IP_model._TwoStage_rescheduling_repair[scenario].keys():
                if self.IP_model._TwoStage_rescheduling_repair[scenario][stage, shift].X >= 0.1:
                    print(f'{scenario},{stage}, {shift} with rescheduling  {self.IP_model._TwoStage_rescheduling_repair[scenario][stage, shift].X}')


    def changeAddVars(self, model: gp.Model, model_toBeCopied:gp.Model, scenario):
        """
        To obtain the integer model, the corresponding first stage decision varibales need to changed accordingly.
        """
        model._TwoStage_rescheduling_repair[scenario] = gp.tupledict()
        for (stage, shift) in model_toBeCopied._rescheduling_repair.keys():
            var = model.getVarByName(f'{scenario}_rescheduling_repair[{stage},{shift}]')
            var.setAttr(GRB.Attr.VType, GRB.INTEGER)
            model._TwoStage_rescheduling_repair[scenario][(stage, shift)] = var

        for (repair, batch, machine) in model_toBeCopied._repair_batch_machine.keys():
            var = model.getVarByName(f'{scenario}_repair_batch_machine[{repair},{batch},{machine}]')
            var.setAttr(GRB.Attr.VType, GRB.BINARY)
            model._scenario_repair_batch_machine[scenario][(repair, batch, machine)] = var


    def define_pricing_problem(self, parameters):

        # Generate Pricing Problem by initializing Graph with networkx
        TeamOP = multi_visit_TOP(parameters)
        self.batch_dependency, self.batch_processing_time = TeamOP.generate_initial_soltuion(parameters)

        TeamOP.generate_pricing_problem(parameters)

        return TeamOP


    def perform_pricing_routing(self, model:gp.Model, TeamOP):
        """
        Function that adds all required production assignments to RMP. It iterates through optimising the RMP, solving dijkstra for incumbent solution, solving the TOP, retreivig the produciton assignment and solving the RMP again.
        :param model: Restricted master problem
        :param TeamOP: Pricing model
        :return:
        """
        print('Enter Pricing Routine')
        parameters = self.parameters

        # perform pricing routing until no better Solution can be found
        while True:
            model.optimize()

            # can only continue if model is solved to optimality
            if not model.Status == GRB.OPTIMAL:
                print('Error! Master Problem is not solvable to optimality!')
                return None

            # Initialize duals for cross dependency between batches
            dual_cross_dependency = gp.tupledict()
            for (batch_prev, machine_prev, batch, machine, stage) in parameters.set_batch_cross_stage_dependency:
                dual_cross_dependency[(batch_prev, machine_prev, batch, machine)] = 0

            # Initialize duals that all depend on the position per machine
            dual_processing_time = gp.tupledict()
            dual_maintenance_time = gp.tupledict()
            dual_repair_time = gp.tupledict()
            for key in parameters.set_position_per_machine:
                dual_processing_time[key] = 0
                dual_maintenance_time[key] = 0
                dual_repair_time[key] = 0

            # safe constraint pointer for required constraints
            constraintPointer_duals = {scenario:{'cross_dependency':gp.tupledict(), 'operationTime':gp.tupledict(), 'maintenanceTime':gp.tupledict(), 'repairTime':gp.tupledict()} for scenario in parameters.scenarios}
            for scenario in parameters.scenarios:
                # safe constraint pointer for cross dependency constraint
                for (batch_prev, machine_prev, batch, machine, stage) in parameters.set_batch_cross_stage_dependency:
                    constraintPointer_duals[scenario]['cross_dependency'][(batch_prev, machine_prev, batch, machine)] = model.getConstrByName(
                        f'{scenario}_batch_cross_dependency[{batch_prev},{machine_prev},{batch},{machine}]')
                # safe constraint pointer for operation time constraint
                for (position,machine) in parameters.set_position_per_machine:
                    constraintPointer_duals[scenario]['operationTime'][position, machine] = model.getConstrByName(
                        f'{scenario}_batch_end_time[{position},{machine}]')
                # safe constraint pointer for maintenance time contraint
                for (maintenance, batch, machine) in parameters._set_maintenance_batch_machine_startOne:
                    constraintPointer_duals[scenario]['maintenanceTime'][maintenance,position, machine] = model.getConstrByName(
                                f'{scenario}_maintenance_time_tracker[{position},{machine},{maintenance}]')
                # safe constraint pointer for repair time constraint
                for (repair, batch, machine) in parameters._set_repair_batch_machine_startOne:
                    constraintPointer_duals[scenario]['repairTime'][repair,position, machine]= model.getConstrByName(
                                f'{scenario}_repair_time_tracker[{position},{machine},{repair}]')
            self.constraintPointer_duals = constraintPointer_duals

            # get duals for each scenario
            for scenario in parameters.scenarios:
                # Get shadow prices for cross dependency constraint, in the format ((position_prev, machine_prev)(position,machine))

                # get the required cross dependency constraint
                for (batch_prev, machine_prev, batch, machine),cross_constraint  in constraintPointer_duals[scenario]['cross_dependency'].items():
                    dual_cross_dependency[(batch_prev, machine_prev, batch, machine)] += cross_constraint.Pi

                # Get shadow prices for operation time, in the format (position, machine)
                for (position,machine),operationTime_constraint  in constraintPointer_duals[scenario]['operationTime'].items():
                    dual_processing_time[(position,machine)] += operationTime_constraint.Pi

                # Get shadow prices for maintenacne time, in the format (position, machine)
                for (maintenance, batch, machine),maintenanceTime_constraint in constraintPointer_duals[scenario]['maintenanceTime'].items():
                    dual_maintenance_time[(position,machine)] += maintenanceTime_constraint.Pi


                # Get shadow prices for repair time, in the ofrmat (poistion, machine)
                for (repair, batch, machine),repairTime_constraint  in constraintPointer_duals[scenario]['repairTime'].items():
                    dual_repair_time[(position,machine)] += repairTime_constraint.Pi


            # generate new pattern
            patternFound, pattern = TeamOP.reOptimize_pricing(dual_cross_dependency, dual_processing_time,
                                                              dual_maintenance_time, dual_repair_time, self.BigM)
            if patternFound:
                # safe pattern to list
                model._pattern_counter += 1
                model._batch_dependency[model._pattern_counter] = pattern.cross_dependency
                model._batch_processing_time[model._pattern_counter] = pattern.processing_time
                # add directly to sum pattern constraint
                constraint_coefficient_list = [1]
                constraint_list = [model._const_sum_patterns]

                # add coefficients and constraints to list for cross dependency
                for (batch_prev, machine_prev, batch, machine),cross_constraint   in constraintPointer_duals[scenario]['cross_dependency'].items():
                    constraint_coefficient_list.append(pattern.cross_dependency[batch_prev, machine_prev, batch, machine])
                    constraint_list.append(cross_constraint)

                # add coefficients and constraints to list for processing time
                for (position,machine),processing_time_constraint  in constraintPointer_duals[scenario]['operationTime'].items():
                    constraint_coefficient_list.append(pattern.processing_time[(position,machine)])
                    constraint_list.append(processing_time_constraint)

                # add coefficients and constraints to list for maintenance time
                for (maintenance, batch, machine),maintenanceTime_constraint in constraintPointer_duals[scenario]['maintenanceTime'].items():
                    constraint_coefficient_list.append(pattern.processing_time[(position, machine)])
                    constraint_list.append(maintenanceTime_constraint)

                # add coefficients and ocnstraints to list for repair time
                for (repair, batch, machine),repairTime_constraint  in constraintPointer_duals[scenario]['repairTime'].items():
                    constraint_coefficient_list.append(pattern.processing_time[(position, machine)])
                    constraint_list.append(repairTime_constraint)

                # add generated column with coefficients to restricted master problem
                model._generated_pattern[model._pattern_counter] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0,
                                                                             column=Column(constraint_coefficient_list,
                                                                                           constraint_list),
                                                                             name=f'patterns[{model._pattern_counter}]')

            else:
                print('\nFinish Column Generation\n')
                # As no new production assginment could be identified the pricing routine stops
                break

        # Branching could start here

    def removeOverdefinedCrossDependency(self,parameters, model):
        """
        Only cross dependencies found in the producion assignments are necessary, all others can be removed.
        """
        # remove unnecessry constraints first
        remove_cons = self.determine_CrossDependencyConstraints_to_remove(model)
        self.saved_removed_cons = gp.tupledict()
        print('Remove unnecessary dependency constraints '  + str(len(remove_cons)*len(parameters.scenarios)))
        for reCon in remove_cons:
            for scenario in parameters.scenarios:
                constraint_toBeRemoved = self.constraintPointer_duals[scenario]['cross_dependency'][reCon]
                lhs, sense, rhs, name= model.getRow(constraint_toBeRemoved), constraint_toBeRemoved.Sense, constraint_toBeRemoved.RHS, constraint_toBeRemoved.ConstrName
                self.saved_removed_cons[reCon] = (lhs, sense, rhs, name)
                model.remove(model._constraint_cross_dependency[reCon])
        
    def determine_CrossDependencyConstraints_to_remove(self, model):
        """
        After all production assignments arre generated, redundant cross dependencies can be identified and removed
        """
        constraint_set_cross_dependency = []
        # iterate through all productiojn assignemnts
        for index, pattern in model._generated_pattern.items():
            for batch_prev, machine_prev, batch, machine in model._batch_dependency[index].keys():
                # if batch_dependency[index][prev_node, node] is equal to 1, then constraint needs to hold
                if model._batch_dependency[index][batch_prev, machine_prev, batch, machine] >= 1:
                    constraint_set_cross_dependency.append((batch_prev, machine_prev, batch, machine))
        remove_constraints = []
        # all constraints not used by any assignment are removed
        for constraint_key in model._batch_dependency[index].keys():
            if constraint_key not in constraint_set_cross_dependency:
                if constraint_key[3] != 'end' and constraint_key[1] != 'start':
                    remove_constraints.append(constraint_key)

        return remove_constraints
        

    def add_initial_generated_pattern(self, hfss):
        """
        As initial solution the initial constructed solution in the pricing problem is used.
        """
        # append first generated pattern
        hfss._pattern_counter = 1
        hfss._batch_dependency = {}
        hfss._batch_processing_time = {}
        hfss._batch_dependency[hfss._pattern_counter] = self.batch_dependency
        hfss._batch_processing_time[hfss._pattern_counter] = self.batch_processing_time







