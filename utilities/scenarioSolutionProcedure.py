import gurobipy as gp
from gurobipy import GRB
import numpy as np
import os
import time
from .gurobi_callback import *
import re
import pickle

class ScenarioSolutionProcedure():

    def __init__(self, parameters, relaxedModels, integerProblem, scenarios,  modelClass):
        self.set_shifts = parameters.set_shifts
        self.set_stages = parameters.set_stages
        self.relaxedModels = relaxedModels
        self.incumbantSolution = {}
        for scenario in relaxedModels.keys():
            self.incumbantSolution[scenario] = {}
            model = relaxedModels[scenario]
            model.Params.OutputFlag = 0
            model.optimize()

        self.integerModels = integerProblem
        self.initialized_branchNbound(relaxedModels, integerProblem, scenarios)
        self.modelClass = modelClass
        self.bestIncumbantInteger = np.inf
        self.cutModels = {}


    def initialized_branchNbound(self, relaxedModels, integerProblem, scenarios):
        # initialized tree
        self.tree = {0: [[], None, None, None, None, None]}

        # safe all attributes for relaxed Models
        constDuals_relaxed = {scenario: {} for scenario in scenarios}

        for scenario in scenarios:
            # safe all constraint pointers to relaxed model
            relaxedModels[scenario].update()
            self.retrieve_constr_pointers(relaxedModels[scenario], "")
            # safe all branching variables to model
            self.retrieve_var_pointer(relaxedModels[scenario],"")

            # safe all constraint pointers for integerProblem
            integerProblem[scenario].update()
            self.retrieve_constr_pointers(integerProblem[scenario],"")
            integerProblem[scenario]._relaxedModel = relaxedModels[scenario]



    def retrieve_constr_pointers(self, model, scenario):

        # list all constraints, for which duals are required for RMP model
        dual_constr = {}
        dual_constr['scheduling_repair'] = {
            (shift, stage): model.getConstrByName(f'{scenario}_rescheduling_cost_repair_constraint[{shift},{stage}]') for shift in
            self.set_shifts.keys() for stage in self.set_stages}

        dual_constr['lowerLimit'] = {
            stage: model.getConstrByName(f'{scenario}_reschedulingLowerBound[{stage}]') for stage in model._stages}

        model._dual_constr = dual_constr

    def retrieve_var_pointer(self, model, scenario):
        # list all variables, for which duals are required
        dual_var = {}
        dual_var['rescheduling_repair'] = {
            (shift, stage): model.getVarByName(f'{scenario}_rescheduling_repair[{stage},{shift}]') for shift in
            self.set_shifts.keys() for stage in self.set_stages}

        dual_var['repairShift'] = {
            (repair,position, machine): model.getVarByName(f'{scenario}_repair_batch_machine[{repair},{position},{machine}]') for (repair,position, machine) in
            model._repair_batch_machine.keys()}





        #model._branchRepairPosition = dual_var['repairShift']
        model._dual_var = dual_var

        # determine all branching variables
        branching_variables = []
        for (shift, stage) in dual_var['rescheduling_repair'].keys():
            branching_variables.append(dual_var['rescheduling_repair'][(shift, stage)])

        model._branchingVars = branching_variables



    def calculateCoverInequalityCuts(self):
        print('Cover Inequalitz Cuts')

    def construction_hfss(self, scenario, SolutionLimit, iteration, RHS):


        # determine used shift
        # optimize IP model
        self.integerModels[scenario]._fileDir = 'solutionConstruction_' + str(iteration) + '/' + scenario
        self.integerModels[scenario]._filePath = self.integerModels[scenario]._fileDir + '/leaf'
        if not os.path.exists(self.integerModels[scenario]._fileDir):
            # if the corresponding scenario directory is not present then create it.
            os.makedirs(self.integerModels[scenario]._fileDir)

        if scenario == 'Pessimistic':
            create_initial_solution(self.modelClass, self.integerModels[scenario], scenario)
        else:
            pointerDict = {'Average': 'Pessimistic', 'Optimistic': 'Average'}
            prevScenarioSolutionPath = 'solutionConstruction_' + str(iteration) + '/' + pointerDict[scenario]
            try:
                for solution in os.listdir(prevScenarioSolutionPath):
                    filePath = prevScenarioSolutionPath + '/' + solution
                    print('read in ' + filePath)
                    self.integerModels[scenario].read(filePath)
                    self.integerModels[scenario].update()
            except:
                print('No node found')
        #self.integerModels[scenario].Params.SolFiles = self.integerModels[scenario]._filePath
        self.integerModels[scenario].Params.MIPGap = 0.02
        self.integerModels[scenario].Params.OutputFlag = 0
        #self.integerModels[scenario].Params.MIPfocus = 3
        self.integerModels[scenario].update()
        self.integerModels[scenario]._counter = 0
        self.integerModels[scenario].optimize(scenario_cb)
        print(f'{scenario} tuned to {self.integerModels[scenario].ObjVal}')
        self.integerModels[scenario].write(self.integerModels[scenario]._filePath + 'leaf_1.sol')


        usedShifts = {}
        usedShiftsDual = {}
        cuts = {}
        cuts[0] = []
        alpha = 0
        for (position, machine, shift) in self.integerModels[scenario]._repair_batch_shift.keys():
            var = self.integerModels[scenario].getVarByName(f'_repair_of_batch_executed_in_shift[{position},{machine},{shift}]')
            if var.X > 0.1:
                stage = self.modelClass.parameters.stage_per_machine[machine]
                if not (shift, stage) in usedShifts.keys():
                    usedShifts[shift, stage] = 0
                usedShifts[shift,stage] -= self.modelClass.parameters.rescheduling_data_cost[
                                                            (self.modelClass.parameters.schedulingCategory.repair, self.modelClass.parameters.stage_per_machine[machine])]
                usedShiftsDual[shift, stage] = -self.modelClass.parameters.rescheduling_data_cost[
                                                            (self.modelClass.parameters.schedulingCategory.repair, self.modelClass.parameters.stage_per_machine[machine])]
                alpha +=self.modelClass.parameters.rescheduling_data_cost[
                                                           (self.modelClass.parameters.schedulingCategory.repair, self.modelClass.parameters.stage_per_machine[machine])]


        cuts[0].append(({'scheduling_repair': usedShiftsDual}, alpha))
        stage_counter = 0

        for stage in self.set_stages:

            for shift in self.set_shifts.keys():
                summedDual = {}
                summedAlpha = 0
                if (shift, stage) in usedShifts.keys():
                    if usedShifts[shift,stage] < 0:
                        summedDual[shift-1, stage] =0.9*-self.modelClass.parameters.rescheduling_data_cost[
                                                                (self.modelClass.parameters.schedulingCategory.repair, stage)]
                        summedDual[shift, stage] =-self.modelClass.parameters.rescheduling_data_cost[
                                                                (self.modelClass.parameters.schedulingCategory.repair, stage)]
                        summedDual[shift+1, stage] =0.9*-self.modelClass.parameters.rescheduling_data_cost[(self.modelClass.parameters.schedulingCategory.repair, stage)]
                    summedAlpha = -usedShifts[shift,stage]
                    cuts[0].append(({'scheduling_repair':summedDual},summedAlpha))
        return cuts



    def branchNbounch_hfss(self, scenario, SolutionLimit, iteration,RHS):
        """
        Custom BnB algorithm to retrieve cuts during Branch-and-Bound, to be used in Benders Decomposition.
        """
        print('#' * 20)
        #print(f'Enter branch and bound for {scenario} subproblem!')
        self.relaxedModels[scenario].update()
        self.relaxedModels[scenario].optimize()
        #print(self.relaxedModels[scenario]._scenario)

        IP_model_Cut_Evaluation = self.integerModels[scenario].copy()
        IP_model_Cut_Evaluation._scenario = scenario

        numberOfFinalLeaves = 4
        FinalLeavesFound = 0

        self.bestIncumbantInteger = np.inf

        upper_bound = np.inf
        tree = self.tree
        IP_counter = 0
        # reset tree obj and incumbant
        for leaf in tree.keys():
            # reset incumbant?
            tree[leaf][2] = None
            # reset obj val
            tree[leaf][3] = None
            # reset leaf node status
            tree[leaf][5] = 'leaf'
        generated_cuts = {}  # tuple list of format (beta_multiplier, alpha)
        self.node_counter = len(tree.keys()) - 1  # root node
        print('Number of node in Tree ' + str(self.node_counter))

        self.relaxedModels[scenario].update()
        self.relaxedModels[scenario].Params.OutputFlag = 0

        #CandSolutionTree = [*tree.keys()]
        CandSolutionTree = [0]
        exploredNodes = []
        infeasibleNodes = []
        nodeCount = len(CandSolutionTree)
        #self.relaxedModels[scenario]._dummyLowerCombined[1, 'Mixing'].RHS = 5

        # until no candidate node is left
        while (len(CandSolutionTree) != 0):

            # get next node
            branch_node = CandSolutionTree.pop(-1)

            if len(CandSolutionTree) % 100 == 0:
                print(len(CandSolutionTree))
            # node is searched
            if branch_node != 0:
                tree[tree[branch_node][1]][5] = None
            #print(f'Enter into new branching Node {branch_node}, tree size {len(CandSolutionTree)}')
            exploredNodes.append(branch_node)
            # get new limits
            for (var, sense, new_limit, old_limit) in tree[branch_node][0]:
                # need to identify correct variable
                print(var.VarName)
                varLP = self.relaxedModels[scenario].getVarByName(var.VarName)
                stageString, shiftString = re.findall(r'\[.*?\]', varLP.VarName)[-1].strip('[]').split(',')
                # update bounds to represent branching
                if sense == 'lb':
                    #varLP.lb = new_limit
                    self.relaxedModels[scenario]._dummyLower[int(shiftString),stageString].RHS = new_limit + RHS['scheduling_repair'][stageString,int(shiftString)]
                    print(f'{stageString} {shiftString} with {new_limit}')
                if sense == 'ub':
                    varLP.ub = new_limit

            # optimize relaxed model
            self.relaxedModels[scenario].update()
            self.relaxedModels[scenario].optimize()

            extendTree = False
            run_IP_model = False
            if self.relaxedModels[scenario].Status == GRB.INFEASIBLE:
                print('model infeasible')
            if self.relaxedModels[scenario].Status == GRB.OPTIMAL:

                tree[branch_node][2] = self.relaxedModels[scenario].objVal
                print(f'Obj in node {self.relaxedModels[scenario].objVal} compared to best bound {self.bestIncumbantInteger}')
                if self.relaxedModels[scenario].ObjVal < self.bestIncumbantInteger:
                    # if no child exists and solution is not RMP incumbant
                    if tree[branch_node][4] == None:
                        if not tree[branch_node][3] == 'unsearch_incumbant_RMP':

                            extendTree = True
                    else:
                        CandSolutionTree.append(tree[branch_node][4][0])
                        CandSolutionTree.append(tree[branch_node][4][1])
                else:
                    # set node as leaf, as we can stop here
                    tree[branch_node][5] = 'leaf'

            elif self.relaxedModels[scenario].Status == GRB.INFEASIBLE:
                # need to pop all subsequent nodes
                #print('infeasible node ' + str(branch_node))
                self.remove_child_nodes(branch_node, tree, CandSolutionTree)
                infeasibleNodes.append(branch_node)

            # add bender cuts
            if self.relaxedModels[scenario].Status != GRB.INFEASIBLE:
                #print(self.relaxedModels[scenario].Status)
                # determine beta mulitplier, not using scenario to be able to copy variables
                dual_beta = self.get_duals_of_constraints(self.relaxedModels[scenario], IP=False, scenario="")

                # dual alpha in format single constant

                dual_alpha = self.get_duals_of_vars(self.relaxedModels[scenario], IP=False, scenario="",limitedVariables=tree[branch_node][0])
                print(dual_beta, dual_alpha)
                generated_cuts[branch_node] = [(dual_beta, dual_alpha)]

                # append all cuts from prdecessors
                if branch_node != 0:
                    #print(generated_cuts[tree[branch_node][1]])
                    if tree[branch_node][1] not in generated_cuts.keys():
                        print(tree[branch_node][1])
                    for cuts in generated_cuts[tree[branch_node][1]]:
                        generated_cuts[branch_node].append(cuts)

            #else:
                #print('Node ' + str(branch_node) + ' is infeasible')

            self.relaxedModels[scenario].update()
            # extend tree if not finished branching
            if extendTree:
                #print('extend tree')
                self.populateCandidates(self.relaxedModels[scenario], CandSolutionTree, tree, branch_node)
            # if finsihed branching, enter IP model to retrieve final leafs

            if (tree[branch_node][3] == 'unsearched_incumbant_RMP') and (FinalLeavesFound < numberOfFinalLeaves):
                tree[branch_node][3] = 'incumbant_RMP'
                tree[branch_node][5] = None
                print('Start to solve IP model in final node')
                FinalLeavesFound += 1
                # copy all determined bounds to IP
                # get new limits

                for (var, sense, new_limit, old_limit) in tree[branch_node][0]:
                    # get corresponding variable in model.IP
                    var_IP = self.integerModels[scenario].getVarByName(var.VarName)
                    #print(var.VarName)
                    #print(new_limit)
                    #print(sense)
                    # set required brancing limits
                    if var.VarName[:20] == "_rescheduling_repair":
                        stageString, shiftString = re.findall(r'\[.*?\]', var_IP.VarName)[-1].strip('[]').split(',')
                        shiftNumber = int(shiftString)

                        if sense == 'lb':
                            var_IP.lb = new_limit
                            self.lowerLimitRepairs[stageString, shiftNumber] = new_limit

                        if sense == 'ub':
                            var_IP.ub = new_limit
                            # update upper repair limit
                            self.upperLimitRepairs[stageString, shiftNumber] = new_limit

                # optimize IP model
                self.integerModels[scenario]._fileDir = 'solution_' + str(iteration)+'/' + scenario + '_' + str(branch_node)
                self.integerModels[scenario]._filePath = self.integerModels[scenario]._fileDir + '/leaf'
                if not os.path.exists(self.integerModels[scenario]._fileDir):
                    # if the corresponding scenario directory is not present then create it.
                    os.makedirs(self.integerModels[scenario]._fileDir)
                # for Average and Optimistic use previous solution as start
                self.integerModels[scenario].Params.SolFiles = self.integerModels[scenario]._filePath
                self.integerModels[scenario].Params.MIPGap = 0.05
                #self.integerModels[scenario].Params.SolutionLimit = SolutionLimit
                self.integerModels[scenario].Params.MIPfocus = 3


                self.integerModels[scenario].Params.OutputFlag = 1
                self.integerModels[scenario].Params.BestBdStop = self.bestIncumbantInteger
                self.integerModels[scenario].update()
                # for Average and Optimistic use previous solution as start
                if scenario == 'Pessimistic':
                    create_initial_solution(self.modelClass, self.integerModels[scenario], scenario)
                else:
                    pointerDict = {'Average':'Pessimistic', 'Optimistic': 'Average'}
                    prevScenarioSolutionPath = 'solution_' + str(iteration)+'/'  + pointerDict[scenario] + '_' + str(branch_node)
                    try:
                        for solution in os.listdir(prevScenarioSolutionPath):
                            filePath = prevScenarioSolutionPath + '/' + solution
                            print('read in ' + filePath)
                            self.integerModels[scenario].read(filePath)
                            self.integerModels[scenario].update()
                    except:
                        print('No node found')

                self.integerModels[scenario]._solutionProcedure = self
                self.integerModels[scenario].update()

                self.integerModels[scenario].optimize()

                for (stage, shift) in self.integerModels[scenario]._rescheduling_repair.keys():
                    if self.integerModels[scenario]._rescheduling_repair[stage, shift].X >= 0.1:
                        print(
                            f'{scenario},{stage}, {shift} with rescheduling  {self.integerModels[scenario]._rescheduling_repair[stage, shift].X}')

                # check if solution is a new incumbent
                if self.integerModels[scenario] != GRB.INFEASIBLE or self.integerModels[scenario] != GRB.UNBOUNDED:
                    if self.bestIncumbantInteger > self.integerModels[scenario].ObjVal:
                        self.bestIncumbantInteger = self.integerModels[scenario].ObjVal
                        #time.sleep(10)

                # generate cuts from stored solution
                fixed_start = time.time()
                filePathsToBeExamined = []
                pointerDict = {'Pessimistic':['Pessimistic'],'Average': ['Pessimistic','Average'], 'Optimistic': ['Pessimistic','Average','Optimistic'] }
                for examinPrevScenario in pointerDict[scenario]:
                    prevScenarioSolutionPath = 'solution_' + str(iteration) + '/' + examinPrevScenario + '_' + str(
                        branch_node)
                    for solution in os.listdir(prevScenarioSolutionPath):
                        filePathSolution = prevScenarioSolutionPath + '/' + solution
                        filePathsToBeExamined.append(filePathSolution)

                subNode = 0

                # add cuts for identified solutions during gurobi optimisation
                for solution in filePathsToBeExamined:

                    subNode += 1
                    cutKey = str(branch_node) + ',' + str(subNode)
                    generated_cuts[cutKey] = []

                    IP_model_Cut_Evaluation.Params.OutputFlag = 0
                    filePath = solution
                    IP_model_Cut_Evaluation.reset()
                    IP_model_Cut_Evaluation.update()
                    IP_model_Cut_Evaluation.read(filePath)
                    IP_model_Cut_Evaluation.update()

                    # set all integer variables other than 0
                    # reset limits

                    for (var, sense, new_limit, old_limit) in tree[branch_node][0]:
                        if sense == 'lb':
                            var.LB = 0
                        if sense == 'ub':
                            var.UB = 100
                    self.relaxedModels[scenario].update()
                    
                    test = []
                    counter = 0
                    oldSol = -np.inf
                    for var in IP_model_Cut_Evaluation.getVars():
                        if var.VType == 'B':
                            if not var.VarName[:28] == "_repair_of_batch_executed_in":
                                if var.Start > 0.1:
                                    varLP =  self.relaxedModels[scenario].getVarByName(var.VarName)
                                    test.append((varLP,""))
                                    varLP.lb = 1
                            else:
                                batch,machine, shiftString = re.findall(r'\[.*?\]', var.VarName)[-1].strip('[]').split(
                                    ',')
                                shiftNumber = int(shiftString)
                                var.Start > 0.1





                    self.relaxedModels[scenario].update()
                    self.relaxedModels[scenario].optimize()
                    oldSol = self.relaxedModels[scenario].ObjVal
                    #print(f'solved model with obj {self.relaxedModels[scenario].ObjVal}')

                    # append all cuts from prdecessors

                    # determine beta mulitplier, not using scenario to be able to copy variables
                    dual_beta = self.get_duals_of_constraints(self.relaxedModels[scenario], IP=False,
                                                              scenario="")
                    # print(dual_beta)
                    # dual alpha in format single constant

                    dual_alpha = self.get_duals_of_vars(self.relaxedModels[scenario], IP=False,
                                                        scenario="",
                                                        limitedVariables=test)

                    if len(dual_beta['scheduling_repair'].keys())+len(dual_beta['lowerBoundRepair'].keys())!=0:
                        if len(generated_cuts[cutKey])==0:
                            generated_cuts[cutKey] = [(dual_beta, dual_alpha)]
                        else:
                            generated_cuts[cutKey].append((dual_beta, dual_alpha))
                        # append all cuts from prdecessors
                    else:
                        #print(f'ignore solution {solution}')
                        del generated_cuts[cutKey]

                    # try to get more cuts by opimizing and adding rescheudling branching based on identified soltuion
                    for var in IP_model_Cut_Evaluation.getVars():
                        if var.VType == 'I':
                            if var.VarName[0:20] =='_rescheduling_repair' and var.Start > 0:
                                stageString, shiftString = re.findall(r'\[.*?\]', var.VarName)[-1].strip('[]').split(
                                    ',')
                                shiftNumber = int(shiftString)

                                self.relaxedModels[scenario]._dual_var['rescheduling_repair'][shiftNumber,stageString].lb = var.Start
                                self.relaxedModels[scenario].update()
                                self.relaxedModels[scenario].optimize()
                                # determine beta mulitplier, not using scenario to be able to copy variables
                                dual_beta = self.get_duals_of_constraints(self.relaxedModels[scenario], IP=False,
                                                                          scenario="")


                                # print(dual_beta)
                                # dual alpha in format single constant

                                dual_alpha = self.get_duals_of_vars(self.relaxedModels[scenario], IP=False,
                                                                    scenario="",
                                                                    limitedVariables=test)
                                dual_alpha += self.relaxedModels[scenario]._dual_var['rescheduling_repair'][shiftNumber,stageString].RC * var.Start


                                self.relaxedModels[scenario]._dual_var['rescheduling_repair'][
                                    shiftNumber, stageString].lb = 0
                                generated_cuts[cutKey].append((dual_beta,dual_alpha))



                                        #print(len(test))
                    if branch_node != 0:
                        for cuts in generated_cuts[tree[branch_node][1]]:
                            generated_cuts[cutKey].append(cuts)
                    # reset branching bounds
                    for var in IP_model_Cut_Evaluation.getVars():
                        if var.VType == 'B':
                            if var.Start > 0.1:
                                varLP =  self.relaxedModels[scenario].getVarByName(var.VarName)
                                varLP.lb = 0
                            else:
                                varLP.ub = 1



                # retrieve all incumbant solutions

                # reset IP model to orginal formulation
                for (var, sense, new_limit, old_limit) in tree[branch_node][0]:
                    # get corresponding variable in model.IP
                    var_IP = self.integerModels[scenario].getVarByName(var.VarName)
                    if var.VarName[:20] == "_rescheduling_repair":
                        # get coresponding shift and string
                        stageString, shiftString = re.findall(r'\[.*?\]', var_IP.VarName)[-1].strip('[]').split(',')
                        shiftNumber = int(shiftString)
                        if sense == 'lb':
                            var_IP.LB = 0

                        if sense == 'ub':
                            var_IP.UB = 100


                self.integerModels[scenario].update()

            # reset limits
            for (var, sense, new_limit, old_limit) in tree[branch_node][0]:
                varLP = self.relaxedModels[scenario].getVarByName(var.VarName)
                stageString, shiftString = re.findall(r'\[.*?\]', varLP.VarName)[-1].strip('[]').split(',')
                if sense == 'lb':
                    #varLP.LB = 0
                    self.relaxedModels[scenario]._dummyLower[int(shiftString), stageString].RHS = 0
                if sense == 'ub':
                    varLP.UB = 100
            self.relaxedModels[scenario].update()


        # remove all cuts that are not leaves

        # determine leaves

        for id, node in tree.items():
            if not node[5] == 'leaf':
                if not id in infeasibleNodes:
                    if id in exploredNodes:
                        del generated_cuts[id]

        #print(f'Best found solution in scenario {scenario}')
        #print(self.bestIncumbantInteger)
        self.incumbantSolution[scenario][iteration] = self.bestIncumbantInteger
        file = open('important', 'wb')
        pickle.dump(self.incumbantSolution, file )

        #print(generated_cuts)

        return generated_cuts, False

    def remove_child_nodes(self, remove_node, tree, CandSolutionTree):
        #print('please add recourse pop function')
        if tree[remove_node][4] != None:
            self.remove_child_nodes(tree[remove_node][4][0], tree, CandSolutionTree)
            self.remove_child_nodes(tree[remove_node][4][1], tree, CandSolutionTree)

    def get_duals_of_constraints(self, model, IP: False, scenario):
        # list all constraints, for which duals are required for IP model
        if IP:
            dual_constr_IP = {}
            dual_constr_IP['scheduling_repair'] = {
                (shift, stage): model.getConstrByName(f'{scenario}_rescheduling_cost_repair_constraint[{shift},{stage}]') for shift
                in
                self.set_shifts.keys() for stage in self.set_stages}

            dual_constr_IP['lowerLimit'] = {
                stage: model.getConstrByName(f'{scenario}_reschedulingLowerBound[{stage}]') for stage in self.set_stages}

            cons_for_duals = dual_constr_IP
        else:
            cons_for_duals = model._dual_constr

        duals_in_leaf = gp.tupledict()
        duals_in_leaf['scheduling_repair'] = gp.tupledict()

        # retrieve duals for all required constraints, other constraints duals are retrieved in get_duals_of_varialbes as they do not depend on first stage decision
        for (shift, stage) in cons_for_duals['scheduling_repair'].keys():

            dual_scheduling = cons_for_duals['scheduling_repair'][(shift, stage)].Pi


            if (dual_scheduling)!=0:
                duals_in_leaf['scheduling_repair'][(shift, stage)] = dual_scheduling


        #print(cons_for_duals['scheduling_repair'][(1, 'Mixing')].Pi)

        dual_lower = gp.tupledict()


        duals_in_leaf['lowerBoundRepair'] = dual_lower

        return duals_in_leaf


    def get_duals_of_vars(self, model, IP: False, scenario, limitedVariables):
        """
        Retrieve duals from all variables by using the grubis RC attribute
        :param model:
        :param IP:
        :param scenario:
        :param limitedVariables:
        :return:
        """
        start_time = time.time()

        dual_alpha = 0

        if IP or True:
            dual_alpha = 0
            for var in model.getVars():
                if var.X == var.lb:
                    dual_alpha += var.lb * var.RC
                if var.X == var.ub:
                    dual_alpha += var.ub * var.RC


        # append duals for constraint that are not impacted by first stage decision - constant for this
        cons = model.getConstrs()
        for con in cons:
            if con.RHS !=0 and con.Pi !=0:
                if con.ConstrName[:36]!='_rescheduling_cost_repair_constraint':
                    if con.ConstrName[:23]!='_reschedulingLowerBound':
                        dual_alpha += (con.RHS * con.Pi)

        return dual_alpha

        # set identified limits



    def populateCandidates(self, model, currentCandidates: [], tree: {}, currentNode: 0):
        # iterate through integer variables to check if they are fractional
        foundBranch = False
        highestValue = 0
        selectVariable = None
        branchedStages = []

        # decide on when to do maintenance first

        search = True
        while search and not foundBranch:
            for var in model._branchingVars:
                if (np.absolute(var.X - round(var.X, 0)) >= 0.01):
                    if var.VarName[:20] == "_rescheduling_repair":
                        #stageString, shiftString = re.findall(r'\[.*?\]', var.VarName)[-1].strip('[]').split(',')
                        selectVariable = var
                        foundBranch = True
                        search = False
                        break
            search = False


        foundBranch = False
        if foundBranch:
            var = selectVariable
            print(f'new variable var {var.VarName} {var.X}')
            #print('found new branch')
            # found fractional value
            # determine child nodes, by limiting ub and lb of variables respectively

            # create child node with new ub on variables
            new_ub = np.floor(var.X)
            old_ub = var.ub
            Node_ub = [(var, 'ub', new_ub, old_ub)]

            # create child node with new ub on variables
            new_lb = np.ceil(var.X)
            old_lb = var.lb
            Node_lb = [(var, 'lb', new_lb, old_lb)]

            # append node limits in root node
            for nodeLimits in tree[currentNode][0]:
                # only add other limits
                if nodeLimits[0].VarName != var.VarName:
                    Node_ub.append(nodeLimits)
                    Node_lb.append(nodeLimits)
                # check if it is another lb when ub
                elif nodeLimits[2] == 'lb':
                    Node_ub.append(nodeLimits)
                elif nodeLimits[2] == 'ub':
                    Node_lb.append(nodeLimits)

            # append identified nodes to tree

            self.node_counter += 1
            tree[self.node_counter] = [Node_ub, currentNode, None, None, None, None]
            currentCandidates.insert(0,self.node_counter)

            self.node_counter += 1
            tree[self.node_counter] = [Node_lb, currentNode, None, None, None, None]
            currentCandidates.append(self.node_counter)

            # safe child nodes in tree
            tree[currentNode][4] = (self.node_counter - 1, self.node_counter)


        # if no branch is found, all variables are integer, thus new incumbant solution
        if not foundBranch:
            tree[currentNode][3] = 'unsearched_incumbant_RMP'
            # if solution is integer, set node to leafe node
            tree[currentNode][5] = 'leaf'
            #print('Found unfinsihed incumbant solution ' + str(currentNode))

        return currentCandidates

    def get_duals_during_manual_branchNbound(self):
        print('test')
        

def create_initial_solution(modelClass, model, scenario):
    '''
    Determine an initial solution by using a greedy construction heuristic. In detail, batches are assigned in
    sequence, starting with the first job, in sequence of the stages, to the machine with the earliest possible
    starting time.
    1 Create decision variables as dictionaries
    2 Iterate through jobs:
        2.1 Iterate through stages:
            Determine earliest starting time of position, considereing previous position and cross dependand position.
            Determine if subsequent slot would violate any maintenance activity, if so schedule maintenance. If repair is required. Also schedule machine shutdown and warm-up.
            Update all time trackers

    :param data:
    :param config:
    :return: None
    '''
    print(f'### Starting to create initial solution for {scenario}. ###')
    import numpy as np
    greedy_solution = None
    parameters = modelClass.parameters
    config = parameters.config

    # track operation time of machine hfss._start_warmUp_batch_machine_slot
    enKeyWarm = config.data.parameter_name.energyConsumption['warmUp']
    enKeyCool = config.data.parameter_name.energyConsumption['coolDown']
    enKeyOperating = config.data.parameter_name.energyConsumption['operating']
    enKeyIdle = config.data.parameter_name.energyConsumption['idle']

    batch_dependency = model._batch_dependency[1]
    batch_processing_time = model._batch_processing_time[1]

    # create dict to track operation time per maintenance
    timetracker_maintenance_machine = {
        machine: {maintenance: 0 for maintenance in model._set_maintenance_by_machine[machine]} for machine in
        parameters.machines}
    timetracker_repair_machine = {
        machine: {repair: 0 for repair in model._set_repair_by_machine[machine]} for machine in
        parameters.machines}

    # set batch -1, to initial machine state
    for machine in parameters.machines:
        for repair in model._set_repair_by_machine[machine]:
            timetracker_repair_machine[machine][repair] = parameters.currentMachineAge[machine]

    # fixed repairs
    additional_blockerRepair_position_machine = {(repair, position, machine): 0  for position, machine in model._set_position_per_machine for repair in model._set_repair_by_machine[machine]}
    fixRepair_position_machine = {(repair, position, machine): False for position, machine in
                                                 model._set_position_per_machine for repair in
                                                 model._set_repair_by_machine[machine]}
    additional_blockerEnd_position_machine = {(position, machine): 0  for position, machine in model._set_position_per_machine }

    # repair per shift
    # shifts: {shift_id: (lower_slot, uppper_slot, shift, day) modelClass.set_shifts
    scheduled_repair_per_shift_per_stage = {(stage, shift): 0 for shift in parameters.set_shifts for stage in parameters.set_stages}

    # Create DV variables dictionary
    generated_patterns = 1

    # Conitnuous Decision Variables
    start_batch_machine = {(position, machine): 0 for (position, machine) in model._set_position_per_machine}
    end_batch_machine = {(position, machine): 0 for (position, machine) in model._set_position_per_machine}
    start_coolDown_batch_machine = {(position, machine): 0 for (position, machine) in
                                    model._set_position_per_machine}
    end_coolDown_batch_machine = {(position, machine): 0 for (position, machine) in model._set_position_per_machine}
    start_repair_batch_machine = {(position, machine): 0 for (position, machine) in model._set_position_per_machine}
    end_repair_batch_machine = {(position, machine): 0 for (position, machine) in model._set_position_per_machine}
    start_warmUp_batch_machine = {(position, machine): 0 for (position, machine) in model._set_position_per_machine}
    end_warmUp_batch_machine = {(position, machine): 0 for (position, machine) in model._set_position_per_machine}
    start_service_batch_machine = {(position, machine): 0 for (position, machine) in model._set_position_per_machine}
    end_service_batch_machine = {(position, machine): 0 for (position, machine) in model._set_position_per_machine}
    # Binary Decision Variables
    start_batch_machine_slot = {(position, machine): 0 for (position, machine) in model._set_position_per_machine}
    end_batch_machine_slot = {(position, machine): 0 for (position, machine) in model._set_position_per_machine}

    # Binary Decision Variables for cool down
    start_coolDown_batch_machine_slot = {(position, machine): 0 for (position, machine) in
                                         model._set_position_per_machine}
    end_coolDown_batch_machine_slot = {(position, machine): 0 for (position, machine) in
                                       model._set_position_per_machine}

    # Binary Decision Variables for mr block
    start_repair_batch_machine_slot = {(position, machine): 0 for (position, machine) in
                                   model._set_position_per_machine}
    end_repair_batch_machine_slot = {(position, machine): 0 for (position, machine) in model._set_position_per_machine}
    
    # Binary Decision Variables for service block
    start_service_batch_machine_slot = {(position, machine): 0 for (position, machine) in
                                   model._set_position_per_machine}
    end_service_batch_machine_slot = {(position, machine): 0 for (position, machine) in model._set_position_per_machine}


    # Binary Decision Variables for WarmUp
    start_warmUp_batch_machine_slot = {(position, machine): 0 for (position, machine) in
                                       model._set_position_per_machine}
    end_warmUp_batch_machine_slot = {(position, machine): 0 for (position, machine) in
                                     model._set_position_per_machine}

    # Integer Variables
    rescheduling_repair = {(stage, shift): 0 for (stage, shift) in model._rescheduling_repair.keys()}

    # rearrange cross dependency
    cross_dependency_by_batch_machine = {(position, machine): [] for (position, machine) in
                                         model._set_position_per_machine}
    for (prev_batch, prev_machine, batch, machine), dependency in batch_dependency.items():
        if dependency == 1:  # only select those with actual dependency
            cross_dependency_by_batch_machine[batch, machine].append((prev_batch, prev_machine))

    # set maintenance batch machine to 0
    for dv_variable in model._maintenance_batch_machine.values():
        dv_variable.Start = 0

    # set repair batch machine to 0
    for dv_variable in model._repair_batch_machine.values():
        dv_variable.Start = 0

    # set duation of block to zero
    for dv_variable in model._duration_service_after_batch.values():
        dv_variable.Start = 0

    # set duation of block to zero
    for dv_variable in model._duration_repair_after_batch.values():
        dv_variable.Start = 0

    # set indicator for shut down
    for dv_variable in model._shut_down_after_batch.values():
        dv_variable.Start = 0

    for dv_variable in model._repair_batch_shift.values():
        dv_variable.Start = 0

    # fixed repair batch assignments


    # iterate through stages
    # iterate through each stage in sequence for specified job
    for index, stage in enumerate(parameters.set_stages):
        # found feasible stage assignment
        feasibleStage = False
        exitCounter = 0
        while not feasibleStage:
            earliestRepair = {shift: np.inf for shift in parameters.set_shifts}
            earliestRepairNPostionNMachine = {shift: None for shift in parameters.set_shifts}
            latestRepair = {shift: np.inf for shift in parameters.set_shifts}
            latestRepairNPostionNMachine = {shift: None for shift in parameters.set_shifts}

            # iterate through all machines per stage
            for machine in parameters.machines_per_stage[stage]:
                # iterate thorugh all positions of machines
                for position in range(0, parameters.positions_per_machine[machine]):
                    # determine ending time of previous batch on same machine
                    prev_batch_max = 0
                    if position >= 1:  # no need to check frist position
                        prev_batch_max = end_service_batch_machine[(position - 1, machine)]

                    # determine ending time of previous batch on previous stage
                    prev_stage_max = 0
                    if index >= 1:  # no need to check first stage
                        for (prev_batch, prev_machine) in cross_dependency_by_batch_machine[position, machine]:
                            if prev_stage_max <= end_batch_machine[(prev_batch, prev_machine)]:
                                prev_stage_max = end_batch_machine[(prev_batch, prev_machine)]

                    # determine if previous stage or previous position has the later time.
                    if prev_batch_max >= prev_stage_max:
                        start_batch_machine[position, machine] = prev_batch_max
                    else:
                        start_batch_machine[position, machine] = prev_stage_max

                    # add processing time to start time to determine end time
                    if batch_processing_time[(position, machine)] > 0:
                        end_batch_machine[(position, machine)] = start_batch_machine[position, machine] + \
                                                                 batch_processing_time[(position, machine)]
                    else:
                        end_batch_machine[(position, machine)] = start_batch_machine[position, machine] + parameters.lowest_processTime_per_machine[machine]

                    # determine starting time slot for batch on machine
                    start_slot = int(start_batch_machine[position, machine])
                    start_batch_machine_slot[(position, machine)] = start_slot

                    # determine ending time slot for batch on machine
                    end_slot = int(end_batch_machine[position, machine])
                    end_batch_machine_slot[(position, machine)] = end_slot

                    # check if subsequent batch would violate maintenance time
                    total_maintenance_time_after_batch = 0
                    for maintenance, mtimetracker in timetracker_maintenance_machine[machine].items():
                        # increase time tracker for maintenance task
                        timetracker_maintenance_machine[machine][maintenance] += batch_processing_time[(position, machine)]

                        schedule_maintenance = False

                        # check if subsequent batch would violate maintenance time
                        if position < parameters.positions_per_machine[machine] - 1:
                            schedule_maintenance = model._parameter_maintenance[(maintenance, machine)][
                                                       config.data.maintenance_name.upper_limit] < (
                                                               timetracker_maintenance_machine[machine][maintenance] + batch_processing_time[
                                                           (position + 1, machine)])


                        model._maintenance_operationTime_batch_machine[(maintenance, position, machine)].Start = timetracker_maintenance_machine[machine][maintenance]


                        # if maintenance due, reset tracker and add corresponding service schedule. e.g. duration
                        if schedule_maintenance:
                            timetracker_maintenance_machine[machine][maintenance] = 0
                            total_maintenance_time_after_batch += \
                            model._parameter_maintenance[(maintenance, machine)][
                                config.data.maintenance_name.duration]
                            model._maintenance_operationTime_batch_machine[
                                (maintenance, position, machine)].Start = 0
                            # set
                            model._maintenance_batch_machine[(maintenance, position, machine)].Start = 1




                    # check if subsequent batch would violate repair time
                    total_repair_time_after_batch = 0
                    schedule_repair = False
                    for repair, timetracker in timetracker_repair_machine[machine].items():
                        # increase time tracker for repair task
                        timetracker_repair_machine[machine][repair] += batch_processing_time[(position, machine)]


                        # check if subsequent batch would violate repair time
                        if position < parameters.positions_per_machine[machine] - 1:
                            schedule_repair = model._parameter_repair[(repair, machine)][
                                                  config.data.maintenance_name.upper_limit][scenario] < (
                                                      timetracker_repair_machine[machine][repair] + batch_processing_time[
                                                  (position + 1, machine)]+additional_blockerRepair_position_machine[repair, position+1, machine])

                        model._repair_operationTime_batch_machine[(repair, position, machine)].Start = timetracker_repair_machine[machine][repair]

                        # if repair is due, time trakcer needs to be considered, time between positions increased and shutdown and warm-up considerd
                        if schedule_repair:
                            timetracker_repair_machine[machine][repair] = 0
                            total_repair_time_after_batch += model._parameter_repair[(repair, machine)][
                                config.data.maintenance_name.duration]
                            model._repair_operationTime_batch_machine[(repair, position, machine)].Start = 0
                            model._repair_batch_machine[(repair, position, machine)].Start = 1

                            start_repair = end_batch_machine[(position, machine)]+parameters.time_for_state_transition[(
                                                                      config.model_discrete.decision_variables.cool_down,
                                                                      machine)] + total_maintenance_time_after_batch + additional_blockerEnd_position_machine[(position, machine)]

                            # determine the shift in which the repair is executed
                            for shift, (lowerSlot, upperSlot, _, _) in parameters.set_shifts.items():
                                if start_repair >= lowerSlot:
                                    if start_repair<= upperSlot:
                                        if earliestRepair[shift] > start_repair:
                                            earliestRepair[shift] = start_repair
                                            earliestRepairNPostionNMachine[shift] = (repair, position, machine)
                                        if latestRepair[shift] > (upperSlot - start_repair):
                                            latestRepair[shift] = (upperSlot - start_repair)
                                            latestRepairNPostionNMachine[shift] = (position, machine)


                    shut_down_required = 0
                    # if any repair schedule, also warmup and shutdown required
                    if total_repair_time_after_batch > 0:
                        shut_down_required = 1
                        model._shut_down_after_batch[(position, machine)].Start = 1


                    # set corresponding variables in model
                    model._duration_service_after_batch[
                        (position, machine)].Start = total_maintenance_time_after_batch

                    model._duration_repair_after_batch[
                        (position, machine)].Start = total_repair_time_after_batch

                    # determine cool down start and end time
                    start_coolDown_batch_machine[(position, machine)] = end_batch_machine[(position, machine)]
                    end_coolDown_batch_machine[(position, machine)] = start_coolDown_batch_machine[
                                                                          (position, machine)] + shut_down_required * \
                                                                      parameters.time_for_state_transition[(
                                                                      config.model_discrete.decision_variables.cool_down,
                                                                      machine)]
                    start_repair_batch_machine[(position, machine)] = end_coolDown_batch_machine[(position, machine)] + additional_blockerEnd_position_machine[(position, machine)]
                    end_repair_batch_machine[(position, machine)] = start_repair_batch_machine[(
                    position, machine)]+  total_repair_time_after_batch
                    start_warmUp_batch_machine[(position, machine)] = end_repair_batch_machine[(position, machine)]
                    end_warmUp_batch_machine[(position, machine)] = start_warmUp_batch_machine[
                                                                        (position, machine)] + shut_down_required * \
                                                                    parameters.time_for_state_transition[(
                                                                    config.model_discrete.decision_variables.warm_up,
                                                                    machine)]
                    start_service_batch_machine[(position, machine)] = end_warmUp_batch_machine[(position, machine)]
                    end_service_batch_machine[(position, machine)] = start_service_batch_machine[(
                    position, machine)] + total_maintenance_time_after_batch
                    

                    # set corresoponding start time variables
                    start_coolDown_batch_machine_slot[(position, machine)] = int(
                        start_coolDown_batch_machine[(position, machine)])
                    end_coolDown_batch_machine_slot[(position, machine)] = int(
                        end_coolDown_batch_machine[(position, machine)])
                    start_repair_batch_machine_slot[(position, machine)] = int(start_repair_batch_machine[(position, machine)])
                    end_repair_batch_machine_slot[(position, machine)] = int(end_repair_batch_machine[(position, machine)])
                    start_service_batch_machine_slot[(position, machine)] = int(start_service_batch_machine[(position, machine)])
                    end_service_batch_machine_slot[(position, machine)] = int(end_service_batch_machine[(position, machine)])
                    if total_repair_time_after_batch > 0:
                        for shift, (lowerSlot, upperSlot, _, _) in parameters.set_shifts.items():
                            if int(start_repair_batch_machine[(position, machine)]) <= upperSlot:
                                rescheduling_repair[stage, shift] = 1
                                model._repair_batch_shift[position, machine, shift].Start = 1
                                scheduled_repair_per_shift_per_stage[stage, shift] += 1
                                break

                    start_warmUp_batch_machine_slot[(position, machine)] = int(
                        start_warmUp_batch_machine[(position, machine)])
                    end_warmUp_batch_machine_slot[(position, machine)] = int(
                        end_warmUp_batch_machine[(position, machine)])
            model.update()
            # check if solution feasible
            feasibleStage = True


    for (stage, shift), startSol in scheduled_repair_per_shift_per_stage.items():

        model._rescheduling_repair[stage, shift].Start = startSol

    model.update()


    # all variables binary need to be set
    for (position, machine) in model._start_batch_machine.keys():
        model._start_batch_machine[(position, machine)].Start = start_batch_machine[position, machine]
        #model._end_batch_machine[(position, machine)].Start = end_batch_machine[position, machine]
        stage = parameters.stage_per_machine[machine]
        if parameters.input_data[parameters.config.data.schedulingPar][stage][
            parameters.config.data.scheduling_parameters.shutDown]:
            model._startTime_shutDown[(position, machine)].Start = start_coolDown_batch_machine[(position, machine)]
            model._startTime_warmUp[(position, machine)].Start = start_warmUp_batch_machine[(position, machine)]
            model._startTime_serviceBlock[(position, machine)].Start = start_service_batch_machine[(position, machine)]

    model.update()

    # set each starting slot to zero, used slot are updated later
    for (position, machine, slot) in model._start_repair_batch_machine_slot.keys():
        model._start_repair_batch_machine_slot[(position, machine, slot)].Start = 0

    # set only used slot for starting time of repair
    for (position, machine) in model._start_batch_machine.keys():
        model._start_repair_batch_machine_slot[(position, machine, start_repair_batch_machine_slot[
            (position, machine)])].Start = 1

        stage = parameters.stage_per_machine[machine]

    model.update()

    model.update()


    # determine completion time
    completion_time_max = 0
    for machine in model._machines_per_stage[model._stages[-1]]:
        if completion_time_max <= model._end_batch_machine[
            (model._postions_per_machine[machine] - 1, machine)].Start:
            completion_time_max = model._end_batch_machine[
                (model._postions_per_machine[machine] - 1, machine)].Start

    model._completion_time_max.Start = completion_time_max
    model.update()
