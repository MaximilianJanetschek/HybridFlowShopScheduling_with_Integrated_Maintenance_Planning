import gurobipy as gp
from gurobipy import *
import numpy as np
import time


class FristStage_MasterProblem():

    def __init__(self, parameters, layout, shifts, config):
        # define scenarios
        self.scenarios = [scenario for scenario in config.data.maintenance_name.scenarios_prob.keys()]
        self.scenarios_prob = {scenario: prob for scenario, prob in config.data.maintenance_name.scenarios_prob.items()}
        print(self.scenarios_prob)
        self.scheduling_cost = parameters[config.data.scheduling]  # format {(maintenance, stage): cost, ..., (repair,stage): cost, ...}
        self.schedulingCategory = config.data.scheduling_cost.category_name
        self.layout = layout
        self.shifts = shifts
        self.config = config
        self.generate_master_problem()


    def generate_master_problem(self):
        model = gp.Model('First Stage Master Problem')
        model._bigM = 1000000
        model.Params.OutputFlag = 0
        model._constraint_sum_u = None
        model._dv_q = gp.tupledict()
        model._dv_u = gp.tupledict()
        model._constraitn_q_upper = gp.tupledict()
        model._constraitn_q_lower = gp.tupledict()
        model._constraitn_q = gp.tupledict()

        model._set_leaf_scenarios = {scenario: [] for scenario in self.scenarios}
        
        # define first stage decision variables
        # just set first stage variables to 0
        model._scheduling_repair = model.addVars(self.layout, self.shifts, vtype=GRB.INTEGER, name='firstStage_scheduling_repair')

        # add value function approx
        model._valueFunction_lower = model.addVars(self.scenarios,vtype=GRB.CONTINUOUS)

        # define objective

        # calculate obj of cx
        firstStage_Cost = gp.LinExpr()


        # cost for scheduled repair
        print(self.scheduling_cost)
        firstStage_Cost += gp.quicksum(model._scheduling_repair[(stage, shift)] * self.scheduling_cost[
            (self.schedulingCategory.repair, stage)] for stage in self.layout for shift in self.shifts)
        model.update()

        # lower limit by value function approximation
        valueFunctionApprox = gp.LinExpr()
        valueFunctionApprox += gp.quicksum(model._valueFunction_lower[scenario] * self.scenarios_prob[scenario] for scenario in self.scenarios)


        model.setObjective(firstStage_Cost + valueFunctionApprox, GRB.MINIMIZE)


        self.MasterProblem = model

    def get_firstStage_variables(self):
        model = self.MasterProblem
        if not model.Status == GRB.OPTIMAL:
            print('Error! Master Problem is not solvable to optimality!')
            raise ValueError
        scheduled_repair = gp.tupledict()
        for stage in self.layout:
            for shift in self.shifts:
                scheduled_repair[(stage, shift)] = model._scheduling_repair[(stage, shift)].X
                if scheduled_repair[(stage, shift)] != 0:
                    print((stage, shift))
                    print(scheduled_repair[(stage, shift)])

        first_stage_key = self.config.model_discrete.decision_variables
        first_stage_decision = {
                                first_stage_key.scheduled_repair: scheduled_repair}

        return first_stage_decision


    def extended_master_problem(self, cuts_per_scenario, iteration):   # scenario format {scenario 1: cuts, scenario 2: cuts}
        model = self.MasterProblem

        model.remove(model._constraitn_q)
        # remove all unnecessary leaves
        for (leaf, scenario) in model._dv_q.keys():
            if leaf not in cuts_per_scenario[scenario].keys():
                print(f'Remvoe leaf {leaf} {scenario}')
                model.remove(model._dv_q[(leaf, scenario)])
                model.remove(model._dv_u[(leaf, scenario)])
                model.remove(model._constraitn_q_upper[(leaf, scenario)])
                model.remove(model._constraitn_q_lower[(leaf, scenario)])
                for c in model._constraitn_q.select('*', leaf, scenario):
                    model.remove(c)
                model._set_leaf_scenarios[scenario].remove(leaf)
                del self.MasterProblem._dv_u[(leaf, scenario)]
                del self.MasterProblem._dv_q[(leaf, scenario)]
        model.remove(model._constraitn_q)
        model.update()


        # add new variables
        # add new leave nodes
        for scenario in cuts_per_scenario.keys():
            print(scenario)
            for leaf in cuts_per_scenario[scenario].keys():
                print(leaf)
                if leaf not in model._set_leaf_scenarios[scenario]:
                    model._set_leaf_scenarios[scenario].append(leaf)
                    # add new decision variables
                    model._dv_q[(leaf, scenario)] = model.addVar(vtype = GRB.CONTINUOUS)

                    #print('adding variables ' + str(leaf) + ' for scenario ' + str(scenario))
                    if iteration != 1:
                        # determine column to add

                        constraint_coefficienlist = [1]
                        constraint_list = [model._constraint_sum_u[scenario]]
                        model._dv_u[(leaf, scenario)] = model.addVar(vtype = GRB.BINARY, column = Column(constraint_coefficienlist, constraint_list))
                    else:
                        model._dv_u[(leaf, scenario)] = model.addVar(vtype = GRB.BINARY)

                    # add constraints

                    # add all identified cuts



                    # value function lower than any tree cut
                    model._constraitn_q_upper[(leaf, scenario)] = model.addConstr(model._valueFunction_lower[scenario] <= model._dv_q[(leaf, scenario)], name='test1')
                    model._constraitn_q_lower[(leaf, scenario)] = model.addConstr(model._valueFunction_lower[scenario] >= model._dv_q[(leaf, scenario)] - model._bigM * ( 1 - model._dv_u[(leaf, scenario)] ),name='test')

                # add all identified cuts
                #print(scenario)
                #print(cuts_per_scenario[scenario][leaf])
                for c_id, cut in enumerate(cuts_per_scenario[scenario][leaf]):
                    print(cut[0]['scheduling_repair'])
                    print(cut[1])
                    model._constraitn_q[(c_id, leaf, scenario)] = model.addConstr(model._dv_q[(leaf, scenario)] >=  gp.quicksum(cut[0]['scheduling_repair'][(shift, stage)] * model._scheduling_repair[stage, shift] for (shift, stage) in cut[0]['scheduling_repair'].keys())  + cut[1],name=f'test3 {scenario}{leaf}')
                    model.update()
        if iteration == 1:
            model._constraint_sum_u = model.addConstrs((model._dv_u.sum('*', scenario) == 1
                for scenario in self.scenarios), name='test3')
        #model._scheduling_repair['Mixing', 9].lb = 2
        #model._scheduling_repair['Slitting', 15].lb = 1
        model.update()
        for scenario in cuts_per_scenario.keys():

            for leaf in cuts_per_scenario[scenario].keys():
                model._dv_u[leaf,scenario].Start =1
                break


    def solve_MP(self):
        self.MasterProblem.Params.OutputFlag=1
        self.MasterProblem.update()

        self.MasterProblem.optimize()
        for leaf, scenario in self.MasterProblem._dv_u.keys():
            if self.MasterProblem._dv_u[(leaf, scenario)].X>=0.5:
                print(leaf, scenario)
                print(self.MasterProblem._dv_u[(leaf, scenario)].X)
                print(self.MasterProblem._dv_q[(leaf, scenario)].X)
                print(self.MasterProblem._valueFunction_lower[scenario].X)



    def get_theta_value(self):
        # theta as sum of lower function values time prob
        model = self.MasterProblem
        self.scenarios_prob
        theta = 0
        for key, valueFunction in model._valueFunction_lower.items():
            theta += self.scenarios_prob[key] * valueFunction.X

        return theta




class ScenarioProblem():

    def __init__(self, right_hand_side):
        self.generate_first_subproblem(right_hand_side=right_hand_side)
        self.tree = {0:[[],None, None, None, None, None]} # format Node_Number: set of limits, predecessor, incumbant?, obj val, childs tuple, leaf node?
    def generate_first_subproblem(self, right_hand_side):
        subproblem = gp.Model()
        set_y_int = [1,2,3]
        set_y_real = [4, 5, 6]
        yd = subproblem.addVars(set_y_int, vtype=GRB.CONTINUOUS, name= 'integer')
        subproblem._yd = yd
        subproblem._set_yd = set_y_int
        yr = subproblem.addVars(set_y_real, vtype=GRB.CONTINUOUS, name='real')
        subproblem._yr = yr
        subproblem._constraint = subproblem.addConstr(2*yd[1] +5*yd[2] -2*yd[3]-2*yr[4]+5*yr[5]+5*yr[6] == right_hand_side, name='constraint')
        subproblem.setObjective(6*yd[1] +4*yd[2] +3*yd[3]+4*yr[4]+5*yr[5]+7*yr[6], GRB.MINIMIZE)
        self.subproblem = subproblem

    def change_RHS(self, newRightHandSide):
        self.subproblem._constraint.RHS = newRightHandSide



    def b_n_b_subproblem(self):
        import time
        model = self.subproblem
        upper_bound = np.inf
        tree = self.tree
        # reset tree obj and incumbant
        for leaf in tree.keys():
            # reset incumbant?
            tree[leaf][2] = None
            # reset obj val
            tree[leaf][3] = None
            # reset leaf node status
            tree[leaf][5] = None
        generated_cuts = {}  # tuple list of format (beta_multiplier, alpha)
        self.node_counter = len(tree.keys())-1  # root node
        print('Number of node in Tree ' + str(self.node_counter))

        model.update()
        model.Params.OutputFlag = 1

        CandSolutionTree = [*tree.keys()]


        while (len(CandSolutionTree)!= 0):
            # get next node
            branch_node = CandSolutionTree.pop(0)
            # get new limits
            for (var, sense, new_limit, old_limit) in tree[branch_node][0]:
                if sense == "lb":
                    var.lb = new_limit
                if sense == "ub":
                    var.ub = new_limit

            # optimize model
            model.update()
            model.optimize()

            extendTree = False
            if model.Status == GRB.OPTIMAL:
                tree[branch_node][2] = model.objVal
                if model.objVal < upper_bound:


                    # if no child exists
                    if tree[branch_node][4] == None and not tree[branch_node][3] == 'incumbant':
                        extendTree = True

                else:
                    # set node as leaf
                    tree[branch_node][5] = 'leaf'

            elif model.Status == GRB.INFEASIBLE:
                # need to pop all subsequent nodes
                print('infeasible node ' + str(branch_node))
                self.remove_child_nodes(branch_node, tree, CandSolutionTree)

            # add bender cuts
            if model.Status != GRB.INFEASIBLE:
                # determine beta mulitplier
                dual_beta = model._constraint.Pi

                # dual alpha
                # get names of integer variables
                names_to_retrieve = (f"integer[{i}]" for i in model._set_yd)

                # get all integer variables
                branching_vars = [model.getVarByName(name) for name in names_to_retrieve]

                dual_alpha = 0
                for var in branching_vars:
                    if var.X == var.lb:
                        dual_alpha += var.lb*var.RC
                    elif var.X == var.ub:
                        dual_alpha += var.ub*var.RC

                generated_cuts[branch_node] = [(dual_beta, dual_alpha)]

                # append all cuts from prdecessors
                if branch_node != 0:
                    for cuts in generated_cuts[tree[branch_node][1]]:
                        generated_cuts[branch_node].append(cuts)
            else:
                print('node ' + str(branch_node) + ' is infeasible')


            model.update()
            # extend tree
            if extendTree:
                self.populateCandidates(model, CandSolutionTree, tree, branch_node)

            # reset limits
            for (var, sense, new_limit, old_limit) in tree[branch_node][0]:
                if sense == "lb":
                    var.LB = 0
                if sense == "ub":
                    var.UB = np.inf
            model.update()

            # update upper bound, if solution is integer
            if tree[branch_node][3] == 'incumbant':
                if upper_bound > tree[branch_node][2]:
                    upper_bound = tree[branch_node][2]


        # remove all cuts that are not leave
        for id, node in tree.items():
            if not node[5] == 'leaf':
                del generated_cuts[id]

        return generated_cuts

    def remove_child_nodes(self, remove_node, tree, CandSolutionTree):
        print('please add recourse pop function')
        if tree[remove_node][4] != None:
            remove_node(tree[remove_node][4][0], tree, CandSolutionTree)
            remove_node(tree[remove_node][4][1], tree, CandSolutionTree)





            # set identified limits


    def populateCandidates(self, model, currentCandidates: [], tree:{}, currentNode:0):
        # get names of integer variables
        names_to_retrieve = (f"integer[{i}]" for i in model._set_yd)

        # get all integer variables
        branching_vars = [model.getVarByName(name) for name in names_to_retrieve]

        # iterate through integer variables to check if they are fractional
        foundBranch = False
        for var in branching_vars:
            # check if fractional, be assessing distance to next int

            if (np.absolute(var.X - round(var.X,0)) >=0.001):
                # found fractional value
                # determine child nodes, by limiting ub and lb of variables respectively

                # create child node with new ub on variables
                new_ub = np.floor(var.X)
                old_ub = var.ub
                Node_ub = [(var, "ub", new_ub, old_ub)]

                # create child node with new ub on variables
                new_lb = np.ceil(var.X)
                old_lb = var.lb
                Node_lb = [(var, "lb", new_lb, old_lb)]



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
                currentCandidates.append(self.node_counter)


                self.node_counter += 1
                tree[self.node_counter] = [Node_lb, currentNode,None, None, None, None]
                currentCandidates.append(self.node_counter)

                # safe child nodes in tree
                tree[currentNode][4] = (self.node_counter -1, self.node_counter)
                foundBranch = True

            if foundBranch:
                break

        # if no branch is found, all variables are integer, thus new incumbant solution
        if not foundBranch:
            tree[currentNode][3] = 'incumbant'
            # if solution is integer, set node to leafe node
            tree[currentNode][5] = 'leaf'
            print('found incubant solution ' + str(currentNode))
        else:
            print(currentCandidates)

        return currentCandidates

def calculate_approximation(scenarios_prob, cuts,first_stage_solution):
    approximation = 0
    for scenario in scenarios_prob.keys():
        best_bound_scenario_min = np.inf
        # enter Hw

        for leaf in cuts[scenario].values():
            best_bound_cut_max = -100000000.0
            for cut in leaf:
                approx_RHS = 0

                for key, var_type in first_stage_solution.items():
                    for key_var, var in var_type.items():
                        if (key_var[1],key_var[0]) in cut[0][key].keys():
                            if var != 0:
                                approx_RHS += var * cut[0][key][key_var[1],key_var[0]]
                cut_value = approx_RHS + cut[1]
                if best_bound_cut_max < (approx_RHS + cut[1]):
                    best_bound_cut_max = approx_RHS + cut[1]



            if best_bound_scenario_min > best_bound_cut_max:
                best_bound_scenario_min = best_bound_cut_max
        if best_bound_scenario_min == -np.inf:
            print(scenario)
            print(cuts[scenario].values())
            print(f'{best_bound_scenario_min} final')
            raise ValueError

        approximation += scenarios_prob[scenario] * best_bound_scenario_min


    return approximation


def generalizedBendersDecompostion(MasterProblem, scenarioProblem, firstStageGreedySolution):

    import time
    start_time = time.time()

    # Initialize
    theta = -np.inf

    # first iteration of problem
    k = 1

    # empty set of cuts
    cuts= {scenario: None for scenario in MasterProblem.scenarios}
    usedShifts = {scenario: None for scenario in MasterProblem.scenarios}
    notConverged = True

    # get solution values
    first_stage_solution = firstStageGreedySolution

    theta = MasterProblem.get_theta_value()
    stopAtK = 1
    print('#'*10)
    print('Start solution tuning \n')
    startTuning = time.time()
    while notConverged:


        # Update the lower approximation function
        # solve scenarioProblems with updated (h_w - Tx), if first iteration generate subproblem
        for scenario in MasterProblem.scenarios:
            # Right hand side only needs to be adapted after frist iteration, as models are initialized with first solution
            if k > 1:
                RHS = MasterProblem.get_firstStage_variables()
            else:
                RHS = firstStageGreedySolution


            # change right hand side correspondingly
            scenarioProblem.change_RHS_from_FristStage(newRightHandSide=RHS, scenario=scenario)

                # update all identified solutions
                # scenarioProblem[scenario].adapt_prevSolutions(newRHS=RHS)

            # generate Cuts for Masterproblem
            solutionLimit = 50 + k
            if scenario !='Pessimistic':
                solutionLimit += 1
            cuts[scenario], usedShifts[scenario] = scenarioProblem.solve_ScenarioProblem_BranchNBound(scenario, solutionLimit, k, RHS)

        cuts=usedShifts


        if k < stopAtK:
            MasterProblem.extended_master_problem(cuts, iteration=k)
            MasterProblem.solve_MP()
            first_stage_solution = MasterProblem.get_firstStage_variables()


        # check if theta equal solution
        approximation = calculate_approximation(MasterProblem.scenarios_prob, cuts, first_stage_solution)
        print('New First Stage Iteration')
        currentTime = np.round(time.time() - start_time, 2)
        print(currentTime)
        print('Approximation')
        print(approximation)
        print('Theta')
        print(MasterProblem.MasterProblem.ObjVal)
        theta = MasterProblem.MasterProblem.ObjVal

        if k >= stopAtK:
            notConverged = False
            print('DDDOOOOONNNNEE')
            print(f'It took {time.time() - startTuning} seconds to tune solution')
            print(first_stage_solution)
            return first_stage_solution

        k+=1

def initialSolutionTuning(scenarioProblem, firstStageGreedySolution, parameters):

    import time
    start_time = time.time()

    # Initialize
    theta = -np.inf

    # first iteration of problem
    k = 1

    # empty set of cuts
    notConverged = True

    # get solution values
    first_stage_solution = firstStageGreedySolution

    stopAtK = 1
    print('#'*10)
    print('Start solution tuning \n')
    startTuning = time.time()



    # Update the lower approximation function
    # solve scenarioProblems with updated (h_w - Tx), if first iteration generate subproblem
    for scenario in parameters.scenarios:
        # Right hand side only needs to be adapted after frist iteration, as models are initialized with first solution

        RHS = firstStageGreedySolution

        # change right hand side correspondingly
        scenarioProblem.change_RHS_from_FristStage(newRightHandSide=RHS, scenario=scenario)

        scenarioProblem.solve_ScenarioProblem_BranchNBound(scenario, [], 0, RHS)


    print(f'It took {time.time() - startTuning} seconds to tune solution')
    print(first_stage_solution)
    return first_stage_solution






