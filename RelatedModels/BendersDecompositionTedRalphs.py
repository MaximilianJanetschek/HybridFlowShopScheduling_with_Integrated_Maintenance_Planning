import gurobipy as gp
from gurobipy import *
import numpy as np


class MasterProblem():

    def __init__(self):
        self.generate_master_problem()
    def generate_master_problem(self):
        master_problem = gp.Model()
        master_problem._set_scenarios = [1,2]
        master_problem._bigM = 1000
        master_problem.Params.OutputFlag = 0
        master_problem._constraint_sum_u = None
        master_problem._dv_q = gp.tupledict()
        master_problem._dv_u = gp.tupledict()
        master_problem._constraitn_q_upper = gp.tupledict()
        master_problem._constraitn_q_lower = gp.tupledict()
        master_problem._constraitn_q = gp.tupledict()

        master_problem._set_leaf_scenarios = {scenario: [] for scenario in master_problem._set_scenarios}
        master_problem._x_1 = master_problem.addVar(vtype=GRB.BINARY)
        master_problem._x_2 = master_problem.addVar(vtype=GRB.BINARY)

        master_problem._valueFunction_lower = master_problem.addVars(master_problem._set_scenarios, vtype=GRB.CONTINUOUS)


        master_problem.addConstr(master_problem._x_1 + master_problem._x_2 <= 1)

        master_problem.setObjective(-3*master_problem._x_1 -3.8*master_problem._x_2 + 0.5 *master_problem._valueFunction_lower[1]+ 0.5 *master_problem._valueFunction_lower[2], GRB.MINIMIZE)

        self.MasterProblem = master_problem

    def get_first_stage_variables(self):
        variables = gp.tupledict()
        variables[1,1] = self.MasterProblem._x_1
        variables[2,2] = self.MasterProblem._x_2
        self.first_stage_dv = variables
        return variables

    def define_first_stage_coefficients(self):
        coeff={(1,1): -2, (2,2): 0.5}
        return coeff

    def extended_master_problem(self, cuts_per_scenario, iteration):   # scenario format {scenario 1: cuts, scenario 2: cuts}
        model = self.MasterProblem

        # remove all unnecessary leaves
        for (leaf, scenario) in model._dv_q.keys():
            if leaf not in cuts_per_scenario[scenario].keys():
                print('Remvoe leaf ' + str(leaf))
                model.remove(model._dv_q[(leaf, scenario)])
                model.remove(model._dv_u[(leaf, scenario)])
                model.remove(model._constraitn_q_upper[(leaf, scenario)])
                model.remove(model._constraitn_q_lower[(leaf, scenario)])
                for c in model._constraitn_q.select('*', leaf, scenario):
                    model.remove(c)
                model._set_leaf_scenarios[scenario].remove(leaf)

        model.update()

        # add new variables
        # add new leave nodes
        for scenario in cuts_per_scenario.keys():
            for leaf in cuts_per_scenario[scenario].keys():
                if leaf not in model._set_leaf_scenarios[scenario]:
                    model._set_leaf_scenarios[scenario].append(leaf)
                    # add new decision variables
                    model._dv_q[(leaf, scenario)] = model.addVar(vtype = GRB.CONTINUOUS)

                    print('adding variables ' + str(leaf) + ' for scenario ' + str(scenario))
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
                    model._constraitn_q_upper[(leaf, scenario)] = model.addConstr(model._valueFunction_lower[scenario] <= model._dv_q[(leaf, scenario)])
                    model._constraitn_q_lower[(leaf, scenario)] = model.addConstr(model._valueFunction_lower[scenario] >= model._dv_q[(leaf, scenario)] - model._bigM *( 1- model._dv_u[(leaf, scenario)] ))

                # add all identified cuts
                for c_id, cut in enumerate(cuts_per_scenario[scenario][leaf]):
                    print('new cut')
                    print(cut[0])
                    print(cut[1])
                    model._constraitn_q[(c_id, leaf, scenario)] = model.addConstr(model._dv_q[(leaf, scenario)] >= (model._parameter_h_w[scenario] - 2*model._x_1 - 0.5*model._x_2)*cut[0]+cut[1])

        if iteration == 1:
            model._constraint_sum_u = model.addConstrs((model._dv_u.sum('*', scenario) == 1
                for scenario in model._set_scenarios), name='test')


        model.update()





    def solve_MP(self):
        self.MasterProblem.update()
        self.MasterProblem.optimize()

    def get_x_values(self):
        model = self.MasterProblem
        if model.Status == GRB.OPTIMAL:
            for scenario in model._set_leaf_scenarios.keys():
                for leaf in model._set_leaf_scenarios[scenario]:
                    print('leaf ' + str(leaf)  + ' in scenario ' + str(scenario))
                    print(model._dv_q[(leaf,scenario)].X)
            print('X 1 and X 2')
            print(model._x_1.X, model._x_2.X)
            theta = (model._valueFunction_lower[1].X+model._valueFunction_lower[2].X)/2
            print('Theta ')
            print( str(theta))
            print('Model objective')
            print(model.objVal)
            first_stage_solution = gp.tupledict()
            for key, x in self.first_stage_dv.items():
                first_stage_solution[key] = x.X

            return first_stage_solution, theta

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
        model.Params.OutputFlag = 0

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

def calculate_approximation(h_w, cuts,first_stage_solution, first_stage_coeff,probability_by_scenario):
    approximation = 0
    for scenario in h_w.keys():
        best_bound_scenario_min = np.inf
        approx_RHS = h_w[scenario]
        for key, variable in first_stage_solution.items():
            approx_RHS += first_stage_coeff[key] * variable

        for leaf in cuts[scenario].values():
            best_bound_cut_max = -np.inf
            for cut in leaf:
                cut_value = cut[0] * approx_RHS + cut[1]
                if best_bound_cut_max < cut_value:
                    best_bound_cut_max = cut_value
            if best_bound_scenario_min > best_bound_cut_max:
                best_bound_scenario_min = best_bound_cut_max
        approximation += probability_by_scenario[scenario] * best_bound_scenario_min

    return approximation



# Initialize
h_w = {1:6,2:12}
probability_by_scenario = {1:0.5, 2:0.5}
MP = MasterProblem()
MP.MasterProblem._parameter_h_w = h_w
theta = -np.inf
first_stage_variables = MP.get_first_stage_variables()
first_stage_coeff = MP.define_first_stage_coefficients()

# first iteration of problem
k = 1

subprolem = {scenario: None for scenario in MP.MasterProblem._set_scenarios}
subproblem_1 = None
subproblem_2 = None
cuts= {scenario: None for scenario in MP.MasterProblem._set_scenarios}

notConverged = True

MP.solve_MP()
# get solution values
first_stage_solution, theta = MP.get_x_values()

while notConverged:


    # Update the lower approximation function
    # solve subproblem 1 and 2 with updated h_w - Tx, if first iteration generate subproblem
    for scenario in MP.MasterProblem._set_scenarios:
        RHS =  h_w[scenario]
        for key, variable in first_stage_solution.items():
            RHS += first_stage_coeff[key] * variable
        if k == 1:
        # generate subproblem

            subprolem[scenario] = ScenarioProblem(right_hand_side=RHS)
        else:
            # change right and side
            subprolem[scenario].change_RHS(newRightHandSide=RHS)

        # generate Cuts for Masterproblem
        cuts[scenario] = subprolem[scenario].b_n_b_subproblem()

    MP.extended_master_problem(cuts, iteration=k)
    MP.solve_MP()

    first_stage_solution, theta = MP.get_x_values()

    # check if theta equal solution
    approximation = calculate_approximation(h_w, cuts, first_stage_solution, first_stage_coeff, probability_by_scenario)
    print('Approximation')
    print(approximation)


    if approximation == theta:
        notConverged = False
        print('DDDOOOOONNNNEE')

    k+=1










