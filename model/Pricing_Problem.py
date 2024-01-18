import itertools

from utilities import *
import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import time
import matplotlib.pyplot as plt


class multi_visit_TOP():
    def __init__(self, parameters):
        print("Pricing is started")
        # initialize Graph
        ProductionNetwork = nx.DiGraph()

        # safe variable for easier acess
        positions = parameters.positions_per_machine
        machine_per_stage = parameters.machines_per_stage
        self.machine_per_stage = parameters.machines_per_stage
        self.stage_per_machine = parameters.stage_per_machine
        stages = parameters.set_stages

        # add machines, positions as nodes to production network
        count_stages = len(stages)
        nodes = []
        # add pos attribute to nodes to be able to retrieve drawing and have machines per stagecentered. Not required for optimisation
        for i in range(0,count_stages):
            temp = -1 * (len(machine_per_stage[stages[i]])/2) +0.5
            for machine in machine_per_stage[stages[i]]:
                for position in range(0,positions[machine]):
                    nodes.append(((position,machine),{'pos':(i,temp)}))
                temp = temp + 1
        ProductionNetwork.add_nodes_from(nodes)
        ProductionNetwork.add_node((0,"start"),pos=(-1,0))
        # increase x position with stage to align node
        ProductionNetwork.add_node((0,"end"),pos=(count_stages,0))
        # add all arcs
        links = []
        count_stages = len(stages)
        for i in range(0,count_stages):
            for machine in machine_per_stage[stages[i]]:
                for position in range(0, positions[machine]):
                    # add start to first stage
                    if i == 0:
                        links.append(((0,"start"),(position,machine),position))
                    else:
                        # if between stages, add every possible cross dependency between machine positions
                        for machine_prev in machine_per_stage[stages[i-1]]:
                            for position_prev in range(0, positions[machine_prev]):
                                links.append(((position_prev,machine_prev), (position,machine), position))
                            # if last stage, add edges to end
                            if i == count_stages-1:
                                links.append(((position,machine), (0,"end"), position))

        # each arc has an assigned weight, which is equal to its position. Thus, in shortest path early position will be used first
        ProductionNetwork.add_weighted_edges_from(links)
        self.G = ProductionNetwork


    def generate_pricing_problem(self, parameters):
            """
            Function that generates the corresponding gurobi model.
            :param networkGraph: networkx graph
            :param jobs: list of jobs to be scheduled, treated as routes
            :return: generated gurobi model
            """
            print("Generate Pricing Problem")
            jobs = parameters.jobs
            machines_per_stage = parameters.machines_per_stage
            set_stages = parameters.set_stages

            # initialize model
            TOP_model = gp.Model("MultiVisit-TeamOrienteeringProblem - Pricing Problem")

            # transform production network to TOP formulation, transfrom tuples
            TOP_model._set_nodes = list(self.G.nodes)
            TOP_model._set_edges = list()
            for i,j in self.G.edges:
                TOP_model._set_edges.append((i[0],i[1],j[0],j[1]))

            # safe sets
            TOP_model._set_jobs = jobs
            TOP_model._set_nodes.remove((0,'start'))
            TOP_model._set_nodes.remove((0,'end'))

            TOP_model._set_edges = gp.tuplelist(TOP_model._set_edges)

            # add variables
            print("Generate Vars of Pricing")

            TOP_model._edge_selection = TOP_model.addVars(TOP_model._set_edges, jobs, vtype=GRB.BINARY, name='edge_selection')
            TOP_model._edge_usage = TOP_model.addVars(TOP_model._set_edges, vtype=GRB.BINARY, name='edge_selection')
            TOP_model._node_processing = TOP_model.addVars(TOP_model._set_nodes, vtype=GRB.CONTINUOUS, name='processing time of batch')

            TOP_model.update()

            print("Generate Constraint of Pricing")
            constraint_starttime = time.time()
            # leave source for every job
            TOP_model.addConstrs((TOP_model._edge_selection.sum(0,'start','*','*',job) == 1 for job in jobs))
            past_time = time.time() - constraint_starttime
            print(f'### Leave source, {past_time:.2f}. ###')

            # enter sink with every job
            TOP_model.addConstrs((TOP_model._edge_selection.sum('*','*',0, 'end',job) == 1 for job in jobs))
            past_time = time.time() - constraint_starttime
            print(f'### Reach destination, {past_time:.2f}. ###')

            #leave every node as often as entered
            TOP_model.addConstrs((TOP_model._edge_selection.sum('*','*',batch,machine , job) == TOP_model._edge_selection.sum(batch,machine , '*', '*' , job) for batch,machine in TOP_model._set_nodes for job in jobs))
            past_time = time.time() - constraint_starttime
            print(f'### Leave every node as often as visited, {past_time:.2f}. ###')

            #leave every node as often as entered
            TOP_model.addConstrs((gp.quicksum(TOP_model._edge_selection.sum('*','*','*',machine , job) for machine in machines_per_stage[stage]) >= 1 for stage in set_stages for job in jobs))
            past_time = time.time() - constraint_starttime
            print(f'### Limiting the edges per machine a job can visit, {past_time:.2f}. ###')

            # maximum visit constraint, gurobipy tuple dict cannot work with double dict
            TOP_model.addConstrs((TOP_model._edge_selection.sum('*','*', batch, machine, '*') <= self.batch_limit[machine] for batch, machine in TOP_model._set_nodes))
            past_time = time.time() - constraint_starttime
            print(f'### Limit on node visits, {past_time:.2f}. ###')

            # determine node relationship
            TOP_model.addConstrs(TOP_model._edge_selection.sum(batch_prev,machine_prev,batch,machine,'*') <= self.batch_limit[machine]*TOP_model._edge_usage[batch_prev, machine_prev, batch,machine] for batch, machine in TOP_model._set_nodes for batch_prev, machine_prev, _ ,_ in TOP_model._set_edges.select('*','*',batch,machine) )
            past_time = time.time() - constraint_starttime
            print(f'### Generate node relationship, {past_time:.2f}. ###')
            #
            # determine node processing time due to visits
            # if batch limit for the machine is equal to 1, then only one arc is entering the node and we can sum overall arcs
            TOP_model._constraint_processing_time = gp.tupledict()
            for batch, machine in TOP_model._set_nodes:
                # if machine have at most one job assigned to position only one constrained is required as it can be expressed as an sum
                if self.batch_limit[machine] == 1:
                    TOP_model.addConstr((gp.quicksum(TOP_model._edge_selection[batch_prev,machine_prev,batch,machine,job] * self.job_machine_processing[job, machine] for job in jobs for batch_prev, machine_prev, _ ,_ in TOP_model._set_edges.select('*','*',batch,machine)) <= TOP_model._node_processing[batch, machine]))
                else:
                    # for a batch process all incoming arc need to be considered indepentently
                    created_constraints = TOP_model.addConstrs((gp.quicksum(TOP_model._edge_selection[batch_prev,machine_prev,batch,machine,job] * self.job_machine_processing[job, machine] for batch_prev, machine_prev, _ ,_ in TOP_model._set_edges.select('*','*',batch,machine)) <= TOP_model._node_processing[batch, machine] for job in jobs ))
                    TOP_model._constraint_processing_time.update(created_constraints)
            past_time = time.time() - constraint_starttime
            print(f'### Generate processing time, {past_time:.2f}. ###')

            # set processsing time constraints as lazy as only a small subset will be binding
            for i, processing_constraint in TOP_model._constraint_processing_time.items():
                processing_constraint.Lazy = 3

            # speed up constraint by enforcing an lower limit on the processing per node, even if fraction of incoming arcs are spread, this gives a tighter boudn
            TOP_model.addConstrs((gp.quicksum(TOP_model._edge_selection[batch_prev,machine_prev,batch,machine,job] * self.job_machine_processing[job, machine] / self.batch_limit[machine] for job in jobs for batch_prev, machine_prev, _ ,_ in TOP_model._set_edges.select('*','*',batch,machine)) <= TOP_model._node_processing[batch, machine] for batch, machine in TOP_model._set_nodes))
            past_time = time.time() - constraint_starttime
            print(f'### Generate processing time - speed up by sum limit, {past_time:.2f}. ###')

            self.TOP_model = TOP_model
            print('##'* 40)
            print(' '* 10 + 'Finished Model')
            print('##' * 40)


    def multi_dijkstra(self, GraphNetwork,withEdgeUpdate: False):
        """
        Function that applies a shortest path algorithm for every job. The full set of shortest path determines a production assignment an can be used as an incumbent solution.
        :param GraphNetwork: Production Graph
        :param withEdgeUpdate: Edges
        :return:
        """
        shortestPath={}

        # track the usage of nodes
        nodes = {node: 0 for node in GraphNetwork.nodes}
        nodes_time = {node: 0 for node in GraphNetwork.nodes}
        edges = {edge: 0 for edge in GraphNetwork.edges}
        # iterate through each job
        for job in self.set_jobs:
            # determine shortest path
            shortestPath[job] = nx.dijkstra_path(GraphNetwork,(0,"start") , (0,"end"), weight='weight')

            # increase all edges
            for node_counter in range(2,len(shortestPath[job])-1):
                prev_position, prev_machine = shortestPath[job][node_counter-1]
                position,machine = shortestPath[job][node_counter]
                edges[(prev_position,prev_machine),(position,machine)] = 1

                GraphNetwork[prev_position,prev_machine][position,machine]['weight'] = 0
                # remove process time from all node entering edges
                if withEdgeUpdate:
                    for (i, j, d) in GraphNetwork.in_edges((position, machine), data=True):
                        GraphNetwork[i][j]['weight'] = GraphNetwork[i][j]['weight'] - GraphNetwork[i][j]['node_weight']
                        GraphNetwork[i][j]['node_weight'] = 0

                GraphNetwork[prev_position, prev_machine][position, machine]['weight'] = 0



            # increase all nodes
            for node_counter in range(1,len(shortestPath[job])-1):
                position,machine = shortestPath[job][node_counter]
                usage_count = nodes[(position,machine)] + 1
                if usage_count == self.batch_limit[machine]:
                    GraphNetwork.remove_node((position,machine))
                nodes[(position,machine)] = usage_count
                if nodes_time[(position,machine)] <= self.job_machine_processing[job,machine]:
                    nodes_time[(position,machine)] = self.job_machine_processing[job,machine]





        return shortestPath, nodes_time, edges


    def generate_initial_soltuion(self, parameters):
        """

        :param parameters:
        :return:
        """

        # safe for easier access
        set_jobs = parameters.jobs
        batch_limit = parameters.jobs_per_batch_by_machine
        job_machine_processing = parameters.process_times_job_machine
        print('Apply dijkstra per job to to get initial solution')

        # copy graph
        GraphNetwork = self.G.copy()

        # safe batch_limit
        self.batch_limit = batch_limit
        self.job_machine_processing = job_machine_processing
        self.set_jobs = set_jobs
        withEdgeUpdate = False
        shortestPath, nodes_time, edges = self.multi_dijkstra(GraphNetwork,withEdgeUpdate)

        self.currentBest = BestFoundSolution(shortestPath, nodes_time)

        # reformat edges
        edges_dict = gp.tupledict()
        for (u,v) in edges.keys():
            edges_dict[u[0], u[1], v[0], v[1]] = edges[u,v]

        return edges_dict, nodes_time

    def generate_new_greedy_solution(self, dual_cross_dependency, dual_processing_time, dual_maintenance_time, dual_repair_time):
        # update all edges
        ProcessingTimeAverage_by_Stage = {"Mixing":32, "Coating&Drying":20, "Calandering":20, "Slitting":18, 'AddDrying':64}

        # Copy graph
        GraphNetwork = self.G.copy()

        # reset all edge weights
        for u, v, d in GraphNetwork.edges(data=True):
            GraphNetwork[u][v]['weight'] = 0
            GraphNetwork[u][v]['node_weight'] = 0


        # update all edges
        for u,v,d in GraphNetwork.edges(data=True):
            processing_time_weight = 0
            if v[1]!= ('end'):
                # update dual_crossdependency
                if u[1] != 'start':
                    GraphNetwork[u][v]['weight'] = GraphNetwork[u][v]['weight'] - dual_cross_dependency[u[0],u[1],v[0],v[1]]
                processing_time_weight = (ProcessingTimeAverage_by_Stage[self.stage_per_machine[v[1]]]) * (dual_processing_time[v] - dual_maintenance_time[v] - dual_repair_time[v])
                GraphNetwork[u][v]['weight'] = GraphNetwork[u][v]['weight'] + processing_time_weight
                GraphNetwork[u][v]['node_weight'] = processing_time_weight
        # reset all edge weights
        removable_arcs = 0

        # get most negative
        most_negative_arc = 0
        for u, v, d in GraphNetwork.edges(data=True):
            if GraphNetwork[u][v]['weight'] < most_negative_arc:
                most_negative_arc = GraphNetwork[u][v]['weight']

        # lift all arcs to at least 0
        for u, v, d in GraphNetwork.edges(data=True):
            GraphNetwork[u][v]['weight'] += most_negative_arc +0.000001


        for u, v, d in GraphNetwork.edges(data=True):
            if GraphNetwork[u][v]['weight'] < 0:
                print('negative arc weights')
                print(GraphNetwork[u][v]['weight'])
            if GraphNetwork[u][v]['weight'] > most_negative_arc:
                removable_arcs += 1
        print('Could remove ' + str(removable_arcs))
        print(len(GraphNetwork.edges))

        withEdgeUpdate = True
        shortestPath, nodes_time, edges = self.multi_dijkstra(GraphNetwork, withEdgeUpdate)


        self.currentBest = BestFoundSolution(shortestPath, nodes_time)

        return edges, nodes_time







    def reOptimize_pricing(self, dual_cross_dependency, dual_processing_time, dual_maintenance_time, dual_repair_time, BigM):
        print("rerun pricing")
        new_pattern = False
        optimizedPattern = None

        # add duals from cross dependency
        obj_cross_dependency = gp.quicksum(self.TOP_model._edge_usage[batch_prev, machine_prev, batch, machine] * dual_cross_dependency[batch_prev, machine_prev, batch, machine] for (batch_prev, machine_prev, batch, machine) in dual_cross_dependency.keys())
        # add duals from processing time
        obj_processing_time = gp.quicksum(self.TOP_model._node_processing[node] * dual_processing_time[node] for node in dual_processing_time.keys())
        # add duals from maintenance time
        obj_maintenance_time = gp.quicksum(self.TOP_model._node_processing[node] * dual_maintenance_time[node] for node in dual_maintenance_time.keys())
        # add duals from repair time
        obj_repair_time = gp.quicksum(self.TOP_model._node_processing[node] * dual_repair_time[node] for node in dual_repair_time.keys())

        # set objective of model
        self.TOP_model.setObjective( obj_processing_time - obj_maintenance_time - obj_repair_time - obj_cross_dependency, GRB.MINIMIZE)

        # generate new initial solution with dijkstra approxiamation
        multiDijkstraStart = time.time()
        self.generate_new_greedy_solution(dual_cross_dependency, dual_processing_time, dual_maintenance_time, dual_repair_time)
        print(f'It took {time.time()-multiDijkstraStart} to perform Dijsksra approximation')
        # add solution as MIP start
        self.currentBest.warm_start_pricing(self.TOP_model)
        self.TOP_model.update()


        # only solution with negative objective can improve (production assignment as a coefficient in RMP objective, thus duals need to negative)
        self.TOP_model.Params.BestBdStop = 0.0
        self.TOP_model.update()

        self.TOP_model.optimize()

        # check status of model
        model_status = self.TOP_model.Status
        if model_status == GRB.OPTIMAL:
            new_pattern = self.TOP_model.ObjVal < -0.0000001
            if new_pattern:
                print("New Pattern Found")
                optimizedPattern = TOP_solution(self.TOP_model._edge_usage, self.TOP_model._node_processing, BigM)
        elif model_status == GRB.UNBOUNDED:
            print("No new pattern, because model is unbounded.")
        elif model_status == GRB.USER_OBJ_LIMIT:
            print("No new pattern, because model cannot achieve value better than 0.")
        else:
            raise ValueError

        return new_pattern, optimizedPattern

class BestFoundSolution():
    def __init__(self, shortestPaths, node_weight):
        self.tours = shortestPaths
        self.node_weight = node_weight
        self.obj_value = 0


    def warm_start_pricing(self, model):
        # set all edges to zero
        for i_batch, i_machine, j_batch, j_machine in model._set_edges:
            for job in model._set_jobs:
                model._edge_selection[i_batch, i_machine, j_batch, j_machine,job].Start = 0
            # set all edge usage to zero
            model._edge_usage[i_batch, i_machine, j_batch, j_machine].Start = 0

        # set all nodes to zero
        for i in model._set_nodes:
            model._node_processing[i].Start = self.node_weight[i]

        for tour_id, vehicle_path in self.tours.items():
            prev_node = None
            for idnode, node in enumerate(vehicle_path):
                if idnode >= 1:
                    i_batch = vehicle_path[idnode-1][0]
                    i_machine = vehicle_path[idnode-1][1]
                    j_batch = vehicle_path[idnode][0]
                    j_machine = vehicle_path[idnode][1]
                    model._edge_selection[i_batch, i_machine, j_batch, j_machine, tour_id].Start = 1
                    model._edge_usage[i_batch, i_machine, j_batch, j_machine].Start = 1

    def calculate_solution_value(self, dual_cross_dependency,dual_processing_time, dual_maintenance_time, dual_repair_time):
        # add duals from cross dependency
        greedy_obj_cross_dependency = sum(self.TOP_model._edge_usage[i[0],i[1],j[0], j[1]].Start * dual_cross_dependency[i,j] for i,j in dual_cross_dependency.keys())
        # add duals from processing time
        greedy_obj_processing_time = sum(self.TOP_model._node_processing[node].Start * dual_processing_time[node] for node in dual_processing_time.keys())
        # add duals from maintenance time
        greedy_obj_maintenance_time = sum(self.TOP_model._node_processing[node].Start * dual_maintenance_time[node] for node in dual_maintenance_time.keys())
        # add duals from repair time
        greedy_obj_repair_time = sum(self.TOP_model._node_processing[node].Start * dual_repair_time[node] for node in dual_repair_time.keys())

        total_obj =  greedy_obj_processing_time - greedy_obj_repair_time - greedy_obj_maintenance_time - greedy_obj_cross_dependency

        return total_obj





class TOP_solution():


    def __init__(self, optimized_cross_dependency, optimized_processing_time,BigM):
        # only non-zero values are saved as zero is assumed for all other arcs to reduce required space.
        import numpy as np
        self.cross_dependency = {}
        for (batch_prev, machine_prev, batch, machine), edge in optimized_cross_dependency.items():
            if edge.X >= 0.9:
                # when edge is used, then selecting pattern l enforces cross dependency constraint, thus 0
                self.cross_dependency[batch_prev, machine_prev,batch, machine] = 0
            else:
                # when edge is not used, then selecting pattern l is not enforcing cross dependency, thus BigM holds
                self.cross_dependency[batch_prev, machine_prev, batch, machine] = BigM

        self.processing_time = {}
        for (batch, machine), node in optimized_processing_time.items():
            if node.X >= 0.9:
                self.processing_time[(batch,machine)] = np.round(node.X,0)
            else:
                self.processing_time[(batch,machine)] = 0


        print("Safe found pattern.")



