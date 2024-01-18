from gurobipy import *
import gurobipy as gp
import numpy as np
import time
import math


def twoStage_cb(model, where):
    """
    Gurobi call back to add identified gap convergence and problem user cuts for full two stage model.
    """
    if where == GRB.Callback.MIPSOL:
        if model.cbGet(GRB.Callback.MIPSOL_SOLCNT) == 0:
            # creates new model attribute '_startobjval'
            model._startobjval = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            model._dataUB.append([0, model._startobjval])


    if where == gp.GRB.Callback.MIP:
        # get current bounds
        cur_obj = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
        cur_bd = model.cbGet(gp.GRB.Callback.MIP_OBJBND)

        # Did objective value or best bound change? If so add the corresponding data points to the list.
        if model._obj > cur_obj:
            model._obj = cur_obj
            model._dataUB.append([time.time() - model._start, cur_obj])
            gap = (cur_obj - cur_bd) / cur_bd
            model._gap.append([time.time() - model._start, gap])

        if model._bd < cur_bd:
            model._bd = cur_bd
            model._dataLB.append([time.time() - model._start,  cur_bd])
            gap = (cur_obj - cur_bd) / cur_bd
            model._gap.append([time.time() - model._start, gap])


    if where == GRB.Callback.MIPNODE:
        status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
        if status == GRB.OPTIMAL:
            if model._counter <= 0:
                # check all scenario solutions for fractional values
                for scenario in model._TwoStage_rescheduling_repair.keys():
                    # collect total repair time per machine
                    repair_batch_machine = model.cbGetNodeRel(model._scenario_repair_batch_machine[scenario])
                    test = {machine: 0 for machine in model._machines}
                    for (repair, batch, machine), sol in repair_batch_machine.items():
                        test[machine] += sol
                    model._counter = 5
                    for machine in test.keys():
                        # add cut if total sum of repairs per machine is not integer
                        if (np.absolute(test[machine] - round(test[machine], 0)) >= 0.01):
                            model.cbCut(model._scenario_repair_batch_machine[scenario].sum('*', "*", machine) >= np.ceil(test[machine]))
                            model._counter = 0
            else:
                model._counter -= 1


def independantScenario_cb(model, where):

    if where == GRB.Callback.MIPSOL:
        if model.cbGet(GRB.Callback.MIPSOL_SOLCNT) == 0:
            # creates new model attribute '_startobjval'
            model._startobjval = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            model._dataUB.append([0, model._startobjval])

    if where == gp.GRB.Callback.MIP:
        cur_obj = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
        cur_bd = model.cbGet(gp.GRB.Callback.MIP_OBJBND)

        # Did objective value or best bound change?
        if model._obj > cur_obj:
            model._obj = cur_obj
            model._dataUB.append([time.time() - model._start, cur_obj])
        if model._bd < cur_bd:
            model._bd = cur_bd
            model._dataLB.append([time.time() - model._start, cur_bd])
    if where == GRB.Callback.MIPNODE:
        status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
        if status == GRB.OPTIMAL:
            if model._counter <= 0:
                # collect total repair time per machine
                repair_batch_machine = model.cbGetNodeRel(model._repair_batch_machine)
                test = {machine: 0 for machine in model._machines}
                for (repair, batch, machine), sol in repair_batch_machine.items():
                    test[machine] += sol
                model._counter = 5
                for machine in test.keys():
                    # add cut if total sum of repairs per machine is not integer
                    if (np.absolute(test[machine] - round(test[machine], 0)) >= 0.01):
                        model.cbCut(model._repair_batch_machine.sum('*', "*", machine) >= np.ceil(test[machine]))
                        model._counter = 0
            else:
                model._counter -= 1

def scenario_cb(model, where):
    if where == GRB.Callback.MIPNODE:
        status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
        if status == GRB.OPTIMAL:
            if model._counter <= 0:
                # collect total repair time per machine
                repair_batch_machine = model.cbGetNodeRel(model._repair_batch_machine)
                test = {machine: 0 for machine in model._machines}
                for (repair, batch, machine), sol in repair_batch_machine.items():
                    test[machine] += sol
                model._counter = 5
                for machine in test.keys():
                    # add cut if total sum of repairs per machine is not integer
                    if (np.absolute(test[machine] - round(test[machine], 0)) >= 0.01):
                        model.cbCut(model._repair_batch_machine.sum('*', "*", machine) >= np.ceil(test[machine]))
                        model._counter = 0
            else:
                model._counter -= 1



