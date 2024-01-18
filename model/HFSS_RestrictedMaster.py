import itertools

from utilities import *
import gurobipy as gp
from gurobipy import *
import numpy as np
from gurobipy import GRB
from .Pricing_Problem import *
import os
import copy


class ParametersOfModel():
    '''
    Class that holds all required parameters for the model.
    '''

    def __init__(self, input_data, layout, config, shifts):
        """
        Initialize all required parameters.
        :param input_data: formatted data on machine process time, maintenance limits, energy consumption, ....
        :param layout: sequence and name of production stages
        :param config: configuration file containing variable and constraint names as well as necessary keys
        :param scenario: set of scenarios, e.g. pessimistic, average, optimistic
        :param shifts: {shift_id: (lower_slot, uppper_slot, shift, day) self.set_shifts
        """
        self.set_shifts = shifts
        self.layout = layout
        self.config = config
        self.horizon = config.model_discrete.set.time_slots.horizon_in_days
        self.scenarios = config.data.maintenance_name.scenarios.keys()
        self.input_data = input_data
        process_data = input_data[config.data.process]
        # parition data
        self.process_data = input_data[config.data.process]
        self.maintenance_data = input_data[config.data.maintenance]
        self.rescheduling_data_cost = input_data[config.data.rescheduling]

        # define sets
        self.set_stages = list(process_data.keys())
        self.machines, self.machines_per_stage, self.stage_per_machine = self.number_machines_per_process_steps(
            self.set_stages, process_data, config)
        self.max_jobs = config.model_discrete.parameters.jobs
        self.jobs = range(1, self.max_jobs)
        self.slots, self.max_number_slots = self.number_of_slots(config)

        # can be probably deeleted?
        self.warmUp = range(1, config.model_discrete.set.limit_shutdown)
        self.shutDown = range(1, config.model_discrete.set.limit_shutdown)

        # Reformat energy consumption input for easy use
        self.energyConsumption_per_machine = self.define_energy_consumption(self.input_data, config)
        self.currentMachineAge, self.currentMachineAgeService = self.define_currentMachineAge(self.input_data, config)

        # define parameters
        self.add_parameters_to_model(self.process_data, self.maintenance_data, config)

        #
        self.task_keys = config.data.maintenance_name

        # determine required slots
        self.slotRange_per_batchNmachine = {}
        self.positions_per_machine = define_number_of_positions(self)

        # determie rescheduling cost
        self.schedulingCategory = config.data.scheduling_cost.category_name

    def number_machines_per_process_steps(self, process_steps, input_data, config):
        '''
        Small function to provide set of maschines by full name, to be able to assess the results later more easily.
        :param process_steps:
        :param input_data:
        :param config:
        :return:
        '''
        machine_set = []
        machines_per_stage = {}
        stage_per_machine = {}
        for single_step in process_steps:
            machines_per_stage[single_step] = []
            for machine_name in input_data[single_step][config.data.parameter_name.process_time].keys():
                machine_set.append(machine_name)
                machines_per_stage[single_step].append(machine_name)
                stage_per_machine[machine_name] = single_step

        return machine_set, machines_per_stage, stage_per_machine

    def define_energy_consumption(self, process_data, config):
        # retrieve keys from configuration file
        energyLevels = config.data.parameter_name.energyConsumption.values()
        # includes
        # idle: 'EnergyConsumptionIdle'
        # operating: 'EnergyConsumptionOperating'
        # warmUp: 'EnergyConsumptionWarmingUp'
        # coolDown: 'EnergyConsumptionCoolingDown'

        # group energy consumption into dict format: energy_consumption[Energy Type][Machine Key]: Demand
        energy_consumption = {enLevel: {} for enLevel in energyLevels}
        for stage, parameter_dict in process_data[config.data.process].items():
            for enName in energyLevels:
                for machine, reqEnergy in parameter_dict[enName].items():
                    energy_consumption[enName][machine] = reqEnergy

        return energy_consumption

    def define_currentMachineAge(self, process_data, config):
        """
        Current age of machine regarding repair and service task is defined in input excel. Corresponding values are retrieved and stored in an appropriate dict format.
        """
        currentMachineAge = {machine: 0 for machine in self.machines}
        currentMachineAgeService = {machine: 0 for machine in self.machines}
        for stage, parameter_dict in process_data[config.data.process].items():
            # load current age of machine for repair
            currentAge = parameter_dict['CurrentAgeRepair']
            for addMachine, addAge in currentAge.items():
                currentMachineAge[addMachine] = addAge
            # load current age of machine for service
            currentAge = parameter_dict['CurrentAgeService']
            for addMachine, addAge in currentAge.items():
                currentMachineAgeService[addMachine] = addAge

        return currentMachineAge, currentMachineAgeService

    def number_of_slots(self, config):
        """
        Determine maximum number time buckets aka slots for the defined time horizon.
        """
        horizon = config.model_discrete.set.time_slots
        max_number_slots = int(
            horizon.horizon_in_days * horizon.shifts_per_day * horizon.hours_per_shift * (60 / horizon.blocks))
        # safe full range for easier access during model initialization
        slots = range(0, max_number_slots)
        return slots, max_number_slots

    def add_parameters_to_model(self, operation_data, maintenance_data, config):
        '''
        Function to add all parameters in input data to model class.
        :param config:
        :return:
        '''

        self.process_times_job_machine = {}
        self.jobs_per_batch_by_machine = {}
        self.time_for_state_transition = gp.tupledict()
        for stage in self.machines_per_stage.keys():
            for machine, process_time in operation_data[stage][config.data.parameter_name.process_time].items():
                # add process time for regular jobs
                for job in self.jobs:
                    self.process_times_job_machine[(job, machine)] = int(process_time)
            # add cool-down and warm-up times
            for machine, duration in operation_data[stage][config.data.parameter_name.duration.warmup].items():
                self.time_for_state_transition[(config.model_discrete.decision_variables.warm_up, machine)] = int(
                    duration)
            for machine, duration in operation_data[stage][config.data.parameter_name.duration.cooldown].items():
                self.time_for_state_transition[(config.model_discrete.decision_variables.cool_down, machine)] = int(
                    duration)
            for machine, jobs_per_batch in operation_data[stage][config.data.parameter_name.jobs_per_batch].items():
                self.jobs_per_batch_by_machine[machine] = int(jobs_per_batch)

        # determine lowest number of required operation time on machine
        self.lowest_processTime_per_machine = {machine: np.inf for machine in self.machines}
        for (job, machine) in self.process_times_job_machine.keys():
            if self.lowest_processTime_per_machine[machine] > self.process_times_job_machine[(job, machine)]:
                self.lowest_processTime_per_machine[machine] = self.process_times_job_machine[(job, machine)]


class HFSS_ColumnGeneration():
    sets = {}

    decision_variables = {}
    parameters = {}

    def __init__(self, parameters: ParametersOfModel, first_stage_decision):
        '''
        Initialise HFSS model.
        :param input_data: formatted data on machine process time, maintenance limits, energy consumption, ....
        :param layout: sequence and name of production stages
        :param config: configuration file containing variable and constraint names as well as necessary keys
        :param scenario: set of scenarios, e.g. pessimistic, average, optimistic
        :param shifts: {shift_id: (lower_slot, uppper_slot, shift, day) self.set_shifts
        :param first_stage_decision: format {scheduled_repair: {(stage, shift): number_of_repair}}
        '''

        self.first_stage_decision = first_stage_decision

        self.gurobi_model = gp.Model('Restricted Master Problem Before Pricing')
        self.defining_HFSS_model(parameters, self.gurobi_model)
        self.parameters = parameters

    def generate_SolutionProcedureForScenario(self, parameters, scenarios, afterPricingDependency,
                                              AfterPricingProcessTime, selectedPattern):
        """
        This function generates for the defined sceanrios all independent scenario problems, e.g. single stage HFSS. Afterwards the construction heuristic following the forward scheduling is used.
        """

        # get current scenario
        self.relaxedScenarioModels = {}
        self.IntegerScenarioModels = {}

        self.removeRedundantVariables(parameters, self.gurobi_model, afterPricingDependency, AfterPricingProcessTime)

        for scenario in scenarios:
            print('\nGenerating model for ' + scenario)
            scenarioKey = ""
            # initialize models as relaxed model first, as with the two stage, the IP model is obtained by copyin the model changing the variables types.
            self.relaxedScenarioModels[scenario] = self.define_RestrictedMasterProblem(self.gurobi_model, scenario,
                                                                                       scenarioKey, parameters,
                                                                                       afterPricingDependency,
                                                                                       AfterPricingProcessTime,
                                                                                       selectedPattern)

            # copy model and attributes to change variables to match IP formulation. Additionally, all attributes in model safed in model._ are copied.
            varKeyScenario = ""
            self.IntegerScenarioModels[scenario] = self.copy_model_as_MIP(self.relaxedScenarioModels[scenario],
                                                                          scenario, varKeyScenario)
            self.IntegerScenarioModels[scenario]._scenario = scenario
            self.IntegerScenarioModels[scenario]._energyOperating = self.relaxedScenarioModels[
                scenario]._energyOperating

        # create forward scheduling solution
        self.solutionProcedure = ScenarioSolutionProcedure(parameters, self.relaxedScenarioModels,
                                                           self.IntegerScenarioModels, scenarios, self)

    def solve_ScenarioProblem_BranchNBound(self, scenario, solutionLimit, iteration, RHS):
        """
        When Benders Decomposition is used, this function changes the right hand side of the corresponding variabels.\
        """
        generated_cuts = None
        usedShifts = self.solutionProcedure.construction_hfss(scenario, solutionLimit, iteration, RHS)
        # generated cuts are removed, because Benders Decomposition requires for dedicated BnP algorihm, out of scope for now.
        # generated_cuts = self.solutionProcedure.branchNbounch_hfss(scenario, solutionLimit, iteration,RHS)
        return generated_cuts, usedShifts


    def changeVariableTypes(self, model: gp.Model, model_toBeCopied: gp.Model, scenario):
        """
        In the relaxed Problem, integer and binary variables are continuous. Therefore, they are also continuous in the new model after calling the copy() function. To match the model to its Mixed-Integer Formulation, they variables pointers and variables types are updated here.
        """

        # changing varibles types follows allways the same logic. It is iterated through the defined varibles indices in the relaxed problem, variables retrieved with the gurobi function get variables by name and then the vartype is changed accordingly. Finally, the new variable pointer is saved.
        model._repair_batch_shift = gp.tupledict()
        for (position, machine, shift) in model_toBeCopied._repair_batch_shift.keys():
            var = model.getVarByName(f'{scenario}_repair_of_batch_executed_in_shift[{position},{machine},{shift}]')
            var.setAttr(GRB.Attr.VType, GRB.BINARY)
            model._repair_batch_shift[(position, machine, shift)] = var

        # see above
        model._maintenance_batch_machine = gp.tupledict()
        for (maintenance, batch, machine) in model_toBeCopied._maintenance_batch_machine.keys():
            var = model.getVarByName(f'{scenario}_maintenance_batch_machine[{maintenance},{batch},{machine}]')
            var.setAttr(GRB.Attr.VType, GRB.BINARY)
            model._maintenance_batch_machine[(maintenance, batch, machine)] = var

        # see above
        model._repair_batch_machine = gp.tupledict()
        for (repair, batch, machine) in model_toBeCopied._repair_batch_machine.keys():
            var = model.getVarByName(f'{scenario}_repair_batch_machine[{repair},{batch},{machine}]')
            var.setAttr(GRB.Attr.VType, GRB.BINARY)
            model._repair_batch_machine[(repair, batch, machine)] = var

        # see above
        model._start_repair_batch_machine_slot = gp.tupledict()
        for (batch, machine, slot) in model_toBeCopied._start_repair_batch_machine_slot.keys():
            var = model.getVarByName(f'{scenario}_start_repair_batch_machine_slot[{batch},{machine},{slot}]')
            var.setAttr(GRB.Attr.VType, GRB.BINARY)
            model._start_repair_batch_machine_slot[(batch, machine, slot)] = var

        # see above
        model._shut_down_after_batch = gp.tupledict()
        for (batch, machine) in model_toBeCopied._shut_down_after_batch.keys():
            var = model.getVarByName(f'{scenario}_shut_down_after_batch[{batch},{machine}]')
            var.setAttr(GRB.Attr.VType, GRB.BINARY)
            model._shut_down_after_batch[(batch, machine)] = var

        # see above
        model._rescheduling_repair = gp.tupledict()
        for (stage, shift) in model_toBeCopied._rescheduling_repair.keys():
            var = model.getVarByName(f'{scenario}_rescheduling_repair[{stage},{shift}]')
            var.setAttr(GRB.Attr.VType, GRB.INTEGER)
            model._rescheduling_repair[(stage, shift)] = var

        # Add time bucket variables as Special-Order-Set to improve process times
        sosVarsRepair = gp.tupledict()
        for (batch, machine) in model_toBeCopied._start_batch_machine.keys():
            sosVarsRepair[batch, machine] = []

        for (batch, machine, slot), var in model._start_repair_batch_machine_slot.items():
            sosVarsRepair[batch, machine].append(var)

        for (batch, machine) in model_toBeCopied._start_batch_machine.keys():
            model.addSOS(GRB.SOS_TYPE1, sosVarsRepair[batch, machine])


    def update_varPointer_toModel(self, model: gp.Model, model_toBeCopied, scenario):
        # need to copy all continuous variables as well to allow for  greedy heuristic

        model._peakEnergyConsumption = model.getVarByName(f'{scenario}_peak_energy')

        model._start_batch_machine = gp.tupledict()
        for (batch, machine) in model_toBeCopied._start_batch_machine.keys():
            var = model.getVarByName(f'{scenario}_start_batch_machine[{batch},{machine}]')
            model._start_batch_machine[(batch, machine)] = var
            if var == None:
                raise ValueError

        model._end_batch_machine = gp.tupledict()
        for (batch, machine) in model_toBeCopied._end_batch_machine.keys():
            var = model.getVarByName(f'{scenario}_end_batch_machine[{batch},{machine}]')
            model._end_batch_machine[(batch, machine)] = var
            if var == None:
                raise ValueError

        model._maintenance_operationTime_batch_machine = gp.tupledict()
        for (maintenance, batch, machine) in model_toBeCopied._maintenance_operationTime_batch_machine.keys():
            var = model.getVarByName(f'{scenario}_maintenance_operation_batch_machine[{maintenance},{batch},{machine}]')
            model._maintenance_operationTime_batch_machine[(maintenance, batch, machine)] = var
            if var == None:
                raise ValueError

        model._repair_operationTime_batch_machine = gp.tupledict()
        for (repair, batch, machine) in model_toBeCopied._repair_operationTime_batch_machine.keys():
            # -1 used for initial state
            if batch >= 0:
                var = model.getVarByName(f'{scenario}_repair_operation_batch_machine[{repair},{batch},{machine}]')
                model._repair_operationTime_batch_machine[(repair, batch, machine)] = var
                if var == None:
                    raise ValueError

        model._duration_service_after_batch = gp.tupledict()
        for (batch, machine) in model_toBeCopied._duration_service_after_batch.keys():
            var = model.getVarByName(f'{scenario}_duration_service_block_after_batch[{batch},{machine}]')
            model._duration_service_after_batch[(batch, machine)] = var
            if var == None:
                raise ValueError

        model._duration_repair_after_batch = gp.tupledict()
        for (batch, machine) in model_toBeCopied._duration_repair_after_batch.keys():
            var = model.getVarByName(f'{scenario}_duration_repair_block_after_batch[{batch},{machine}]')
            model._duration_repair_after_batch[(batch, machine)] = var
            if var == None:
                raise ValueError

        model._startTime_shutDown = gp.tupledict()
        for (batch, machine) in model_toBeCopied._startTime_shutDown.keys():
            var = model.getVarByName(f'{scenario}_startTime_shutDown[{batch},{machine}]')
            model._startTime_shutDown[(batch, machine)] = var
            if var == None:
                raise ValueError

        model._startTime_serviceBlock = gp.tupledict()
        for (batch, machine) in model_toBeCopied._startTime_serviceBlock.keys():
            var = model.getVarByName(f'{scenario}_startTime_serviceBlock[{batch},{machine}]')
            model._startTime_serviceBlock[(batch, machine)] = var
            if var == None:
                raise ValueError

        model._startTime_warmUp = gp.tupledict()
        for (batch, machine) in model_toBeCopied._startTime_warmUp.keys():
            var = model.getVarByName(f'{scenario}_startTime_warmUp[{batch},{machine}]')
            model._startTime_warmUp[(batch, machine)] = var
            if var == None:
                raise ValueError

        model._completion_time_max = model.getVarByName(f'{scenario}_completion_Time')

    def copy_ModelAttrributes(self, model: gp.Model, model_toBeCopied: gp.Model):
        """
        Because scenario problems are initialized without batch dependency ignore in copy
        """
        # copy all model attributes safed in model._ to the new IP model
        if hasattr(model_toBeCopied, '_batch_dependency'):
            model._batch_dependency = model_toBeCopied._batch_dependency
            model._batch_processing_time = model_toBeCopied._batch_processing_time
            model._maintenance_batch_machine = model_toBeCopied._maintenance_batch_machine
            model._set_position_per_machine = model_toBeCopied._set_position_per_machine
            model._set_shifts = model_toBeCopied._set_shifts
            model._set_position_machine_slot_end = model_toBeCopied._set_position_machine_slot_end
            model._set_repair_position_machine = model_toBeCopied._set_repair_position_machine
            self._set_repair_position_machine = model_toBeCopied._set_repair_position_machine
            self._parameter_repair = model_toBeCopied._parameter_repair

        model._set_maintenance_by_machine = model_toBeCopied._set_maintenance_by_machine
        model._set_repair_by_machine = model_toBeCopied._set_repair_by_machine
        model._set_maintenance_machine = model_toBeCopied._set_maintenance_machine
        model._set_repair_machine = model_toBeCopied._set_repair_machine
        model._max_number_slots = model_toBeCopied._max_number_slots

        model._parameter_maintenance = model_toBeCopied._parameter_maintenance
        model._parameter_repair = model_toBeCopied._parameter_repair
        model._machines = model_toBeCopied._machines
        model._postions_per_machine = model_toBeCopied._postions_per_machine
        model._machines_per_stage = model_toBeCopied._machines_per_stage
        model._stages = model_toBeCopied._stages

        model._slotRange_per_batchNmachine = model_toBeCopied._slotRange_per_batchNmachine

        model.update()

    def copy_model_as_MIP(self, model_toBeCopied, scenario, varKeyScenario):

        model_toBeCopied.update()

        copy_start_time = time.time()
        model = model_toBeCopied.copy()
        # remap variables
        # list all variables, for which duals are required
        # no var type setting for rescheduling as handled by optimization
        self.changeVariableTypes(model, model_toBeCopied, varKeyScenario)

        self.update_varPointer_toModel(model, model_toBeCopied, varKeyScenario)

        # copy constraitns
        self.copyReschedulingConstr(model, varKeyScenario)

        self.copy_ModelAttrributes(model, model_toBeCopied)

        finish_time = time.time() - copy_start_time
        print('Finished Model copy after ' + str(finish_time))

        # warm start model with MIP solution:
        create_initial_solution(self, model, scenario)

        return model

    def copyReschedulingConstr(self, model, scenario):
        """
        Update constraint pointers to be able to use them later.
        """
        model._constraint_rescheduling_repair = {
            (shift, stage): model.getConstrByName(f'{scenario}_rescheduling_cost_repair_constraint[{shift},{stage}]')
            for shift
            in
            self.parameters.set_shifts.keys() for stage in self.parameters.set_stages}

        model._lowerBoundConstraint = {
            stage: model.getConstrByName(f'{scenario}_reschedulingLowerBound[{stage}]') for stage in
            self.parameters.set_stages}

        model._dummyLower = {
            stage: model.getConstrByName(f'{scenario}_dummyLower[{shift},{stage}]') for shift
            in
            self.parameters.set_shifts.keys() for stage in self.parameters.set_stages}

        model._dummyUpper = {
            stage: model.getConstrByName(f'{scenario}_dummyUpper[{shift},{stage}]') for shift
            in
            self.parameters.set_shifts.keys() for stage in self.parameters.set_stages}

        model.update()

    def defining_HFSS_model(self, parameters, hfss):
        '''
        The Hybrid flow shop model is defined as Gurobi Model.
        :param input_data:
        :return:
        '''

        # Define Maintenance Task
        self.define_maintenance_task(parameters, hfss)

        self.save_attributes_to_model(hfss, parameters)

        # saved variables
        saved_variables = 0
        for machine in parameters.positions_per_machine.keys():
            for position in range(0, parameters.positions_per_machine[machine] + 1):
                lower_limit = max(position * parameters.lowest_processTime_per_machine[machine] - 1, 0)
                # upper_limit = min(((parameters.max_number_slots-1) - (parameters.positions_per_machine[machine] - 1 - position) * parameters.lowest_processTime_per_machine[machine]) + 1,parameters.max_number_slots-1)
                upper_limit = parameters.max_number_slots
                parameters.slotRange_per_batchNmachine[(position, machine)] = (lower_limit, upper_limit)
                saved_variables += (parameters.max_number_slots - 1) - (upper_limit - lower_limit)
        hfss._slotRange_per_batchNmachine = parameters.slotRange_per_batchNmachine

        return hfss

    def define_pricing_problem(self, parameters):
        # Generate Pricing Problem
        TeamOP = multi_visit_TOP(parameters)
        self.batch_dependency, self.batch_processing_time = TeamOP.generate_initial_soltuion(parameters)

        TeamOP.generate_pricing_problem(parameters)

        return TeamOP

    def define_RestrictedMasterProblem(self, hfss, scenario, scenarioKey, parameters, afterPricingDependency,
                                       AfterPricingProcessTime, selectedPattern):
        # define decision variables
        time_dv = time.time()
        relaxedScenarioProblem = gp.Model()
        relaxedScenarioProblem._scenario = scenario
        self.save_attributes_to_model(relaxedScenarioProblem, parameters)
        self.copy_ModelAttrributes(relaxedScenarioProblem, hfss)

        self.add_decision_variables_to_model(relaxedScenarioProblem, MIP_indicator=False, scenario="",
                                             parameters=parameters, scenarioKey=scenarioKey)
        past_time = time.time() - time_dv
        print(f'### It took to create decision varialbes, {past_time:.2f}. ###')

        # define constraints
        self.add_constraints_to_model(relaxedScenarioProblem, "", parameters, selectedPattern, afterPricingDependency,
                                      AfterPricingProcessTime)

        obj_rescheduling_cost_repair, obj_peakEnergyConsumption = self.modelSetObjective(relaxedScenarioProblem,
                                                                                         parameters)

        relaxedScenarioProblem.setObjective(obj_rescheduling_cost_repair + obj_peakEnergyConsumption, GRB.MINIMIZE)

        relaxedScenarioProblem.update()

        return relaxedScenarioProblem

    def removeRedundantVariables(self, parameters, model, CrossDependecy, ProcessTimes):

        # forward schedule to determine earliest start and earliest end time
        forwardSchedulingStart = {(position, machine): np.inf for (position, machine) in ProcessTimes[1].keys()}

        for machine in model._postions_per_machine.keys():
            for position in reversed(range(0, parameters.positions_per_machine[machine])):
                removePosition = True
                for pattern in CrossDependecy.keys():
                    if ProcessTimes[pattern][(position, machine)] != 0:
                        removePosition = False
                if removePosition:
                    del forwardSchedulingStart[(position, machine)]
                    model._postions_per_machine[machine] = model._postions_per_machine[machine] - 1
                else:
                    break

        forwardUsedProTime = {(position, machine): np.inf for (position, machine) in forwardSchedulingStart.keys()}

        for pattern in CrossDependecy.keys():
            # determine shortest process time per machine
            minProcessTime = {(position, machine): np.inf for (position, machine) in ProcessTimes[1].keys()}
            for posNmacKey, timePosition in ProcessTimes[pattern].items():
                if minProcessTime[posNmacKey] > timePosition and timePosition > 0:
                    minProcessTime[posNmacKey] = timePosition
            # get previous position
            crossAssignment = gp.tupledict()
            for (prevP, prevM, position, machine), crossDependent in CrossDependecy[pattern].items():
                if crossDependent == 1:
                    crossAssignment[(position, machine)] = (prevP, prevM)

            for index, stage in enumerate(parameters.set_stages):
                # iterate through all machines per stage
                for machine in parameters.machines_per_stage[stage]:
                    # iterate thorugh all positions of machines
                    for position in range(0, parameters.positions_per_machine[machine]):
                        # check earliest end time of prior position
                        if position > 0:
                            startLimit = forwardSchedulingStart[(position - 1, machine)] + minProcessTime[
                                (position, machine)]
                        else:
                            startLimit = 0
                        # check earliest end time of prior stage coil
                        if index > 0:
                            # check if position is non empty
                            if ProcessTimes[pattern][(position, machine)] > 0:
                                # position in previous stage
                                crossStart = forwardSchedulingStart[crossAssignment[(position, machine)]] + \
                                             minProcessTime[crossAssignment[(position, machine)]]
                                if startLimit <= crossStart:
                                    startLimit = crossStart

                        # assign highest value
                        if forwardSchedulingStart[(position, machine)] > startLimit:
                            forwardSchedulingStart[(position, machine)] = int(startLimit)
                            forwardUsedProTime[(position, machine)] = minProcessTime[(position, machine)]

        self.forwardScheduleTime = forwardUsedProTime

        # backward schedule to determine latest end and latest start time
        # forward schedule to determine earliest start and earliest end time

        latestTime = parameters.max_number_slots - 1
        backwardsSchedulingStart = {(position, machine): 0 for (position, machine) in forwardSchedulingStart.keys()}
        backwardsUsedProTime = {(position, machine): 0 for (position, machine) in forwardSchedulingStart.keys()}
        for pattern in CrossDependecy.keys():
            # determine shortest process time per machine
            minProcessTime = {(position, machine): np.inf for (position, machine) in ProcessTimes[1].keys()}
            for posNmacKey, timePosition in ProcessTimes[pattern].items():
                if minProcessTime[posNmacKey] > timePosition:
                    minProcessTime[posNmacKey] = timePosition
            # get previous position
            crossAssignment = gp.tupledict()
            for (prevP, prevM, position, machine), crossDependent in CrossDependecy[pattern].items():
                if crossDependent == 1:
                    crossAssignment[(prevP, prevM)] = (position, machine)
            for index, stage in enumerate(reversed(parameters.set_stages)):

                # iterate through all machines per stage
                for machine in parameters.machines_per_stage[stage]:
                    # iterate thorugh all positions of machines
                    for position in reversed(range(0, parameters.positions_per_machine[machine])):
                        # check earliest end time of prior position
                        # check earliest end time of prior position
                        if position < parameters.positions_per_machine[machine] - 1:
                            endLimit = backwardsSchedulingStart[(position + 1, machine)] - minProcessTime[
                                (position + 1, machine)]
                        else:
                            endLimit = parameters.max_number_slots - 1
                            # check earliest end time of prior stage coil
                        if index > 0:
                            # check if position is non empty
                            if ProcessTimes[pattern][(position, machine)] > 0:
                                # position in previous stage
                                crossStart = backwardsSchedulingStart[crossAssignment[(position, machine)]] - \
                                             minProcessTime[crossAssignment[(position, machine)]]
                                if endLimit >= crossStart:
                                    endLimit = crossStart

                        # assign highest value
                        if backwardsSchedulingStart[(position, machine)] < endLimit:
                            backwardsSchedulingStart[(position, machine)] = int(endLimit)
                            backwardsUsedProTime[(position, machine)] = minProcessTime[(position, machine)]

        self.backwardScheduleTime = backwardsUsedProTime


        # remove all positions that are not used
        for tupKey in forwardSchedulingStart.keys():
            model._slotRange_per_batchNmachine[tupKey] = (
            forwardSchedulingStart[tupKey], backwardsSchedulingStart[tupKey])


    def modelSetObjective(self, model, parameters):
        config = parameters.config
        # determine total cost for ad-hoc scheduled reparis
        obj_rescheduling_cost_repair = gp.quicksum(parameters.rescheduling_data_cost[
                                                       (parameters.schedulingCategory.repair, stage)] *
                                                   model._rescheduling_repair[(stage, shift)] for
                                                   (stage, shift) in model._rescheduling_repair.keys())

        # determine total energy cost based on energy consumption per machine and given electricity price
        obj_summedEnergyConsumption = gp.quicksum(
            model._totalEnergyconsumptionMachine[machine] * parameters.input_data[config.data.generalPar]['EnergyCost'][
                'ParameterValue'] for machine in model._machines)


        return obj_rescheduling_cost_repair, obj_summedEnergyConsumption

    def addBack_removed_CrossDependency(self, model: gp.Model):
        print('Add back previously removed dependency constraints ' + str(len(self.saved_removed_cons.keys())))
        for addCon in self.saved_removed_cons.keys():
            lhs, sense, rhs, name = self.saved_removed_cons[addCon]
            model._constraint_cross_dependency[addCon] = model.addConstr(lhs, sense, rhs, name)
        model.update

    def define_MIP_Problem(self, hfss):

        config = self.config
        scenario = self.scenario
        input_data = self.input_data

        # parition data
        process_data = input_data[config.data.process]
        maintenance_data = input_data[config.data.maintenance]

        print('Starting IP')
        hfss_IP = gp.Model('HFSS as MasterProblem')

        self.save_attributes_to_model(hfss_IP)
        # load patterns from Restricted Master Problem
        self.add_generated_patterns(hfss_IP, hfss)
        self.define_maintenance_task(maintenance_data, hfss_IP, config)

        # define decision variables
        time_dv = time.time()
        self.add_decision_variables_to_model(hfss_IP, MIP_indicator=True)
        past_time = time.time() - time_dv
        print(f'### It took to create decision varialbes, {past_time:.2f}. ###')

        # define constraints
        self.add_constraints_to_model(hfss_IP, config, scenario, IP=True)

        create_initial_solution(self, hfss_IP, self.upperLimitRepair, self.lowerLimitRepair)
        hfss_IP.setObjective(self.obj_completion_time, GRB.MINIMIZE)
        hfss_IP.update()

        # self.create_initial_solution(process_data, config)
        # hfss.Params.MIPFocus = 3
        hfss_IP.Params.LazyConstraints = 1

        hfss_IP._fileDir = 'solution/' + scenario
        hfss_IP._filePath = hfss_IP._fileDir + '/leaf'
        if not os.path.exists(hfss_IP._fileDir):
            # if the corresponding scenario directory is not present then create it.
            os.makedirs(hfss_IP._fileDir)
        hfss_IP.Params.SolFiles = hfss_IP._filePath
        hfss_IP.Params.MIPGap = 0.05

        hfss_IP.update()
        hfss_IP.optimize()

        fixed_start = time.time()
        # generate cuts from stored solution
        for solution in os.listdir(hfss_IP._fileDir):
            filePath = hfss_IP._fileDir + '/' + solution
            hfss_IP.read(filePath)
            fixedModel = hfss_IP.fixed()
            print('Finished Fixing Model')
            print((time.time() - fixed_start))

        print('###################################')
        print('')
        print('done')

        return hfss

    def save_attributes_to_model(self, hfss, parameters):
        # define the required position per machine as set for easier access
        hfss._set_position_per_machine_per_stage = gp.tupledict()
        for stage in parameters.set_stages:
            hfss._set_position_per_machine_per_stage[stage] = {}
            for machine in parameters.machines_per_stage[stage]:
                hfss._set_position_per_machine_per_stage[stage][machine] = parameters.positions_per_machine[machine]
        parameters.set_position_per_machine_per_stage = hfss._set_position_per_machine_per_stage

        # safe parameters for easier access
        self.determine_cross_dependency(parameters)
        hfss._stages = parameters.set_stages
        hfss._machines = parameters.machines
        hfss._machines_per_stage = parameters.machines_per_stage
        hfss._postions_per_machine = parameters.positions_per_machine
        hfss._jobs = parameters.jobs
        hfss._max_number_slots = parameters.max_number_slots
        hfss._slots = parameters.slots

    def add_initial_generated_pattern(self, hfss):
        # append first generated pattern
        hfss._pattern_counter = 1
        hfss._batch_dependency = {}
        hfss._batch_processing_time = {}
        hfss._batch_dependency[hfss._pattern_counter] = self.batch_dependency
        hfss._batch_processing_time[hfss._pattern_counter] = self.batch_processing_time

    def add_generated_patterns(self, hfss_IP, hfss_RMP):
        hfss_IP._batch_dependency = hfss_RMP._batch_dependency
        hfss_IP._batch_processing_time = hfss_RMP._batch_processing_time
        hfss_IP._RMP_generated_patterns = hfss_RMP._generated_pattern

    def define_maintenance_task(self, parameter, model):
        # set_repair_machine and set_maintenance_machine are generated earlier
        model._set_repair_machine = gp.tuplelist()
        model._set_repair_by_machine = {machine: None for machine in parameter.machines}
        model._set_maintenance_machine = gp.tuplelist()
        model._set_maintenance_by_machine = {machine: None for machine in parameter.machines}

        # initialize sets
        for machine in parameter.machines:
            model._set_repair_by_machine[machine] = []
            model._set_maintenance_by_machine[machine] = []

        self.define_duration_and_limit_maintenance(model, parameter)

    def define_duration_and_limit_maintenance(self, model, parameter):
        task_keys = parameter.task_keys
        maintenance_task = parameter.maintenance_data
        model._parameter_repair = gp.tupledict()
        model._parameter_maintenance = gp.tupledict()
        for task in maintenance_task:
            # rearrange all repairs task to fit in desired dictionary format
            if task[task_keys.shutdown]:
                model._parameter_repair[(task[task_keys.task], task[task_keys.machine])] = {
                    task_keys.duration: task[task_keys.duration],
                    task_keys.upper_limit: {scenario: task[scenario] for scenario in task_keys.scenarios.keys()}}
                model._set_repair_machine.append((task[task_keys.task], task[task_keys.machine]))
                model._set_repair_by_machine[task[task_keys.machine]].append(task[task_keys.task])
            else:
                # if no shut down requied, then they are service task, which are not differentiated
                model._parameter_maintenance[(task[task_keys.task], task[task_keys.machine])] = {
                    task_keys.duration: task[task_keys.duration], task_keys.upper_limit: task['Average']}
                model._set_maintenance_machine.append((task[task_keys.task], task[task_keys.machine]))
                model._set_maintenance_by_machine[task[task_keys.machine]].append(task[task_keys.task])
        parameter._set_maintenance_by_machine = model._set_maintenance_by_machine
        parameter._set_repair_by_machine = model._set_repair_by_machine

    def add_decision_variables_to_model(self, hfss, MIP_indicator: False, scenario, parameters, scenarioKey):
        '''
        Create all decision variables for optimzation model.
        '''
        print('### Starting to create Decision Variables. ###')

        hfss._set_stages = parameters.set_stages
        hfss._set_shifts = parameters.set_shifts
        hfss._slots = parameters.slots

        # define sets per machine
        hfss._set_position_per_machine = gp.tuplelist()
        hfss._set_position_by_machine = gp.tupledict()
        hfss._set_maintenance_position_machine = gp.tuplelist()
        hfss._set_repair_position_machine = gp.tuplelist()
        for machine, positions in hfss._postions_per_machine.items():
            hfss._set_position_by_machine[machine] = []
            for position in range(0, positions):
                hfss._set_position_by_machine[machine].append(position)
                hfss._set_position_per_machine.append((position, machine))
                for maintenance in hfss._set_maintenance_by_machine[machine]:
                    hfss._set_maintenance_position_machine.append((maintenance, position, machine))
                for repair in hfss._set_repair_by_machine[machine]:
                    hfss._set_repair_position_machine.append((repair, position, machine))

        parameters.set_position_per_machine = hfss._set_position_per_machine

        # determine lower and upper limit on slots
        hfss._set_position_machine_slot = gp.tuplelist()
        hfss._set_slots_per_positionNmachine = {(position, machine): [] for (position, machine) in
                                                hfss._set_position_per_machine}
        hfss._set_position_machine_slot_end = gp.tuplelist()
        hfss._set_slots_per_positionNmachine_end = {(position, machine): [] for (position, machine) in
                                                    hfss._set_position_per_machine}

        # determine the possible slots per positions to narrow down formulation
        for (position, machine) in hfss._set_position_per_machine:
            lower_limit_slot = parameters.slotRange_per_batchNmachine[(position, machine)][0]
            upper_limit_limit_slot = parameters.slotRange_per_batchNmachine[(position, machine)][1]
            for slot in range(lower_limit_slot, upper_limit_limit_slot + 1):
                hfss._set_position_machine_slot.append((position, machine, slot))
                hfss._set_slots_per_positionNmachine[(position, machine)].append(slot)
            # adjust for end slot, if forwardSchedule is available the model can be narraoed down
            if hasattr(self, "forwardScheduleTime"):
                lower_limit_slot = parameters.slotRange_per_batchNmachine[(position, machine)][0] + \
                                   self.forwardScheduleTime[(position, machine)]
                if (position + 1, machine) in self.backwardScheduleTime.keys():
                    upper_limit_limit_slot = parameters.slotRange_per_batchNmachine[(position + 1, machine)][1] - \
                                             self.backwardScheduleTime[(position + 1, machine)]
                else:
                    upper_limit_limit_slot = parameters.max_number_slots - 1
                for slot in range(lower_limit_slot, upper_limit_limit_slot + 1):
                    hfss._set_position_machine_slot_end.append((position, machine, slot))
                    hfss._set_slots_per_positionNmachine_end[(position, machine)].append(slot)
            # if no forward scheduling is availabe us standard assignemnt where only lowest processing time is considered
            else:
                lower_limit_slot = parameters.slotRange_per_batchNmachine[(position + 1, machine)][0]
                upper_limit_limit_slot = parameters.slotRange_per_batchNmachine[(position + 1, machine)][1]
                for slot in range(lower_limit_slot, upper_limit_limit_slot + 1):
                    hfss._set_position_machine_slot_end.append((position, machine, slot))
                    hfss._set_slots_per_positionNmachine_end[(position, machine)].append(slot)

        # additional cost, if repair is re-scheduled in stage slot
        hfss._rescheduling_repair = hfss.addVars(hfss._set_stages, hfss._set_shifts.keys(), vtype=GRB.CONTINUOUS,
                                                 lb=0, name=scenario + '_rescheduling_repair')


        hfss._set_position_machine_shift = gp.tuplelist()
        for shift in hfss._set_shifts.keys():
            for (position, machine) in hfss._set_position_per_machine:
                hfss._set_position_machine_shift.append((position, machine, shift))


        # determine if repair starts in batch i
        hfss._repair_batch_shift = gp.tupledict()

        for (position, machine) in hfss._set_position_per_machine:
            if not hasattr(self, "forwardScheduleTime"):
                for shift in hfss._set_shifts.keys():
                    # repair cannot be delayed into shift
                    if not parameters.slotRange_per_batchNmachine[(position + 1, machine)][1] < hfss._set_shifts[shift][
                        0]:
                        # repair cannot start as early as shift ends
                        if not hfss._set_shifts[shift][1] < \
                               parameters.slotRange_per_batchNmachine[(position + 1, machine)][
                                   0]:
                            hfss._repair_batch_shift[position, machine, shift] = hfss.addVar(vtype=GRB.CONTINUOUS,
                                                                                             lb=0, ub=1,
                                                                                             name=f'{scenario}_repair_of_batch_executed_in_shift[{position},{machine},{shift}]')

            else:

                for shift in hfss._set_shifts.keys():

                    if position < hfss._postions_per_machine[machine] - 1:
                        upper_limit_slot = parameters.slotRange_per_batchNmachine[(position + 1, machine)][1] - \
                                           self.backwardScheduleTime[(position + 1, machine)]
                    else:
                        upper_limit_slot = parameters.max_number_slots
                    # repair cannot be delayed into shift
                    if not upper_limit_slot < hfss._set_shifts[shift][0]:
                        # repair cannot start as early as shift ends
                        if not hfss._set_shifts[shift][1] < parameters.slotRange_per_batchNmachine[(position, machine)][
                            0] + \
                               self.forwardScheduleTime[(position, machine)]:
                            hfss._repair_batch_shift[position, machine, shift] = hfss.addVar(vtype=GRB.CONTINUOUS,
                                                                                             lb=0, ub=1,
                                                                                             name=f'{scenario}_repair_of_batch_executed_in_shift[{position},{machine},{shift}]')


        # starting time of batch k on machine j
        hfss._start_batch_machine = hfss.addVars(hfss._set_position_per_machine, vtype=GRB.CONTINUOUS, lb=0.0,
                                                 name=scenarioKey + '_start_batch_machine')

        # end time of batch k on machine j
        hfss._end_batch_machine = hfss.addVars(hfss._set_position_per_machine, vtype=GRB.CONTINUOUS, lb=0.0,
                                               name=scenarioKey + '_end_batch_machine')

        # operation time on machine
        hfss._totalEnergyconsumptionMachine = hfss.addVars(hfss._machines, vtype=GRB.CONTINUOUS,
                                                           name=scenarioKey + '_energyConsumptionMachine')

        # operation time after batch by maintenance
        hfss._maintenance_operationTime_batch_machine = hfss.addVars(hfss._set_maintenance_position_machine,
                                                                     vtype=GRB.CONTINUOUS,
                                                                     name=scenarioKey + '_maintenance_operation_batch_machine')

        # repair after batch
        hfss._repair_batch_machine = gp.tupledict()
        hfss._repair_batch_machine = hfss.addVars(hfss._set_repair_position_machine, vtype=GRB.CONTINUOUS, lb=0.0,
                                                  ub=1.0,
                                                  name=scenarioKey + '_repair_batch_machine')

        hfss._maintenance_batch_machine = hfss.addVars(hfss._set_maintenance_position_machine, vtype=GRB.CONTINUOUS,
                                                       lb=0.0,
                                                       ub=1.0, name=scenarioKey + '_maintenance_batch_machine')

        # operation time after batch by repair
        hfss._repair_operationTime_batch_machine = hfss.addVars(hfss._set_repair_position_machine, vtype=GRB.CONTINUOUS,
                                                                name=scenarioKey + '_repair_operation_batch_machine')

        # duration block for maintenance and repair
        hfss._duration_service_after_batch = hfss.addVars(hfss._set_position_per_machine, vtype=GRB.CONTINUOUS,
                                                          name=scenarioKey + '_duration_service_block_after_batch')

        # duration block for maintenance and repair
        hfss._duration_repair_after_batch = hfss.addVars(hfss._set_position_per_machine, vtype=GRB.CONTINUOUS,
                                                         name=scenarioKey + '_duration_repair_block_after_batch')

        # binary indicator if shut down is performed in block after batch
        hfss._shut_down_after_batch = hfss.addVars(hfss._set_position_per_machine, vtype=GRB.CONTINUOUS,
                                                   name=scenarioKey + '_shut_down_after_batch')


        hfss._start_repair_batch_machine_slot = gp.tupledict()
        # continuous times
        hfss._startTime_shutDown = gp.tupledict()
        hfss._startTime_repairBlock = gp.tupledict()
        hfss._startTime_serviceBlock = gp.tupledict()
        hfss._startTime_warmUp = gp.tupledict()
        for stage in hfss._set_stages:
            for machine in parameters.machines_per_stage[stage]:
                # check if there is a shut down and warm up for stage
                for position in hfss._set_position_by_machine[machine]:

                    # start time of m&r block after batch b
                    hfss._startTime_repairBlock[position, machine] = hfss.addVar(vtype=GRB.CONTINUOUS,
                                                                                 name=f'{scenarioKey}_startTime_repairBlock[{position},{machine}]')

                    # start time of shut down after batch b
                    hfss._startTime_shutDown[(position, machine)] = hfss.addVar(
                        vtype=GRB.CONTINUOUS,
                        name=f'{scenarioKey}_startTime_shutDown[{position},{machine}]')
                    # start time of m&r block after batch b
                    hfss._startTime_serviceBlock[(position, machine)] = hfss.addVar(
                        vtype=GRB.CONTINUOUS,
                        name=f'{scenarioKey}_startTime_serviceBlock[{position},{machine}]')

                    # start time of warm-up block after batch b
                    hfss._startTime_warmUp[(position, machine)] = hfss.addVar(
                        vtype=GRB.CONTINUOUS,
                        name=f'{scenarioKey}_startTime_warmUp[{position},{machine}]')

                    # define slot variables only for required time frame. After production assignments are generated, i.e. ForwardScheduledTime is available the time interval can be signifancty reduced.
                    if not hasattr(self, "forwardScheduleTime"):
                        for slot in hfss._set_slots_per_positionNmachine_end[(position, machine)]:

                            hfss._start_repair_batch_machine_slot[(position, machine, slot)] = hfss.addVar(
                                vtype=GRB.CONTINUOUS, lb=0.0, ub=1,
                                name=f'{scenarioKey}_start_repair_batch_machine_slot[{position},{machine},{slot}]')
                    else:
                        if position < hfss._postions_per_machine[machine] - 1:
                            upper_limit_slot = parameters.slotRange_per_batchNmachine[(position + 1, machine)][
                                                   1] - self.backwardScheduleTime[(position + 1, machine)]
                        else:
                            upper_limit_slot = parameters.max_number_slots
                        for t in range(parameters.slotRange_per_batchNmachine[(position, machine)][0] +
                                       self.forwardScheduleTime[(position, machine)], upper_limit_slot + 1):
                            hfss._start_repair_batch_machine_slot[(position, machine, t)] = hfss.addVar(
                                vtype=GRB.CONTINUOUS, lb=0.0, ub=1,
                                name=f'{scenarioKey}_start_repair_batch_machine_slot[{position},{machine},{t}]')

        # determine peak energy consumption in each slot
        hfss._peakEnergyConsumption = hfss.addVar(vtype=GRB.CONTINUOUS, name=scenarioKey + '_peak_energy')

        # max completion time
        hfss._completion_time_max = hfss.addVar(1, vtype=GRB.CONTINUOUS, name=scenarioKey + '_completion_Time')

    def change_RHS_from_FristStage(self, newRightHandSide, scenario):
        '''
        If the first stage decision changes, update RHS of the scenario.
        :param model:
        :param newRightHandSide: format {constraint_name: {(stage, shift): newRHS, ... }
        :return:
        '''
        config = self.parameters.config
        # change right-hand-side for maintenance rescheudling
        first_stage_key = config.model_discrete.decision_variables
        # update frist Stage decision for solutio procedure
        self.firstStage_repair = newRightHandSide[first_stage_key.scheduled_repair]

        model = self.relaxedScenarioModels[scenario]

        first_stage_key = config.model_discrete.decision_variables
        newRHS_Constraint = newRightHandSide[first_stage_key.scheduled_repair]
        # update assignment of repairs to shift and stage
        for (shift, stage) in model._constraint_rescheduling_repair.keys():
            # print(newRHS_Constraint[(stage, shift)])
            model._constraint_rescheduling_repair[(shift, stage)].RHS = newRHS_Constraint[(stage, shift)]

        newStageRHS = {stage: 0 for stage in model._stages}
        for (shift, stage) in model._constraint_rescheduling_repair.keys():
            newStageRHS[stage] += newRHS_Constraint[(stage, shift)]

        model.update()

        # update integer problem
        model = self.IntegerScenarioModels[scenario]
        # change right-hand-side for maintenance rescheudling
        newRHS_Constraint = newRightHandSide[first_stage_key.scheduled_repair]
        for (shift, stage) in model._constraint_rescheduling_repair.keys():
            model._constraint_rescheduling_repair[(shift, stage)].RHS = newRHS_Constraint[(stage, shift)]

        newStageRHS = {stage: 0 for stage in model._stages}
        for (shift, stage) in model._constraint_rescheduling_repair.keys():
            newStageRHS[stage] += newRHS_Constraint[(stage, shift)]


        model.update()

    def determine_cross_dependency(self, parameter):
        # define set for cross dependecnies
        parameter.set_batch_cross_stage_dependency = gp.tuplelist()
        for stage in range(0, len(parameter.set_stages) - 1):
            for machine_prev in parameter.machines_per_stage[parameter.set_stages[stage]]:
                for batch_prev in range(0, parameter.set_position_per_machine_per_stage[parameter.set_stages[stage]][
                    machine_prev]):
                    for machine in parameter.machines_per_stage[parameter.set_stages[stage + 1]]:
                        for batch in range(0,
                                           parameter.set_position_per_machine_per_stage[
                                               parameter.set_stages[stage + 1]][machine]):
                            stage_name = parameter.set_stages[stage]
                            parameter.set_batch_cross_stage_dependency.append(
                                (batch_prev, machine_prev, batch, machine, stage_name))

    def add_constraints_to_model(self, hfss, scenario, parameters, generatedPattern, AfterPricingDependency=None,
                                 AfterPricingProcessTime=None, ):
        '''
        Function to add all constrains to the respective model. Seperated for overview.
        :param hfss:
        :return:
        '''
        import time
        config = parameters.config
        constraint_starttime = time.time()
        self.BigM = parameters.max_number_slots

        dv_name = config.model_discrete.decision_variables
        hfss._set_shifts = parameters.set_shifts

        # define set for cross dependency per batch
        self.set_batch_cross_stage_dependency = gp.tuplelist()
        for stage in range(0, len(hfss._stages) - 1):
            for machine_prev in hfss._machines_per_stage[hfss._stages[stage]]:
                for batch_prev in range(0, hfss._set_position_per_machine_per_stage[hfss._stages[stage]][machine_prev]):
                    for machine in hfss._machines_per_stage[hfss._stages[stage + 1]]:
                        for batch in range(0,
                                           hfss._set_position_per_machine_per_stage[hfss._stages[stage + 1]][machine]):
                            stage_name = hfss._stages[stage]
                            self.set_batch_cross_stage_dependency.append(
                                (batch_prev, machine_prev, batch, machine, stage_name))

        hfss.update()

        # add rescheduling cost constraint repair
        first_stage_key = config.model_discrete.decision_variables
        fristStage_repair = self.first_stage_decision[first_stage_key.scheduled_repair]
        self.firstStage_repair = fristStage_repair
        hfss._constraint_rescheduling_repair = hfss.addConstrs((gp.quicksum(
            hfss._repair_batch_shift.sum('*', machine, shift) for machine in hfss._machines_per_stage[stage]) - (
                                                                hfss._rescheduling_repair[stage, shift]) <=
                                                                fristStage_repair[(stage, shift)] for stage in
                                                                hfss._set_stages for shift in hfss._set_shifts.keys()),
                                                               name=scenario + '_rescheduling_cost_repair_constraint')

        # some dummy constraints are added, to be used in Benders Decomposition for easier branching. Otherwise they will not affect the model
        hfss._dummyLower = hfss.addConstrs((gp.quicksum(
            decVar for machineTest in hfss._machines_per_stage[stage] for decVar in
            hfss._repair_batch_shift.select('*', machineTest, shift)) >= -1 for stage in hfss._set_stages for shift in
                                            hfss._set_shifts.keys()), name=scenario + '_dummyLower')
        hfss._dummyUpper = hfss.addConstrs((
            gp.quicksum(decVar for machineTest in hfss._machines_per_stage[stage] for decVar in
                        hfss._repair_batch_shift.select('*', machineTest, shift)) <= 100 for
            stage in hfss._set_stages for shift in hfss._set_shifts.keys()),
            name=scenario + '_dummyUpper')

        hfss.update()

        hfss.addConstrs(gp.quicksum(hfss._repair_batch_machine.sum('*', '*', machine) for machine in
                                    hfss._machines_per_stage[stage]) <= gp.quicksum(
            hfss._repair_batch_shift.sum('*', machine, '*') for machine in hfss._machines_per_stage[stage]) for stage in
                        hfss._set_stages)
        # repairs are either pre scheduled or ad hoc

        # batches cross dependency
        # determine all cross dependecy which are non-0
        constraint_set_cross_dependency = gp.tuplelist()

        # AfterPricingDependency is not empty after pricing is exectued. Knowing all assingments allows to remvoe redundancies
        if AfterPricingDependency != None:
            hfss._batch_processing_time = AfterPricingProcessTime
            hfss._batch_dependency = AfterPricingDependency

            for index in AfterPricingDependency.keys():
                for batch_prev, machine_prev, batch, machine in AfterPricingDependency[index].keys():
                    # if batch_dependency[index][prev_node, node] is equal to 1, then constraint needs to hold
                    if AfterPricingDependency[index][batch_prev, machine_prev, batch, machine] >= 1:
                        constraint_set_cross_dependency.append((batch_prev, machine_prev, batch, machine))
        else:
            # remove stage attribute as it is not necessary and simplifies later retrieval
            for (batch_prev, machine_prev, batch, machine, stage) in parameters.set_batch_cross_stage_dependency:
                constraint_set_cross_dependency.append((batch_prev, machine_prev, batch, machine))

        hfss._constraint_cross_dependency = hfss.addConstrs(
            (hfss._end_batch_machine[key[0], key[1]] - self.BigM * gp.quicksum(
                pattern * (1 - hfss._batch_dependency[index][key]) for index, pattern in
                generatedPattern.items()) <= hfss._start_batch_machine[key[2], key[3]] for key in
             constraint_set_cross_dependency),
            name=scenario + '_batch_cross_dependency')
        past_time = time.time() - constraint_starttime
        print(f'### Creating batch dependency constrains, {past_time:.2f}. ###')


        # batch can only finsih after processing time
        hfss._constraint_processing_time = hfss.addConstrs((hfss._end_batch_machine[batch, machine] - gp.quicksum(
            hfss._batch_processing_time[index][batch, machine] * pattern for index, pattern in
            generatedPattern.items()) >= hfss._start_batch_machine[batch, machine] for (batch, machine) in
                                                            hfss._set_position_per_machine),
                                                           name=scenario + '_batch_end_time')
        past_time = time.time() - constraint_starttime
        print(f'### Finish batch after processing time constrains, {past_time:.2f}. ###')


        # track operation time of machine hfss._start_warmUp_batch_machine_slot
        enKeyWarm = config.data.parameter_name.energyConsumption['warmUp']
        enKeyCool = config.data.parameter_name.energyConsumption['coolDown']
        enKeyOperating = config.data.parameter_name.energyConsumption['operating']
        enKeyIdle = config.data.parameter_name.energyConsumption['idle']
        hfss._energyOperating = parameters.energyConsumption_per_machine[enKeyOperating]

        # modelling energy consumption
        for machine in hfss._machines:
            # check if there is a shut down and warm up for stage
            if parameters.input_data[parameters.config.data.schedulingPar][parameters.stage_per_machine[machine]][
                parameters.config.data.scheduling_parameters.shutDown]:
                # add energy consumption as defined in the model
                hfss.addConstr((gp.quicksum(
                    hfss._batch_processing_time[1][batch, machine] *
                    parameters.energyConsumption_per_machine[enKeyOperating][machine]
                    + hfss._shut_down_after_batch[batch, machine] * (
                                parameters.energyConsumption_per_machine[enKeyCool][machine] *
                                parameters.time_for_state_transition[(dv_name.cool_down, machine)] +
                                parameters.energyConsumption_per_machine[enKeyWarm][machine] *
                                parameters.time_for_state_transition[(dv_name.warm_up, machine)])
                    + parameters.energyConsumption_per_machine[enKeyIdle][machine] * (
                                hfss._start_batch_machine[batch + 1, machine] - hfss._startTime_serviceBlock[
                            batch, machine]) for batch in hfss._set_position_by_machine[machine] if
                    batch < hfss._postions_per_machine[machine] - 1) <= hfss._totalEnergyconsumptionMachine[machine]

                                ), name='energyConsumption')
            else:
                # add energy consumption as defined in the model, however neglecting shutdown and warmup as they are not added to the positin to reduce variables
                hfss.addConstr((gp.quicksum(
                    hfss._batch_processing_time[1][batch, machine] *
                    parameters.energyConsumption_per_machine[enKeyOperating][machine]
                    + parameters.energyConsumption_per_machine[enKeyIdle][machine] * (
                            hfss._start_batch_machine[batch + 1, machine] - hfss._startTime_serviceBlock[
                        batch, machine]) for batch in hfss._set_position_by_machine[machine] if
                    batch < hfss._postions_per_machine[machine] - 1) <= hfss._totalEnergyconsumptionMachine[machine]

                                ), name='energyConsumption')


        past_time = time.time() - constraint_starttime
        print(f'### Machine, slot operation tiome trackiong, {past_time:.2f}. ###')

        minOperatingTimeMachine = {machine: np.inf for machine in hfss._machines}

        for index, pattern in generatedPattern.items():
            totalBatchTimeMachine = {machine: 0 for machine in hfss._machines}
            for batch, machine in hfss._start_batch_machine.keys():
                totalBatchTimeMachine[machine] += hfss._batch_processing_time[index][batch, machine]
            for machine in hfss._machines:
                if minOperatingTimeMachine[machine] > totalBatchTimeMachine[machine]:
                    minOperatingTimeMachine[machine] = totalBatchTimeMachine[machine]

        # hfss.addConstrs((hfss._operation_machine_slot.sum(machine, '*') >= hfss._shut_down_after_batch.sum('*', machine)*(parameters.energyConsumption_per_machine[enKeyWarm][machine]*parameters.time_for_state_transition[(dv_name.cool_down, machine)]+parameters.energyConsumption_per_machine[enKeyCool][machine]*parameters.time_for_state_transition[(dv_name.warm_up, machine)])+parameters.energyConsumption_per_machine[enKeyOperating][machine]*minOperatingTimeMachine[machine] for machine in hfss._machines),name=f'{scenario}_energyrequirement')

        # track time between batches for maintenance
        hfss._set_maintenance_batch_machine_startOne = gp.tuplelist()
        for (maintenance, batch, machine) in hfss._set_maintenance_position_machine:
            if batch >= 1:
                hfss._set_maintenance_batch_machine_startOne.append((maintenance, batch, machine))
        parameters._set_maintenance_batch_machine_startOne = hfss._set_maintenance_batch_machine_startOne

        # add tracker for service time as defined in the model
        hfss._constraint_maintenance_time_tracking = hfss.addConstrs(
            (gp.quicksum(hfss._batch_processing_time[index][batch, machine] * pattern for index, pattern in
                         generatedPattern.items()) + hfss._maintenance_operationTime_batch_machine[
                 maintenance, batch - 1, machine] <= hfss._maintenance_operationTime_batch_machine[
                 maintenance, batch, machine] + hfss._parameter_maintenance[(maintenance, machine)][
                 config.data.maintenance_name.upper_limit] * hfss._maintenance_batch_machine[
                 maintenance, batch, machine] for (maintenance, batch, machine) in
             hfss._set_maintenance_batch_machine_startOne), name=scenario + '_maintenance_time_tracker')

        # set operation time between maintenance to zero
        hfss._constraint_reset_maintenanceTime = hfss.addConstrs(
            (hfss._maintenance_operationTime_batch_machine[maintenance, batch, machine] <=
             hfss._parameter_maintenance[(maintenance, machine)][
                 config.data.maintenance_name.upper_limit] * (
                     1 - hfss._maintenance_batch_machine[maintenance, batch, machine]) for
             maintenance, batch, machine in hfss._set_maintenance_position_machine),
            name=scenario + '_reset_maintenance_time')

        # track time between batches for repair
        hfss._set_repair_batch_machine_startOne = gp.tuplelist()
        for (repair, batch, machine) in hfss._set_repair_position_machine:
            if batch >= 1:
                hfss._set_repair_batch_machine_startOne.append((repair, batch, machine))
        parameters._set_repair_batch_machine_startOne = hfss._set_repair_batch_machine_startOne

        # set batch -1, to initial machine state
        for machine in parameters.machines:
            for (repair, _) in hfss._set_repair_machine.select('*', machine):
                hfss._repair_operationTime_batch_machine[repair, -1, machine] = parameters.currentMachineAge[machine]

        # increase repair time
        hfss._constraint_repair_time_tracking = hfss.addConstrs(
            (gp.quicksum(hfss._batch_processing_time[index][batch, machine] * pattern for index, pattern in
                         generatedPattern.items()) + hfss._repair_operationTime_batch_machine[
                 repair, batch - 1, machine] <= hfss._repair_operationTime_batch_machine[
                 repair, batch, machine] + hfss._parameter_repair[(repair, machine)][
                 config.data.maintenance_name.upper_limit][hfss._scenario] * hfss._repair_batch_machine[
                 repair, batch, machine] for (repair, batch, machine) in hfss._set_repair_position_machine),
            name=scenario + '_repair_time_tracker')

        # set operation time between repair to zero
        hfss._constraint_reset_repairTime = hfss.addConstrs((hfss._repair_operationTime_batch_machine[
                                                                 repair, batch, machine] -
                                                             hfss._parameter_repair[(repair, machine)][
                                                                 config.data.maintenance_name.upper_limit][
                                                                 hfss._scenario] * (
                                                                     1 - hfss._repair_batch_machine[
                                                                 repair, batch, machine]) <= 0
                                                             for repair, batch, machine
                                                             in hfss._set_repair_position_machine),
                                                            name=scenario + '_reset_repair_time')

        # determine required block after batch
        hfss.addConstrs((gp.quicksum(
            hfss._maintenance_batch_machine[maintenance, batch, machine] *
            hfss._parameter_maintenance[(maintenance, machine)][
                config.data.maintenance_name.duration] for (maintenance, _) in
            hfss._set_maintenance_machine.select('*', machine)) <= hfss._duration_service_after_batch[batch, machine]
                         for
                         (batch, machine) in hfss._set_position_per_machine),
                        name=scenario + '_service_time_after_batch')

        # determine required block after batch
        hfss.addConstrs((gp.quicksum(
            hfss._repair_batch_machine[repair, batch, machine] * hfss._parameter_repair[(repair, machine)][
                config.data.maintenance_name.duration] for (repair, _) in
            hfss._set_repair_machine.select('*', machine)) <= hfss._duration_repair_after_batch[batch, machine] for
                         (batch, machine) in hfss._set_position_per_machine),
                        name=scenario + '_repair_time_after_batch')

        # determine if there is a shutdown after batch
        hfss.addConstrs((gp.quicksum(
            hfss._repair_batch_machine[repair, batch, machine] for (repair, _) in
            hfss._set_repair_machine.select('*', machine)) <= (len(hfss._set_repair_machine.select('*', machine))) *
                         hfss._shut_down_after_batch[batch, machine] for (batch, machine) in
                         hfss._set_position_per_machine), name=scenario + '_shutdown_batch_indicator')

        # determine starting time of shutdown
        for stage in hfss._set_stages:

            for machine in parameters.machines_per_stage[stage]:
                for position in hfss._set_position_by_machine[machine]:
                    # set start time of service block
                    hfss.addConstr(
                        (hfss._startTime_warmUp[position, machine] +
                         parameters.time_for_state_transition[(dv_name.warm_up, machine)] *
                         hfss._shut_down_after_batch[position, machine] == hfss._startTime_serviceBlock[
                             position, machine]),
                        name=f'{scenario}_start_time_maintenance[{position},{machine}]')

                    # schedule next batch after warmup
                    if position < hfss._postions_per_machine[machine] - 1:
                        # schedule next batch after warmup
                        hfss.addConstr(
                            (hfss._startTime_serviceBlock[position, machine] +
                             hfss._duration_service_after_batch[position, machine] <= hfss._start_batch_machine[
                                 position + 1, machine]), name=f'{scenario}_batch_after_service[{position},{machine}]')

                # check if there is a shut down and warm up for stage
                if parameters.input_data[parameters.config.data.schedulingPar][stage][
                    parameters.config.data.scheduling_parameters.shutDown]:
                    for position in hfss._set_position_by_machine[machine]:
                        hfss.addConstr(
                            (hfss._end_batch_machine[position, machine] <= hfss._startTime_shutDown[position, machine]),
                            name=f'{scenario}_start_time_shutdown[{position},{machine}]')

                        # schedule mr block after shut down

                        # determine if repair starts in batch i
                        if not hasattr(self, "forwardScheduleTime"):
                            hfss.addConstr(
                                (hfss._startTime_shutDown[position, machine] +
                                 parameters.time_for_state_transition[(dv_name.cool_down, machine)] *
                                 hfss._shut_down_after_batch[position, machine] <= gp.quicksum(
                                            slot * hfss._start_repair_batch_machine_slot[position, machine, slot] for
                                            slot in hfss._set_slots_per_positionNmachine_end[(position, machine)])),
                                name=f'{scenario}_start_time_block[{position},{machine}]')

                            # schedule warmup after repair block
                            hfss.addConstr(
                                (gp.quicksum(
                                    slot * hfss._start_repair_batch_machine_slot[position, machine, slot] for slot in
                                    hfss._set_slots_per_positionNmachine_end[(position, machine)]) +
                                 hfss._duration_repair_after_batch[position, machine] <=
                                 hfss._startTime_warmUp[position, machine]),
                                name=f'{scenario}_start_time_warmup[{position},{machine}]')

                            for shift in hfss._set_shifts.keys():
                                # repair cannot be delayed into shift
                                if not parameters.slotRange_per_batchNmachine[(position + 1, machine)][1] <= \
                                       hfss._set_shifts[shift][0]:
                                    # repair cannot start as early as shift ends
                                    if not hfss._set_shifts[shift][1] <= \
                                           parameters.slotRange_per_batchNmachine[(position + 1, machine)][0]:
                                        hfss.addConstr((gp.quicksum(
                                            hfss._start_repair_batch_machine_slot[position, machine, slot] for slot in
                                            range(max(hfss._set_shifts[shift][0],
                                                      parameters.slotRange_per_batchNmachine[(position + 1, machine)][
                                                          0]),
                                                  min(hfss._set_shifts[shift][1] + 1,
                                                      parameters.slotRange_per_batchNmachine[(position + 1, machine)][
                                                          1]))) +
                                                        gp.quicksum(
                                                            hfss._repair_batch_machine[repair, position, machine] for
                                                            (repair, _) in
                                                            hfss._set_repair_machine.select('*', machine)) <= 1 +
                                                        hfss._repair_batch_shift[
                                                            (position, machine, shift)]),
                                                       name=f'{scenario}_determine_if_repair_in_shift[{position},{machine},{shift}]')
                                        hfss.addConstr((gp.quicksum(
                                            hfss._start_repair_batch_machine_slot[position, machine, slot] for slot in
                                            range(max(hfss._set_shifts[shift][0],
                                                      parameters.slotRange_per_batchNmachine[(position + 1, machine)][
                                                          0]),
                                                  min(hfss._set_shifts[shift][1] + 1,
                                                      parameters.slotRange_per_batchNmachine[(position + 1, machine)][
                                                          1])))
                                                        >=
                                                        hfss._repair_batch_shift[
                                                            (position, machine, shift)]),
                                                       name=f'{scenario}_test[{position},{machine},{shift}]')

                        else:

                            if position < hfss._postions_per_machine[machine] - 1:
                                upper_limit_slot = parameters.slotRange_per_batchNmachine[(position + 1, machine)][
                                                       1] - self.backwardScheduleTime[(position + 1, machine)]
                            else:
                                upper_limit_slot = parameters.max_number_slots

                            hfss.addConstr(
                                (hfss._startTime_shutDown[position, machine] +
                                 parameters.time_for_state_transition[(dv_name.cool_down, machine)] *
                                 hfss._shut_down_after_batch[position, machine] <= gp.quicksum(
                                            slot * hfss._start_repair_batch_machine_slot[position, machine, slot] for
                                            slot in
                                            range(parameters.slotRange_per_batchNmachine[(position, machine)][0] +
                                                  self.forwardScheduleTime[(position, machine)],
                                                  upper_limit_slot + 1))),
                                name=f'{scenario}_start_time_block[{position},{machine}]')

                            # schedule warmup after repair block
                            hfss.addConstr(
                                (gp.quicksum(
                                    slot * hfss._start_repair_batch_machine_slot[position, machine, slot] for slot in
                                    range(parameters.slotRange_per_batchNmachine[(position, machine)][0] +
                                          self.forwardScheduleTime[(position, machine)], upper_limit_slot + 1)) +
                                 hfss._duration_repair_after_batch[position, machine] <=
                                 hfss._startTime_warmUp[position, machine]),
                                name=f'{scenario}_start_time_warmup[{position},{machine}]')

                            for shift in hfss._set_shifts.keys():

                                if position < hfss._postions_per_machine[machine] - 1:
                                    upper_limit_slot = parameters.slotRange_per_batchNmachine[(position + 1, machine)][
                                                           1] - self.backwardScheduleTime[(position + 1, machine)]
                                else:
                                    upper_limit_slot = parameters.max_number_slots
                                # repair cannot be delayed into shift
                                if not upper_limit_slot <= hfss._set_shifts[shift][0]:
                                    # repair cannot start as early as shift ends
                                    if not hfss._set_shifts[shift][1] <= \
                                           parameters.slotRange_per_batchNmachine[(position, machine)][0] + \
                                           self.forwardScheduleTime[(position, machine)]:
                                        hfss.addConstr((gp.quicksum(
                                            hfss._start_repair_batch_machine_slot[position, machine, slot] for slot in
                                            range(max(hfss._set_shifts[shift][0],
                                                      parameters.slotRange_per_batchNmachine[(position, machine)][0] +
                                                      self.forwardScheduleTime[(position, machine)]),
                                                  min(hfss._set_shifts[shift][1] + 1, upper_limit_slot))) +
                                                        gp.quicksum(
                                                            hfss._repair_batch_machine[repair, position, machine] for
                                                            (repair, _) in
                                                            hfss._set_repair_machine.select('*', machine)) <= 1 +
                                                        hfss._repair_batch_shift[
                                                            (position, machine, shift)]),
                                                       name=f'{scenario}_determine_if_repair_in_shift[{position},{machine},{shift}]')
                # shut down is not required
                else:
                    for position in hfss._set_position_by_machine[machine]:

                        # determine if repair starts in batch i
                        if not hasattr(self, "forwardScheduleTime"):

                            for shift in hfss._set_shifts.keys():
                                # repair cannot be delayed into shift
                                if not parameters.slotRange_per_batchNmachine[(position + 1, machine)][1] <= \
                                       hfss._set_shifts[shift][0]:
                                    # repair cannot start as early as shift ends, only use the time interval within the respective shift to sum slot variables
                                    if not hfss._set_shifts[shift][1] < \
                                           parameters.slotRange_per_batchNmachine[(position + 1, machine)][0]:
                                        hfss.addConstr((gp.quicksum(
                                            hfss._start_repair_batch_machine_slot[position, machine, slot] for slot in
                                            range(max(hfss._set_shifts[shift][0],
                                                      parameters.slotRange_per_batchNmachine[(position + 1, machine)][
                                                          0]),
                                                  min(hfss._set_shifts[shift][1] + 1,
                                                      parameters.slotRange_per_batchNmachine[(position + 1, machine)][
                                                          1]))) +
                                                        gp.quicksum(
                                                            hfss._repair_batch_machine[repair, position, machine] for
                                                            (repair, _) in
                                                            hfss._set_repair_machine.select('*', machine)) <= 1 +
                                                        hfss._repair_batch_shift[
                                                            (position, machine, shift)]),
                                                       name=f'{scenario}_determine_if_repair_in_shift[{position},{machine},{shift}]')

                            hfss.addConstr(
                                (hfss._startTime_shutDown[position, machine] +
                                 parameters.time_for_state_transition[(dv_name.cool_down, machine)] *
                                 hfss._shut_down_after_batch[position, machine] <= gp.quicksum(
                                            slot * hfss._start_repair_batch_machine_slot[position, machine, slot] for
                                            slot in hfss._set_slots_per_positionNmachine_end[(position, machine)])),
                                name=f'{scenario}_start_time_block[{position},{machine}]')

                            # schedule warmup after repair block
                            hfss.addConstr(
                                (gp.quicksum(
                                    slot * hfss._start_repair_batch_machine_slot[position, machine, slot] for slot in
                                    hfss._set_slots_per_positionNmachine_end[(position, machine)]) +
                                 hfss._duration_repair_after_batch[position, machine] <=
                                 hfss._startTime_warmUp[position, machine]),
                                name=f'{scenario}_start_time_warmup[{position},{machine}]')


                        else:

                            for shift in hfss._set_shifts.keys():
                                # identify upper limit for repair slots
                                if position < hfss._postions_per_machine[machine] - 1:
                                    upper_limit_slot = parameters.slotRange_per_batchNmachine[(position + 1, machine)][
                                                           1] - self.backwardScheduleTime[(position + 1, machine)]
                                else:
                                    upper_limit_slot = parameters.max_number_slots
                                # repair cannot be delayed into shift
                                if not upper_limit_slot <= hfss._set_shifts[shift][0]:
                                    # repair cannot start as early as shift ends, only use the time interval within the respective shift to sum slot variables
                                    if not hfss._set_shifts[shift][1] <= \
                                           parameters.slotRange_per_batchNmachine[(position, machine)][0] + \
                                           self.forwardScheduleTime[(position, machine)]:
                                        hfss.addConstr((gp.quicksum(
                                            hfss._start_repair_batch_machine_slot[position, machine, slot] for slot in
                                            range(max(hfss._set_shifts[shift][0],
                                                      parameters.slotRange_per_batchNmachine[(position, machine)][0] +
                                                      self.forwardScheduleTime[(position, machine)]),
                                                  min(hfss._set_shifts[shift][1] + 1, upper_limit_slot))) +
                                                        gp.quicksum(
                                                            hfss._repair_batch_machine[repair, position, machine] for
                                                            (repair, _) in
                                                            hfss._set_repair_machine.select('*', machine)) <= 1 +
                                                        hfss._repair_batch_shift[
                                                            (position, machine, shift)]),
                                                       name=f'{scenario}_determine_if_repair_in_shift[{position},{machine},{shift}]')

                            if position < hfss._postions_per_machine[machine] - 1:
                                upper_limit_slot = parameters.slotRange_per_batchNmachine[(position + 1, machine)][
                                                       1] - self.backwardScheduleTime[(position + 1, machine)]
                            else:
                                upper_limit_slot = parameters.max_number_slots

                            hfss.addConstr(
                                (hfss._startTime_shutDown[position, machine] +
                                 parameters.time_for_state_transition[(dv_name.cool_down, machine)] *
                                 hfss._shut_down_after_batch[position, machine] <= gp.quicksum(
                                            slot * hfss._start_repair_batch_machine_slot[position, machine, slot] for
                                            slot in
                                            range(parameters.slotRange_per_batchNmachine[(position, machine)][0] +
                                                  self.forwardScheduleTime[(position, machine)],
                                                  upper_limit_slot + 1))),
                                name=f'{scenario}_start_time_block[{position},{machine}]')

                            # schedule warmup after repair block
                            hfss.addConstr(
                                (gp.quicksum(
                                    slot * hfss._start_repair_batch_machine_slot[position, machine, slot] for slot in
                                    range(parameters.slotRange_per_batchNmachine[(position, machine)][0] +
                                          self.forwardScheduleTime[(position, machine)], upper_limit_slot + 1)) +
                                 hfss._duration_repair_after_batch[position, machine] <=
                                 hfss._startTime_warmUp[position, machine]),
                                name=f'{scenario}_start_time_warmup[{position},{machine}]')

                        # schedule warmup after repair block
                        if position < hfss._postions_per_machine[machine] - 1:
                            hfss.addConstr(
                                (gp.quicksum(
                                    slot * hfss._start_repair_batch_machine_slot[position, machine, slot] for slot in
                                    hfss._set_slots_per_positionNmachine_end[(position, machine)]) +
                                 hfss._duration_repair_after_batch[position, machine] +
                                 hfss._duration_service_after_batch[position, machine] <=
                                 hfss._start_batch_machine[
                                     position + 1, machine]),
                                name=f'{scenario}_batch_after_service[{position},{machine}]')

                for position in hfss._set_position_by_machine[machine]:
                    # add constraints for speed up
                    # hfss.addConstr(hfss._start_batch_machine_slot.sum(position, machine, '*') == 1 )

                    hfss.addConstr(
                        hfss._start_repair_batch_machine_slot.sum(position, machine, '*') == 1,
                        name=f'{scenario}_repairBatchSlot[{position},{machine}]')



    def change_constraint_to_scenario(self, model: gp.Model, scenario, config):
        '''
        Function used to adopt model copy to scenario.
        :return:
        '''

        # maintenance reset big M for time tracking with chgCoeff build in gurobi function
        for (maintenance, batch, machine) in model._constraint_maintenance_time_tracking.keys():
            new_Coeff = model._parameter_maintenance[(maintenance, machine)][config.data.maintenance_name.upper_limit]
            constraint_toBeChanged = model._constraint_maintenance_time_tracking[(maintenance, batch, machine)]
            var_toBeChanged = model._maintenance_batch_machine[maintenance, batch, machine]
            model.chgCoeff(constraint_toBeChanged, var_toBeChanged, new_Coeff)

        for (maintenance, batch, machine) in model._constraint_reset_maintenanceTime.keys():
            new_Coeff = model._parameter_maintenance[(maintenance, machine)][config.data.maintenance_name.upper_limit]
            constraint_toBeChanged = model._constraint_reset_maintenanceTime[(maintenance, batch, machine)]
            var_toBeChanged = model._maintenance_batch_machine[maintenance, batch, machine]
            model.chgCoeff(constraint_toBeChanged, var_toBeChanged, new_Coeff)
            model._constraint_reset_maintenanceTime[(maintenance, batch, machine)].RHS = new_Coeff

        for (repair, batch, machine) in model._constraint_repair_time_tracking.keys():
            new_Coeff = model._parameter_repair[(repair, machine)][config.data.repair_name.upper_limit][scenario]
            constraint_toBeChanged = model._constraint_repair_time_tracking[(repair, batch, machine)]
            var_toBeChanged = model._repair_batch_machine[repair, batch, machine]
            model.chgCoeff(constraint_toBeChanged, var_toBeChanged, new_Coeff)

        for (repair, batch, machine) in model._constraint_reset_repairTime.keys():
            new_Coeff = model._parameter_repair[(repair, machine)][config.data.repair_name.upper_limit][scenario]
            constraint_toBeChanged = model._constraint_reset_repairTime[(repair, batch, machine)]
            var_toBeChanged = model._repair_batch_machine[repair, batch, machine]
            model.chgCoeff(constraint_toBeChanged, var_toBeChanged, new_Coeff)
            model._constraint_reset_repairTime[(repair, batch, machine)].RHS = new_Coeff

    def determine_CrossDependencyConstraints_to_remove(self, model):
        constraint_set_cross_dependency = []
        for index, pattern in model._generated_pattern.items():
            for batch_prev, machine_prev, batch, machine in model._batch_dependency[index].keys():
                # if batch_dependency[index][prev_node, node] is equal to 1, then constraint needs to hold
                if model._batch_dependency[index][batch_prev, machine_prev, batch, machine] >= 1:
                    constraint_set_cross_dependency.append((batch_prev, machine_prev, batch, machine))
        remove_constraints = []
        for constraint_key in model._batch_dependency[index].keys():
            if constraint_key not in constraint_set_cross_dependency:
                if constraint_key[3] != 'end' and constraint_key[1] != 'start':
                    remove_constraints.append(constraint_key)

        return remove_constraints
