import os
import pandas as pd
import numpy as np
import gurobipy as gp


def load_Data_from_excel(name_table, config):
    """
    Loads data from input excel. Each sheet defines a single stage in the production process. Each sheet in turn,
    consists of the columns Machine ID, required time in hours for processing one job, available machine time in hours
    before next cleaning, time_for_cleaning in hours which defines the time it takes to clean the respective machine.

    :return: dictionary in format stage: parameter_name: machine_ID: value.
    """
    time_conversion = 60 / 15
    convert_to_time_buckets = False
    if name_table == 'discrete':
        convert_to_time_buckets = True

    # Get defined sequence of stages
    production_layout = config.process_layout.stages

    # Check and convert input data to avoid erros, process_time: float32, batches_between_cleaning: float32, time_for_cleaning: float32
    column_data_type = {}
    column_data_type[config.data.parameter_name.process_time] = np.single

    # Define file path and load excel as dataframe with pandas module.
    fileName = os.path.join(os.getcwd(), config.data.path_excel_discrete)
    input_excel = pd.read_excel(fileName, sheet_name=production_layout,dtype=column_data_type, index_col=0)

    # convert process time to time buckets
    if convert_to_time_buckets:
        keys_to_be_conferted = []
        keys_to_be_conferted.append(config.data.parameter_name.duration.warmup)
        keys_to_be_conferted.append(config.data.parameter_name.duration.cooldown)
        for stage, df_stage in input_excel.items():
            for transformKey in keys_to_be_conferted:
                df_stage[transformKey] = np.ceil(df_stage[transformKey] * 60 / 15)

    # Convert dataframe to dictionary.
    process_parameters = {stage: df_table.to_dict() for stage, df_table in input_excel.items()}

    # Read in Maintenance tasks
    column_data_type = {}
    column_data_type[config.data.maintenance_name.shutdown] = np.bool_
    input_maintenance = pd.read_excel(fileName, sheet_name=config.data.maintenance_name.sheet_name,dtype=column_data_type)
    if convert_to_time_buckets:
        for scenario, column_name in config.data.maintenance_name.scenarios.items():
            input_maintenance[scenario] = np.ceil(input_maintenance[column_name] * time_conversion)
        input_maintenance[config.data.maintenance_name.duration] = np.ceil(input_maintenance[config.data.maintenance_name.duration] * time_conversion)
        input_maintenance[config.data.maintenance_name.upper_limit] = np.ceil(input_maintenance[config.data.maintenance_name.upper_limit] * time_conversion)

    # Convert dataframe to dictionary
    # goal format {machine: { task: { duration: .., max: ...}}
    maintenance_parameters = input_maintenance.to_dict('records')

    # convert into time buckets

    # Read in Repair Scheduling Cost
    # Read in Maintenance tasks
    column_data_type = {}
    schedulingSheet = config.data.scheduling_cost
    column_data_type[schedulingSheet.cost] = np.single

    input_scheudlingCost = pd.read_excel(fileName, sheet_name=schedulingSheet.sheetName,
                                      dtype=column_data_type)

    input_scheudlingCost= input_scheudlingCost.to_dict('records')
    scheduling_parameter = gp.tupledict()
    rescheduling_parameter = gp.tupledict()
    for schedulingCost in input_scheudlingCost:
        scheduling_parameter[(schedulingCost[schedulingSheet.category], schedulingCost[schedulingSheet.stage])] = schedulingCost[schedulingSheet.cost]
        rescheduling_parameter[(schedulingCost[schedulingSheet.category], schedulingCost[schedulingSheet.stage])] = \
        schedulingCost[schedulingSheet.re_cost]

    # Read in Scheduling Attributes
    schedulingSheet = config.data.scheduling_parameters.sheetName
    column_data_type = {}
    input_generalSchedulingParameters= pd.read_excel(fileName, sheet_name=schedulingSheet, dtype=column_data_type, index_col=0)
    generalSchedulingParameters = input_generalSchedulingParameters.to_dict('index')
    print(generalSchedulingParameters)

    # determine time_slot shift assignment, format {(shift,day): (lower_slot, upper_slot)}
    shift_dict = gp.tupledict()
    if convert_to_time_buckets:
        for day in range(0,config.model_discrete.set.time_slots.horizon_in_days):
            for shift in range(0,config.model_discrete.set.time_slots.shifts_per_day):
                lower_slot = (day * 24*60/15) + shift *  config.model_discrete.set.time_slots.hours_per_shift * time_conversion
                upper_slot = lower_slot + config.model_discrete.set.time_slots.hours_per_shift * time_conversion -1
                shift_dict[shift + (day*config.model_discrete.set.time_slots.shifts_per_day)] = (int(lower_slot), int(upper_slot), int(shift), int(day))

    # load general energy  cost
    parameterSheet = config.data.parameters.sheetName
    column_data_type = {}
    input_generalParameters = pd.read_excel(fileName, sheet_name=parameterSheet, dtype=column_data_type, index_col=0)
    generalParameter_dict = input_generalParameters.to_dict('index')


    parameters = {config.data.process: process_parameters, config.data.maintenance: maintenance_parameters, config.data.scheduling: scheduling_parameter, config.data.rescheduling: rescheduling_parameter, config.data.generalPar: generalParameter_dict, config.data.schedulingPar: generalSchedulingParameters}

    return parameters, production_layout, shift_dict

