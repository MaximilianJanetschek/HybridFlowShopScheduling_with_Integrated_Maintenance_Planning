def define_number_of_positions(parameter):
    """
    This functions determines the slots per machine. Because the considered production line has identical machine and identical jobs the same process time can be used to identify the maximum number of position. Additionally, a set of safety slots is added.
    :param parameter:
    :return:
    """
    process_steps = parameter.set_stages
    machines_per_stage = parameter.machines_per_stage
    jobs_per_batch = parameter.jobs_per_batch_by_machine
    max_jobs = parameter.max_jobs
    safety_slots = parameter.config.model_discrete.parameters.add_slots

    import numpy as np
    positions_per_machine = {}
    for stage in process_steps:
        number_machines_per_stage = len(machines_per_stage[stage])
        for machine in machines_per_stage[stage]:
            number_jobs_per_batch = jobs_per_batch[machine]
            positions_per_machine[machine] = np.intc(np.ceil((max_jobs / (number_machines_per_stage *  number_jobs_per_batch)) + safety_slots))
    return positions_per_machine