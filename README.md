# Hybrid Flow Shop Scheduling with Integrated Maintenance Planning

Over-supply in markets, such as in the case of battery cells, increases cost pressure on manufacturers. Energy
consumption and maintenance planning allows for significant cost reductions due to their significant share. Therefore,
an energy-conscious scheduling model for integrated maintenance planning is needed. Extending the common
Hybrid Flow Shop Scheduling (HFSS) production layout with the corresponding combination is not studied before. Thus, this thesis proposes a new
Mixed-Integer Linear Programm formulation. Column generation, formulation tightening, and improvements on the B\&B bound lower the
required solution time. Despite deploying exact optimisation, the solution procedure solves a three-week scheduling
problem for a reference line on battery electrodes in less than 500 seconds. The investigated application for a rolling
horizon and rescheduling context underlines practical applicability. The results motivate real-world implementation and
further developments.

# Technical Setup

Model settings can be adjusted in the configuration file, i.e. "./config/config.yml". 
Production parameters, such as stage layout, process time, maintenance task, and energy consumption, can be changed in the corresponding Excel-File. For each case study, a unique Excel File is used. 


# Requirements
The project was developed with pyhton 3.11. Package Requirements and corresponding version are listed in the requirements.txt file.
