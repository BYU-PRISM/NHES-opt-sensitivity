# NHES-opt-sensitivity
This project is focused on the optimization of nuclear hybrid energy systems (NHES). The following code is used to run an NHES dispatcher and perform a sensitivity analysis of the system variables.  The dispatcher, code, and data were used in a study titled "Sensitivity analysis and uncertainty quantification of combined design and dispatch optimization for a nuclear hybrid energy system" which can be found here: _____________. 

The code for the dispatcher can be found under the "simple_nhes.py" file.  

Synthetic time series were created to represent electrical load, solar gneration, and wind generation.  "signals.py" analyzes these time series and provides basic characteristics and statistsics. It also provides characteristic profiles/signals from the timeseries.

The output data from the runs can be found on FigShare, the "get_data.py" will download these files for the user.

The various functions used to perform the sensitivity analysis can be found under the file "analyze.py".

Code for creating plots of the data and sensitivity analysis results can be found under "plotting.py".
