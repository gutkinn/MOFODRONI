# MOFODRONI

This repository summarizes open-access code and results for the ESA-funded project MOFODRONI 'Monitoring and forecasting agricultural drought for rainfed rice in Nigeria, using multi-source data.' (ESA Contract No. 4000133905/21/I-EF). The code in this repository also requires external data inputs which can be found in this public Zenodo dataset (https://zenodo.org/records/19593937). The work conducte during this project is accessible via five main files:

1. *scripts/MOFODRONI_VICI.ipynb*:
   - this notebook goes through the process of extracting NDVI and SWI data from original versions, then calculating anomaly values. The notebook code only calculates anomaly values for a single combination of zones and upper threshold percentages, however all other combinations can be found in the Zenodo repository under the 'analysis' and 'analysis_swia' folders
  
2. *scripts/run_anomaly_calculations.py*:
   - this python file is called to calculated anomalmy values for both NDVIA and SWIA, based on user-defined zone numbers and upper threshold values
  
3. *scripts/run_comparison_seasonal.py*:
   - this python file is used for comparing multivariate regression model results for models fit to different combinations of input variables (such as farmer surveys, meteorological variables, and anomaly variables). Anomaly variables statistics in this file are calculated on an entire growing season. Configuration variables for this file can be found in the _scripts/config.ini_ file.
  
4. *scripts/run_comparison_phenological.py*:
   - this python file is used for comparing multivariate regression model results for models fit to different combinations of input variables (such as farmer surveys, meteorological variables, and anomaly variables). Anomaly variables statistics in this file are calculated on two specific phenological stages of the rice growing season (vegetative and maturity). Configuration variables for this file can be found in the _scripts/config.ini_ file.

5. *scripts/MOFODRONI_visualize*:
   - this notebook visualizes some of the results included in the publicaiton submission, especially those comparing different model values and anomaly value spatial distributions.

Note: the terms 'VICI' and 'NDVIA' are used interchangeably in this repository and code, because of links between the original inspiration for the project and the submission of project results for publication. 
