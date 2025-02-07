# SSTES_FDM
Finite difference model of the sand storage at the Urbandale Centre for Home Energy Research

To use the code, download the three script files and create a directory structure where your root directory has a folder called "Exp_Data_for_Models" and another for the script files. The experimental data that will be used as inputs to the FDM will be put into Exp_Data_for_Models, and can be downloaded from the Harvard Dataverse at this DOI: https://doi.org/10.7910/DVN/5KR8IL. 

The script Master_v5_4.py is the script that needs to be run first - it has the numerical model that will generate the solution output files. 
The script Master_process_simu_data.py should be run second - it will process the output files and visualize the solution data.
The script func_readSerrData.py is an accessory script, containing functions that are pulled by the other file(s).

**Update**: A User had indicated that when they downloaded the files from model and tried to run it, they were missing a file called "Serrano_dat_headers.csv". I have now added that file to the repository.
