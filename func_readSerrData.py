# -*- coding: utf-8 -*-
"""
Scripts to read in data files of from Processed Serrano data
Created on 2021-03-05
@author: Rebecca
"""

# %% Read in data files of ERV_[date].dat


def read_ERV(filename, dat_headers, main_dir, data_dir):

    import pandas as pd
     
    

    
    
    
    

    #import csv file with and assign headers from dat_headers df
    
    Serr_ERV = pd.read_table(data_dir.joinpath(filename), 
                                 header = None, 
                                 names = dat_headers['ERV'].dropna(),  
                                 sep = '\s+',
                                 comment="#")
    
    
    
    
    return Serr_ERV


# %% Read in data files of Diurnal_[date].dat


def read_Diurnal(filename, dat_headers, main_dir, data_dir):

    import pandas as pd
     
    

    
    
    
    

    #import csv file with and assign headers from dat_headers df
    
    Serr_Diurnal = pd.read_table(data_dir.joinpath(filename), 
                                 header = None, 
                                 names = dat_headers['Diurnal'].dropna(),  
                                 sep = '\s+',
                                 comment="#")
    
    
    
    
    return Serr_Diurnal

# %% Read in data files of GroundT_[date].dat

def read_GroundT(filename, dat_headers, main_dir, data_dir):

    import pandas as pd

    
    
    
    

    #import csv file with and assign headers from dat_headers df
    Serr_GroundT = pd.read_table(data_dir.joinpath(filename), 
                                 header = None, 
                                 names = dat_headers['GroundT'].dropna(),    
                                 sep = '\s+',
                                 comment="#")

    
    
    
    return Serr_GroundT



# %% Read in data files of Indoors_[date].dat

def read_indoors(filename, dat_headers, main_dir, data_dir):

    import pandas as pd
     
    

    
    
    
    

    #import csv file with and assign headers from dat_headers df
    Serr_indoors = pd.read_table(data_dir.joinpath(filename), 
                               header = None, 
                               names = dat_headers['Indoors'].dropna(),  
                               sep = '\s+',
                               comment="#")

    
    
    
    return Serr_indoors


# %% Read in data files of Loops_deltaTs_[date].dat

def read_loops_deltaTs(filename, dat_headers, main_dir, data_dir):

    import pandas as pd
     
    

    
    
    
    

    #import csv file with and assign headers from dat_headers df
    Serr_loops_deltaTs = pd.read_table(data_dir.joinpath(filename), 
                                 header = None, 
                                 names = dat_headers['Loops_deltaTs'].dropna(),    
                                 sep = '\s+',
                                 comment="#")

    
    
    
    return Serr_loops_deltaTs


# %% Read in data files of Loops_flow_rates_[date].dat

def read_loops_flow_rates(filename, dat_headers, main_dir, data_dir):

    import pandas as pd
     
    

    
    
    
    

    #import csv file with and assign headers from dat_headers df
    Serr_loops_flow_rates = pd.read_table(data_dir.joinpath(filename), 
                                 header = None, 
                                 names = dat_headers['Loops_flow_rates'].dropna(),  
                                 sep = '\s+',
                                 comment="#")

    
    
    
    return Serr_loops_flow_rates


# %% Read in data files of Loops_heat_transfer_[date].dat

def read_loops_heat_transfer(filename, dat_headers, main_dir, data_dir):

    import pandas as pd
     
    

    
    
    
    

    #import csv file with and assign headers from dat_headers df
    col_headers = dat_headers['Loops_heat_transfer'].dropna()
    Serr_loops_heat_transfer = pd.read_table(data_dir.joinpath(filename), 
                                 header = None, 
                                 names = col_headers,
                                 sep = '\s+',
                                 comment="#")

    
    
    
    return Serr_loops_heat_transfer

# %% Read in data files of Loops_temperatures_[date].dat

def read_loops_temperatures(filename, dat_headers, main_dir, data_dir):

    import pandas as pd
     
    

    
    
    
    

    #import csv file with and assign headers from dat_headers df
    Serr_loops_temperatures = pd.read_table(data_dir.joinpath(filename), 
                                 header = None, 
                                 names = dat_headers['Loops_temperatures'].dropna(),    
                                 sep = '\s+',
                                 comment="#")

    
    
    
    return Serr_loops_temperatures


# %%Read in data files of Performance_metrics_data_[date].dat

def read_performance_metrics_data(filename, dat_headers, main_dir, data_dir):

    import pandas as pd
     
    

    
    
    
    

    #import csv file with and assign headers from dat_headers df
    Serr_performance_metrics_data = pd.read_table(data_dir.joinpath(filename), 
                                 header = None, 
                                 names = dat_headers['Performance_metrics_data'].dropna(),    
                                 sep = '\s+',
                                 comment="#")

    
    
    
    return Serr_performance_metrics_data



# %% Read in data files of Sand_temperatures_[date].dat

def read_sand_temps(filename, dat_headers, main_dir, data_dir):

    import pandas as pd
     
    

    
    
    
    

    #import csv file with TCs and assign headers from dat_headers df
    Serr_sand_temps = pd.read_table(data_dir.joinpath(filename), 
                                 header = None, 
                                 names = dat_headers['Sand_temperatures'].dropna(),    
                                 sep = '\s+',
                                 comment="#")

    
    
    
    return Serr_sand_temps
            


# %% Read in data files of Weather_[date].dat

def read_Weather(filename, dat_headers, main_dir, data_dir):

    import pandas as pd 
    

    
    
    
    

    #import csv file with and assign headers from dat_headers df
    Serr_Weather = pd.read_table(data_dir.joinpath(filename), 
                                 header = None, 
                                 names = dat_headers['Weather'].dropna(),    
                                 sep = '\s+',
                                 comment="#")

    
    
    
    return Serr_Weather