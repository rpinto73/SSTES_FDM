# -*- coding: utf-8 -*-
"""
Created on Tue May 19 16:33:00 2020
@author: Rebecca

 This is a Master_v5_3.py script based on SVN rev 897. 
 Changes from SVN rev 897: 
     -I have removed the module matplotlib 
     -I changed the double backslash \\ that denotes directories in Windows with the single forward slash / used in Linux.
     -I removed the math.remainder and replaced it with the modulus operator (%) since it wasn't introduced until Python3.7'

by: Rebecca PINTO
2020-01-15"""
# %% Import Packages
import pandas as pd
import numpy as np
import re
import os
import math
from datetime import date
from decimal import Decimal
from pathlib import Path

#import my functions from other Python files in the same folder
#import func_readSerrData as func_serr

#TO DEBUG, insert "breakpoint()" where you want the simulation to stop

#%% 0. GET RUN NUMBER
run_num = int(input("Enter the run number: "))
# run_num = 999

#%% 00. USER DECISIONS 
# rewind = 0.25 #corresponds to 720 iterations at dt=300s

# tolerance for steady-state IC while loop, IC_SS function
tolerance = 0.01 #C

#%%% Is the north wall BC different?
# north_BC_diff = int(input("Is north wall boundary at an elevated temperature (1=Yes/0=No): "))
north_BC_diff=0
#%%% Should all nodes in concrete and XPS layers be output for visualization (1=Yes/0=No)?
all_wall_layer_nodes = 1

#%% FUNCTIONS from func_readSerrData

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

def read_sand_temps(filename, dat_headers, main_dir, data_dir):

    import pandas as pd
    
    

    #import csv file with TCs and assign headers from dat_headers df
    Serr_sand_temps = pd.read_table(data_dir.joinpath(filename), 
                                 header = None, 
                                 names = dat_headers['Sand_temperatures'].dropna(),    
                                 sep = '\s+',
                                 comment="#")

    
    return Serr_sand_temps
            
#%% FUNCTIONS
def remove_newline_in_string(string):
    #remove the \n somehow inserted into the vectors when str() is used
    regex = re.compile(r'\n')
    string_out = regex.sub("",string)
    return string_out

def check_mod(rem,divisor):
    rem = round(rem,3)
    rem = float(rem)

    if rem == divisor:
        #if the remainder is equal to the divisor, then the remainder should really 
        #be zero, and it's a particularity of floating point arithmetic and binary numbers
        #and the modulus operator that have made it not equal to zero
        rem = 0
        
    return rem

def boundaryConditions(startday,day,node_zcoords,domain_H, T_Nbound, T_SEWbound):
    global T_avg, ampl, T_avg_N, ampl_N, north_BC_diff
    
    DOY = startday + day
    
    #if in higher years
    if DOY > 365*4:
        #Translate to day in year 0 (initial year)
        DOY = DOY-365*4
    elif DOY > 365*3:
        DOY = DOY-365*3
    elif DOY > 365*2:
        DOY = DOY-365*2
    elif DOY > 365:
        DOY = DOY-365
    
    #Boundary temperatures of far-field conditions around sand store --> use Kusada eqn:
    #T_soil = T_avg_yearly_temp - ampl*exp(-depth_below_surface/damp_dpth)*cos(2*pi*DOY/365 - depth_below_surface/damp_dpth - phase)
    
    #Fill side wall matrices with boundary conditions
    for k in range(0,len(node_zcoords)):
        T_SEWbound[k] = T_avg - ampl*math.exp(-(domain_H-node_zcoords[k])/damp_dpth)*math.cos(2*math.pi*DOY/365 - (domain_H-node_zcoords[k])/damp_dpth - phase)
        
        #different parameters on north domain boundary
        if north_BC_diff == 1:
            T_Nbound[k] = T_avg_N - ampl_N*math.exp(-(domain_H-node_zcoords[k])/damp_dpth)*math.cos(2*math.pi*DOY/365 - (domain_H-node_zcoords[k])/damp_dpth - phase)
        else:
            T_Nbound[k] = T_SEWbound[k]
        
        
    return T_Nbound,T_SEWbound


def updateBCs(T_Nbound, T_SEWbound,timestep):
    global Temps,U,V,W
    #APPLY DIRICHLET BOUNDARY CONDITIONS
    for k in range(1,W-1):
        #NORTH B.C. applies:
        Temps[0,:,k,timestep] = T_Nbound[k]
        
        #SOUTH WALL B.C. applies:
        Temps[U-1,:,k,timestep] = T_SEWbound[k]
        
        #WEST WALL B.C. applies:
        Temps[:,0,k,timestep] = T_SEWbound[k]
        
        #EAST WALL B.C. applies:                 
        Temps[:,V-1,k,timestep] = T_SEWbound[k]
    
    
    #BOTTOM WALL B.C. applies:
    Temps[:,:,0,timestep] = T_SEWbound[0]
    #TOP WALL B.C. applies:
    Temps[:,:,W-1,timestep] = T_SEWbound[W-1]
    
    
def print2newfile(filename,day,hour,array2print,dim_X, dim_Y, dim_Z, viz = False):
   # global start_YYY, start_MM, start_DD, end_YYY, 
    #note: the brackets around the temparray allow the array to be treated as a list of only one list, instead of as a list of lists....so it prints out the delimiters between the numbers
    #https://stackoverflow.com/questions/42068144/numpy-savetxt-is-not-adding-comma-delimiter
    temparray = np.concatenate(([day], [hour], array2print.flatten(order='F')))
    with open(filename,'w') as f:
        #writes parameters to file - dimensions of array
        f.write("#output_array_dims " + str(dim_X) + "," + str(dim_Y) + "," + str(dim_Z)+"\n")
        #writes parameters to file - start and end dates
        f.write("#start_and_end_dates_of_simulation " + str(start_YYYY) + "-" + str(start_MM) + "-" + str(start_DD)+ "," + str(end_YYYY) + "-" + str(end_MM) + "-" + str(end_DD)+"\n")       
        f.write("#Number of days of simulation: " + str(numberofdays_simu) + "\n")
        if viz == True:
            #writes x-, y-, and z-coordinate vectors of visualized nodes to file
            f.write("#x-coords " + remove_newline_in_string(str(node_xcoords)) +"\n")  
            f.write("#x-viz indices " + remove_newline_in_string(str(VIZ_X_index)) +"\n")  
            f.write("#y-coords " + remove_newline_in_string(str(node_ycoords)) +"\n")
            f.write("#y-viz indices " + remove_newline_in_string(str(VIZ_Y_index)) +"\n") 
            f.write("#z-coords " + remove_newline_in_string(str(node_zcoords)) +"\n")
            f.write("#z-viz indices " + remove_newline_in_string(str(VIZ_Z_index)) +"\n")
            #writes x- and y-coordinate vectors of location of TC nodes to file
            f.write("#TC_xcoords " + remove_newline_in_string(str(TC_xcoords)) +"\n")
            f.write("#TC_ycoords " + remove_newline_in_string(str(TC_ycoords)) +"\n")
        #write out initial condition in the form of a flattened array
        np.savetxt(f,[temparray],newline=' ',  delimiter=',',fmt='%.6f')
    
    
def append2file(filename,day,hour,array2print):
    #note: the brackets around the temparray allow the array to be treated as a list of only one list, instead of as a list of lists....so it prints out the delimiters between the numbers
    #https://stackoverflow.com/questions/42068144/numpy-savetxt-is-not-adding-comma-delimiter
    temparray = np.concatenate(([day], [hour], array2print.flatten(order='F')))
    with open(filename,'a') as f:
        f.write("\n")
        np.savetxt(f,[temparray],newline=' ',  delimiter=',',fmt='%.6f')
        
def IC_SS(dt):
    global U,V,W, Temps,Lambda, rho, Cp, node_xcoords, node_ycoords, node_zcoords
    global pwr_bott_indx,x_pwr_low_indx, x_pwr_high_indx, y_pwr_low_indx, y_pwr_high_indx
    global pwr_mid_indx, pwr_upr_indx, nn, tolerance, rewind, diff, get_out

    import matplotlib.pyplot as plt
    #initialize
    prev=0
    diff=20
    
    fig,ax=plt.subplots()
    fig1,ax1=plt.subplots()
    fig2,ax2=plt.subplots()
    
    #while (rewind*10*24*3600/300 > nn):#rewinds of 10 days at dt=300s
    
    while (diff > tolerance):
        nn = nn+1
        print(nn,diff)
        
        # if math.remainder(nn,100) == 0:
        #     #save the graphs
        #     ax.set_title("After " + str(nn) + " iterations, max diff=" + str(diff))
        #     fig.savefig(main_dir.joinpath("After " + str(nn) + " iterations_x.png"), dpi=150, bbox_inches='tight')
        #     ax1.set_title("After " + str(nn) + " iterations")
        #     fig1.savefig(main_dir.joinpath("After " + str(nn) + " iterations_y.png"), dpi=150, bbox_inches='tight')
        #     ax2.set_title("After " + str(nn) + " iterations")
        #     fig2.savefig(main_dir.joinpath("After " + str(nn) + " iterations_z.png"), dpi=150, bbox_inches='tight')
            
            #initialize new figures for the next 1000 lines
        #     fig,ax=plt.subplots()
        #     fig1,ax1=plt.subplots()
        #     fig2,ax2=plt.subplots()
            
        # ax.plot(range(0,Temps[:,31,24,0].shape[0]),Temps[:,31,24,0])
        # ax1.plot(range(0,Temps[31,:,24,0].shape[0]),Temps[31,:,24,0])
        # ax2.plot(range(0,Temps[31,31,:,0].shape[0]),Temps[31,31,:,0])
       
        
        #We are not iterating over boundary walls or the sand store
        for i in range(1,U-1):
            for j in range(1,V-1): 
                for k in range(1,W-1):
                    
                    #cubic volume of the sand store, don't iterate over it
                    if (k > indx_bndrz_z[6] and k < indx_bndrz_z[7] and
                        i > indx_bndrz_x[6] and i < indx_bndrz_x[11] and
                        j > indx_bndrz_y[6] and j < indx_bndrz_y[11]):
                        
                        continue #skip the rest of the for loop
                    else:
                        #prev time step temp
                        old_T = Temps[i,j,k,prev]
                        
                        #GETTING TEMPS AROUND NODE FROM PREV TIMESTEP AND LAMBDAs, TO USE IN GOV EQN
                        North = Temps[i-1,j,k,prev]
                        South = Temps[i+1,j,k,prev]
                        West = Temps[i,j-1,k,prev]
                        East = Temps[i,j+1,k,prev]
                        Bottom = Temps[i,j,k-1,prev]
                        Top = Temps[i,j,k+1,prev]
                        
                        lmbda_S = Lambda[i+1,j,k]
                        lmbda_E = Lambda[i,j+1,k]
                        lmbda_T = Lambda[i,j,k+1]
                        
                        #APPLY GOVERNING EQUATION 
                        #variable substitution [for easier reading of GE]
                        #subscripts _ip1/_im1 mean "i plus 1, i minus 1"
                        g = dt/ (rho[i,j,k] * Cp[i,j,k])
                        lmbda = Lambda[i,j,k]
                        x_i = node_xcoords[i]
                        x_ip1 = node_xcoords[i+1]  #ip1 -> i plus one
                        x_im1 = node_xcoords[i-1]  #im1 -> i minus one
                        y_j = node_ycoords[j]
                        y_jp1 = node_ycoords[j+1]
                        y_jm1 = node_ycoords[j-1]
                        z_k = node_zcoords[k]
                        z_kp1 = node_zcoords[k+1]
                        z_km1 = node_zcoords[k-1]
                     
                        #else use GE without Q_source term
                        Temps[i,j,k,1] = old_T + g *( (lmbda_S-lmbda)*(South-old_T)/((x_ip1-x_i)**2) + 
                            (2*lmbda/(x_ip1-x_im1))*( ((South-old_T)/(x_ip1-x_i))-((North-old_T)/(x_im1-x_i)) ) + 
                            (lmbda_E-lmbda)*(East-old_T)/((y_jp1-y_j)**2) + 
                            (2*lmbda/(y_jp1-y_jm1))*( ((East-old_T)/(y_jp1-y_j))-((West-old_T)/(y_jm1-y_j)) ) + 
                            (lmbda_T-lmbda)*(Top-old_T)/((z_kp1-z_k)**2) + 
                            (2*lmbda/(z_kp1-z_km1))*( ((Top-old_T)/(z_kp1-z_k))-((Bottom-old_T)/(z_km1-z_k)) ) )
                        
       
        #take difference between the maximum temperatures in the new and old timesteps            
        diff = np.max(abs(Temps[:,:,:,1]-Temps[:,:,:,0]))
        
        if np.max(Temps)>1000 or np.min(Temps)<-1000 or np.isinf(Temps).sum()>0 or np.isnan(Temps).sum()>0:
            get_out=1
            print("Exited for loop - unstable FDM model")
            break
        
        if diff > 20:
            print("yes")
        if abs(diff) > 20:
            print("abs_yes")
        #put temps from just-calculated timestep into "previous" timestep
        Temps[:,:,:,0] = Temps[:,:,:,1]    
        
    return diff        
    
def GoverningEqn(dt,n):
    global U,V,W, Temps,Lambda, rho, Cp, node_xcoords, node_ycoords, node_zcoords, solar_to_Sand_W, Sand_to_SH_W
    global pwr_bott_indx,x_pwr_low_indx, x_pwr_high_indx, y_pwr_low_indx, y_pwr_high_indx
    global pwr_mid_indx, pwr_upr_indx
    #We are not iterating over boundary nodes, because their temperatures
    #are known
    prev = 0
    
    for i in range(1,U-1):
        for j in range(1,V-1):
            for k in range(1,W-1):
                #prev time step temp
                old_T = Temps[i,j,k,prev]
                
                #GETTING TEMPS AROUND NODE FROM PREV TIMESTEP AND LAMBDAs, TO USE IN GOV EQN
                North = Temps[i-1,j,k,prev]
                South = Temps[i+1,j,k,prev]
                West = Temps[i,j-1,k,prev]
                East = Temps[i,j+1,k,prev]
                Bottom = Temps[i,j,k-1,prev]
                Top = Temps[i,j,k+1,prev]
                
                lmbda_S = Lambda[i+1,j,k]
                lmbda_E = Lambda[i,j+1,k]
                lmbda_T = Lambda[i,j,k+1]
                
                #APPLY GOVERNING EQUATION 
                #variable substitution [for easier reading of GE]
                #subscripts _ip1/_im1 mean "i plus 1, i minus 1"
                g = dt/ (rho[i,j,k] * Cp[i,j,k])
                lmbda = Lambda[i,j,k]
                x_i = node_xcoords[i]
                x_ip1 = node_xcoords[i+1]  #ip1 -> i plus one
                x_im1 = node_xcoords[i-1]  #im1 -> i minus one
                y_j = node_ycoords[j]
                y_jp1 = node_ycoords[j+1]
                y_jm1 = node_ycoords[j-1]
                z_k = node_zcoords[k]
                z_kp1 = node_zcoords[k+1]
                z_km1 = node_zcoords[k-1]
                
               
                #if node being calculated is within the bottom power slab area
                if (k==pwr_bott_indx and (i >= x_pwr_low_indx and i <= x_pwr_high_indx) and (j >= y_pwr_low_indx and j <= y_pwr_high_indx)):
                    #use GE with Q_source term
                    Temps[i,j,k,1] = old_T + g *( solar_to_Sand_W[n-1,2] - Sand_to_SH_W[n-1,2] +
                        (lmbda_S-lmbda)*(South-old_T)/((x_ip1-x_i)**2) + 
                        (2*lmbda/(x_ip1-x_im1))*( ((South-old_T)/(x_ip1-x_i))-((North-old_T)/(x_im1-x_i)) ) + 
                        (lmbda_E-lmbda)*(East-old_T)/((y_jp1-y_j)**2) + 
                        (2*lmbda/(y_jp1-y_jm1))*( ((East-old_T)/(y_jp1-y_j))-((West-old_T)/(y_jm1-y_j)) ) + 
                        (lmbda_T-lmbda)*(Top-old_T)/((z_kp1-z_k)**2) + 
                        (2*lmbda/(z_kp1-z_km1))*( ((Top-old_T)/(z_kp1-z_k))-((Bottom-old_T)/(z_km1-z_k)) ) )
                    
                #if node being calculated is within the middle power slab area  
                elif (k==pwr_mid_indx and (i >= x_pwr_low_indx and i <= x_pwr_high_indx)
                        and (j >= y_pwr_low_indx and j <= y_pwr_high_indx)):
                    #use GE with Q_source term
                    Temps[i,j,k,1] = old_T + g *( solar_to_Sand_W[n-1,2] - Sand_to_SH_W[n-1,2] +
                        (lmbda_S-lmbda)*(South-old_T)/((x_ip1-x_i)**2) + 
                        (2*lmbda/(x_ip1-x_im1))*( ((South-old_T)/(x_ip1-x_i))-((North-old_T)/(x_im1-x_i)) ) + 
                        (lmbda_E-lmbda)*(East-old_T)/((y_jp1-y_j)**2) + 
                        (2*lmbda/(y_jp1-y_jm1))*( ((East-old_T)/(y_jp1-y_j))-((West-old_T)/(y_jm1-y_j)) ) + 
                        (lmbda_T-lmbda)*(Top-old_T)/((z_kp1-z_k)**2) + 
                        (2*lmbda/(z_kp1-z_km1))*( ((Top-old_T)/(z_kp1-z_k))-((Bottom-old_T)/(z_km1-z_k)) ) )
                    
                #if node being calculated is within the top power slab area
                elif (k==pwr_upr_indx and (i >= x_pwr_low_indx and i <= x_pwr_high_indx)
                        and (j >= y_pwr_low_indx and j <= y_pwr_high_indx)):
                    #use GE with Q_source term
                    Temps[i,j,k,1] = old_T + g *( solar_to_Sand_W[n-1,2] - Sand_to_SH_W[n-1,2] +
                        (lmbda_S-lmbda)*(South-old_T)/((x_ip1-x_i)**2) + 
                        (2*lmbda/(x_ip1-x_im1))*( ((South-old_T)/(x_ip1-x_i))-((North-old_T)/(x_im1-x_i)) ) + 
                        (lmbda_E-lmbda)*(East-old_T)/((y_jp1-y_j)**2) + 
                        (2*lmbda/(y_jp1-y_jm1))*( ((East-old_T)/(y_jp1-y_j))-((West-old_T)/(y_jm1-y_j)) ) + 
                        (lmbda_T-lmbda)*(Top-old_T)/((z_kp1-z_k)**2) + 
                        (2*lmbda/(z_kp1-z_km1))*( ((Top-old_T)/(z_kp1-z_k))-((Bottom-old_T)/(z_km1-z_k)) ) )
                else:
                    #else use GE without Q_source term
                    Temps[i,j,k,1] = old_T + g *( (lmbda_S-lmbda)*(South-old_T)/((x_ip1-x_i)**2) + 
                        (2*lmbda/(x_ip1-x_im1))*( ((South-old_T)/(x_ip1-x_i))-((North-old_T)/(x_im1-x_i)) ) + 
                        (lmbda_E-lmbda)*(East-old_T)/((y_jp1-y_j)**2) + 
                        (2*lmbda/(y_jp1-y_jm1))*( ((East-old_T)/(y_jp1-y_j))-((West-old_T)/(y_jm1-y_j)) ) + 
                        (lmbda_T-lmbda)*(Top-old_T)/((z_kp1-z_k)**2) + 
                        (2*lmbda/(z_kp1-z_km1))*( ((Top-old_T)/(z_kp1-z_k))-((Bottom-old_T)/(z_km1-z_k)) ) )
    
        
# def partialAvg_of_Temps_in_zDir(XYZarray,min_zIndx,max_zIndx):           
#     #This function calculates the average of a 3-D chunk of an array, from bottom z index, up to and including the top z-index. 
#     global indx_bndrz_z, moisture_layer_height
    
#     temp_sum = 0 
#     temp_avg = 0
#     for k in range(min_zIndx,max_zIndx + 1):
#         temp_sum = temp_sum + np.mean(XYZarray[:,:,k]) 
#     temp_avg = temp_sum/len(range(min_zIndx,max_zIndx + 1))
    
#     return temp_avg
    
# %% Define Directory Paths
#get current directory
main_dir = Path(os.getcwd())
#define parent directory
parent = main_dir.parents[0]

#define directory holding input data
data_dir = parent.joinpath('Exp_Data_for_models')
#define directory where graphs will be saved
graph_dir = main_dir.joinpath('Diff_runs')

#Read in Headers file
dat_headers = pd.read_csv(data_dir.joinpath('Serrano_dat_headers.csv'))


# %% 1. IMPORT EXPERIMENTAL DATA FILES
#FILENAMES OF DATA FILES
#(NB! filename1 MUST have a start AND end date, even if they are equal, ie. only one day of data)

#Excel sheet with different run descriptions
diff_runs_file = graph_dir.joinpath('2021-07-13 Description of different runs.xlsx')
diff_runs = pd.read_excel(diff_runs_file, header=1, index_col=0)

#check the period column to get the experimental file names accordingly
if diff_runs.Period[run_num].rstrip() == 'cooldown June 2021':
    
    #%%%% Cooldown (mode 1)
    #sand store temperatures file
    filename1 = 'Sand_temperatures_2021-05-28-to-2021-07-06.dat' 
    #loops heat transfer file
    filename2 ='Loops_heat_transfer_2021-05-28-to-2021-07-06.dat'

elif diff_runs.Period[run_num].rstrip() == 'charging Mar 2021':

    #%%%% Charging (mode 2)
    #sand store temperatures file
    filename1 = 'Sand_temperatures_2021-03-21-to-2021-03-23.dat' 
    #loops heat transfer file
    filename2 ='Loops_heat_transfer_2021-03-21-to-2021-03-23.dat'

elif diff_runs.Period[run_num].rstrip() == 'discharging Nov 2020':

    #%%%% Discharging (mode 3)
    #sand store temperatures file
    filename1 = 'Sand_temperatures_2020-11-25-to-2020-11-27.dat' 
    #loops heat transfer file
    filename2 ='Loops_heat_transfer_2020-11-25-to-2020-11-27.dat'

elif diff_runs.Period[run_num].rstrip() == 'charging + discharging Nov 29':

    #%%%% Charging and Discharging (mode 4)
    #sand store temperatures file
    filename1 = 'Sand_temperatures_2020-11-28-to-2020-11-30.dat' 
    #loops heat transfer file
    filename2 ='Loops_heat_transfer_2020-11-28-to-2020-11-30.dat'
    
elif diff_runs.Period[run_num].rstrip() == 'charging + discharging Nov 23':

    #%%%% Charging and Discharging 2 (mode 5)
    #sand store temperatures file
    filename1 = 'Sand_temperatures_2020-11-23-to-2020-11-23.dat' 
    #loops heat transfer file
    filename2 ='Loops_heat_transfer_2020-11-23-to-2020-11-23.dat'
elif diff_runs.Period[run_num].rstrip() == 'heating season 2021':

    #%%%% Entire heating season (mode 6)
    #sand store temperatures file
    filename1 = 'Sand_temperatures_2020-11-19-to-2021-04-30.dat' 
    #loops heat transfer file
    filename2 ='Loops_heat_transfer_2020-11-19-to-2021-04-30.dat'
else:
    print("Experimental data files not correctly selected. Please check run number and that the Excel input file is correct.")    
#%%% Processing of dates
#Parse out dates between which to run the simulation

#find year 1
f1_name, f1_2, f1_3 = re.split('(_20)',filename1)

st1 = filename1.find('_20')
date_start = filename1[st1+1:st1+11]

end1 = filename1.find('to-')
date_end = filename1[end1+3:-4]

#convert to datetime format
start_YYYY, start_MM, start_DD = date_start.split('-')
end_YYYY, end_MM, end_DD = date_end.split('-')

date_start = date(int(start_YYYY),int(start_MM),int(start_DD))
date_end = date(int(end_YYYY),int(end_MM),int(end_DD))

#Read in Headers file
dat_headers = pd.read_csv(data_dir.joinpath('Serrano_dat_headers.csv'))

#Read in sand temps data into a DataFrame
Exp_sand_temps = read_sand_temps(filename1, dat_headers, main_dir, data_dir)

#Read in loops heat transfer data into DataFrame
Exp_heat_transfer_data = read_loops_heat_transfer(filename2, dat_headers, main_dir, data_dir)
#Initialize boolean variable
missing_heat_data = False

# %%1. Timestep Parameters - CAN BE CHANGED

dt = 300 #time step, in seconds
output_temp_freq_h = 1    #how often the TC temperatures from the 3-D sand store will be output to the data file, in hours
visualization_output_freq_h = 24 #how often the temperatures from the domain nodes (more nodes than TC temps) will be output to the data file, in hours

output_temp_freq_int = (output_temp_freq_h * 3600)/dt #frequency that temperatures of the 3-D sand store will be output to the data file, in number of timestep intervals
viz_output_freq_int = (visualization_output_freq_h * 3600)/dt #frequency that temperatures of the domain will be output to the data file, in number of timestep intervals

#How many days in the simulation?
if (date_start == date_end):
    numberofdays_simu = 1
else:
    numberofdays_simu = (date_end - date_start).days + 1

tot_time = numberofdays_simu*24*3600 #1h = 3600s
timeIntrvls_per_day = int(24*3600/dt)	# number of time step intervals per day
totTimeIntrvls = int(tot_time/dt)	# total number of time step intervals

# %%1. Kusuda parameters - CAN BE CHANGED

#KUSUDA PARAMETERS for Ottawa(ground temp) 
T_avg = 8 #(C)
ampl = 18.507 
#depth from surface (m)
damp_dpth = 3 
phase =0.38 

#different parameters on north domain boundary due to presence of house
if north_BC_diff == 1:
    T_avg_N = 8+4 #(C)
    ampl_N = 14 
else:
    T_avg_N = 8
    ampl_N = 18.507

# %% 2. Process Experimental Sand Temp data into Numpy arrays
"""
This script processes the experimental data, output by Serrano, from a .dat 
or .csv file into Matlab matrix form, so that it can be used later

RECALL!!!! k = 0 is bottom of sand store, while C-layer is bottom-most plane
"""

#Put time data into a new numpy array (shows the time in hours from the start of the year)
time_h = Exp_sand_temps['time'].copy().to_numpy()

#rows and columns of sand temp data
[A, B] = Exp_sand_temps.shape

#create 4-D array to hold temp data
Exp_sand_temps_4d = np.zeros(shape=[6,6,3,A])

#create loop to map expt'l data to 4-D matrix
for indx1 in range(0,A):
    
    indx2 = 1 #index for the number of columns in the matrix. It starts at 1 since the first column is the time-stamp (zero-indexed)
    
    for k in range (2,-1,-1):
        for i in range (0,6):
            for j in range (0,6):
                Exp_sand_temps_4d[i,j,k,indx1] = Exp_sand_temps.iloc[indx1,indx2]
                
                #increment the column to be read next time
                indx2 = indx2 + 1
                #check that it exists, and that it's not the end of the line
                if indx2 >= B:
                    #break out of the innermost loop    
                    break 
                
         
#take averages of sand temps (lowest index is bottom-most layer of sand store)
avg_expmtl_temps = np.mean(np.mean(Exp_sand_temps_4d,axis=0),axis=0).transpose()

#convert time in hours into minutes from start of year
time_min = time_h*60
    
#Calculate day of year data starts
DOY = math.floor(time_h[0]/24)+1
# %% 2. Geometry of Domain
# %%% High-level geometry

#x-y-z origin lies at north-west corner, bottom of sand store
sand_L = 6 #(m) length of sand store  x = 0 m to x = 6 m
sand_W = 6 #(m) width of sand store  y = 0 m to y = 6 m
sand_H = 3 #(m) height of sand store, from bottom z = 0 m to top z = 3 m

XPS_th = 0.35 #(m) spec'd thickness of insulation around sand store (14")
conc_th = 0.10 #(m) spec'd thickness of concrete

soil_top_th = 1.5 #(m) thickness of soil layer on top of sand store to surface
soil_nrthside_th = 3.5 #(m) thickness of soil layer on north side of sand store to soil near house 
soil_th = 10 #(m) thickness of soil layer on S, E, W, and bottom sides of sand store to far-field temp  

no_TC = 0.5            #(m) buffer of 0.5m between sand store wall and 1st TCs (according to Briana Kemery's Visio document:https://chernode.mae.carleton.ca/sbes_svn/CHEeR_instrumentation/Sensor_connections_to_DAQs/Sand_store_TC_and_moisture_meter_placements.vsdx
node_space = (sand_L - 2*no_TC)/5 #Calculating the distance (m) between TCs in the sand store, assuming that there are 6 x 6 TCs equally spaced on each layer,

no_pwr = 0.3			#(m) length around outside of sand store where the PEX tubes are not layed down (estimated from photos)
pwr_bott = 0.4			#(m) bottom z-value of x-z plane where PEX tubing is laid down
pwr_mid = 1.3			#(m) middle z-value of x-z plane where PEX tubing is laid down
pwr_upr = 2.2			#(m) top z-value of x-z plane where PEX tubing is laid down

domain_L = soil_th + soil_nrthside_th + conc_th*2 + XPS_th*2 + sand_L #(m) length of numerical domain, x-axis
domain_W =  soil_th*2 + conc_th*2 + XPS_th*2 + sand_W#(m) width of numerical domain, y-axis
domain_H =  soil_th + soil_top_th + conc_th + XPS_th*2 + sand_H#(m) height of numerical domain, z-axis


#%%% Node Placement (varying mesh size)

#In x (or y-direction), from x=0 to the south. 
#0m = north far-field temperature

"""
The following values can be changed to see the effect of number of nodes on the final solution
"""
dx_soil = 0.3 #(m) dx of nodes in coarse grid horiz soil layer: in between the soil far-field boundary and the "increased node spacing layer"
dx_soilconc_boundary = 0.1 #(m) dx of nodes in fine grid horiz soil layer: between the coarse grid and the concrete layer 
soilconc_boundary_th = 0.2 #(m) thickness of fine grid horiz soil layer
nodes_conc = 2 #number of nodes to make up concrete layer (horiz and vertical)
nodes_XPS = 2 #number of nodes to make up XPS layer (horiz and vertical)
nodes_no_pwr = 2 # number of horiz. nodes in sand layer, in the "no_pwr" area
dx_sand = 0.2 #(m) dx of horiz. nodes in sand layer, inside the "pwr" area

dz_soil_top = 0.3 #(m)dz of nodes in top vert. soil layer
dz_pwr_th = 0.03 #(m) the vertical node spacing in the three power slab layers. Each power slab layer has three equi-sized dz layers 
                        #around and on it, with the node of the middle layer directly on the power slab line (i.e. 0.4m, 1.3m, and 2.2m 
                        #above the sand store bottom)
nodes_dz1_sand = 2 #number of nodes in between pwr_bott layer and bottom of sand store (not including the three nodes around the pwr_bott layer)
nodes_dz2_sand = 4 #number of nodes in between pwr_mid layer and pwr_bott layer (not including the three nodes around each pwr layer)
nodes_dz3_sand = 4 #number of nodes in between pwr_upr layer and pwr_mid layer (not including the three nodes around each pwr layer)
nodes_dz4_sand = 4 #number of nodes in between pwr_upr layer and top of sand store(not including the three nodes around pwr_upr layer)

#%%% Horizontal Coordinate Vectors
#%%%% SOIL COORDS

#(Note:the vector of dx's is the same as the vector for dy's, except for
#the length of the northside soil thickness)
#============================================================
#============================================================
#number of nodes in coarser soil grid
QN1 = math.floor(round((soil_nrthside_th - soilconc_boundary_th)/dx_soil,3))
#remainder of coarse soil length needing to be accounted for
RN1 = (Decimal(soil_nrthside_th) - Decimal(soilconc_boundary_th)) % Decimal(dx_soil)
RN1 = check_mod(RN1,dx_soil)


#number of nodes in fine soil grid
Q2 = math.floor(round(soilconc_boundary_th/dx_soilconc_boundary,3))
#remainder of fine soil length needing to be accounted 
R2 = Decimal(soilconc_boundary_th) % Decimal(dx_soilconc_boundary)
R2 = check_mod(R2,dx_soilconc_boundary)

#SUBVECTOR in which to put X-COORDINATES OF NORTH SOIL NODES
#--------------------------------------------------
Hsoil_coords_n = np.zeros(shape=[1+QN1+Q2])

#far-field north boundary x=0 m
Hsoil_coords_n[0] = 0
#1st node's x-coordinate, in metres:
Hsoil_coords_n[1] = (dx_soil/2) + RN1 + R2
#starting at index 3, coarse nodes' x-coordinates in metres
for i in range(2,(1+QN1)):
    Hsoil_coords_n[i] = Hsoil_coords_n[i-1] + dx_soil

#1st node in fine grid area:
i = i+1
Hsoil_coords_n[i] = Hsoil_coords_n[i-1] + dx_soil/2 + dx_soilconc_boundary/2
i = i+1
#fine nodes' x-coordinates in metres
for i in range(i,len(Hsoil_coords_n)):
    Hsoil_coords_n[i] = Hsoil_coords_n[i-1] + dx_soilconc_boundary


#SUBVECTOR in which to put Y-COORDINATES OF WEST SOIL NODES
#--------------------------------------------------
#number of nodes in coarser soil grid
Q1 = math.floor((soil_th - soilconc_boundary_th)/dx_soil)
#remainder of coarse soil length needing to be accounted for
R1 = (soil_th - soilconc_boundary_th) % dx_soil

Hsoil_coords_w = np.zeros(shape=[1+Q1+Q2])

#far-field west boundary x=0 m
Hsoil_coords_w[0] = 0
#1st node's y-coordinate, in metres:
Hsoil_coords_w[1] = (dx_soil/2) + R1 + R2
#starting at index 3, coarse nodes' x-coordinates in metres
for i in range(2,(1+Q1)):
    Hsoil_coords_w[i] = Hsoil_coords_w[i-1] + dx_soil

#1st node in fine grid area:
i = i+1
Hsoil_coords_w[i] = Hsoil_coords_w[i-1] + dx_soil/2 + dx_soilconc_boundary/2
i = i+1
#fine nodes' x-coordinates in metres
for i in range(i,len(Hsoil_coords_w)):
    Hsoil_coords_w[i] = Hsoil_coords_w[i-1] + dx_soilconc_boundary


#SUBVECTOR in which to put X-COORDINATES OF SOUTH SOIL NODES
#--------------------------------------------------
Hsoil_coords_s = np.zeros(shape=[1+Q1+Q2])

Hsoil_coords_s[(0)] = soil_nrthside_th + conc_th*2 + XPS_th*2 + sand_L + dx_soilconc_boundary/2
#fine nodes' x-coordinates in metres
for i in range(1,Q2):
    Hsoil_coords_s[i] = Hsoil_coords_s[i-1] + dx_soilconc_boundary

#1st node in coarse grid area:
i = i+1
Hsoil_coords_s[i] = Hsoil_coords_s[i-1] + dx_soil/2 + dx_soilconc_boundary/2
i = i+1
#coarse nodes' x-coordinates in metres
for i in range(i,(len(Hsoil_coords_s)-1)):
    Hsoil_coords_s[i] = Hsoil_coords_s[i-1] + dx_soil

#far-field south boundary x=domain_L
Hsoil_coords_s[(len(Hsoil_coords_s)-1)] = domain_L


#SUBVECTOR in which to put Y-COORDINATES OF EAST SOIL NODES
#--------------------------------------------------
Hsoil_coords_e = np.zeros(shape=[1+Q1+Q2])

Hsoil_coords_e[(0)] = soil_th + conc_th*2 + XPS_th*2 + sand_W + dx_soilconc_boundary/2
#fine nodes' x-coordinates in metres
for i in range(1,Q2):
    Hsoil_coords_e[i] = Hsoil_coords_e[i-1] + dx_soilconc_boundary

#1st node in coarse grid area:
i = i+1
Hsoil_coords_e[i] = Hsoil_coords_e[i-1] + dx_soil/2 + dx_soilconc_boundary/2
i = i+1
#coarse nodes' x-coordinates in metres
for i in range(i,(len(Hsoil_coords_e)-1)):
    Hsoil_coords_e[i] = Hsoil_coords_e[i-1] + dx_soil

#far-field south boundary y=domain_W
Hsoil_coords_e[(len(Hsoil_coords_e)-1)] = domain_W


# %%%% CONCRETE COORDS
#============================================================
Hconc_coords_n = np.zeros(shape=[nodes_conc])
Hconc_coords_w = np.zeros(shape=[nodes_conc])
Hconc_coords_s = np.zeros(shape=[nodes_conc])
Hconc_coords_e = np.zeros(shape=[nodes_conc])

#dx for concrete:
Q3 = math.floor(conc_th*100/nodes_conc)/100
#remainder of concrete length needing to be accounted for
R3 = (conc_th*100 % nodes_conc)/100

#1st conc node
Hconc_coords_n[0] = soil_nrthside_th + Q3/2 + R3
Hconc_coords_w[0] = soil_th + Q3/2 + R3
Hconc_coords_s[0] = soil_nrthside_th + conc_th + XPS_th*2 + sand_L + Q3/2
Hconc_coords_e[0] = soil_th + conc_th + XPS_th*2 + sand_W + Q3/2

#remainder of nodes
for i in range(1,nodes_conc):
    Hconc_coords_n[i] = Hconc_coords_n[i-1] + Q3
    Hconc_coords_w[i] = Hconc_coords_w[i-1] + Q3
    Hconc_coords_s[i] = Hconc_coords_s[i-1] + Q3
    Hconc_coords_e[i] = Hconc_coords_e[i-1] + Q3


#%%%% XPS COORDS
#============================================================
HXPS_coords_n = np.zeros(shape=[nodes_XPS])
HXPS_coords_w = np.zeros(shape=[nodes_XPS])
HXPS_coords_s = np.zeros(shape=[nodes_XPS])
HXPS_coords_e = np.zeros(shape=[nodes_XPS])

#dx for XPS:
Q4 = math.floor(XPS_th*100/nodes_XPS)/100
#remainder of XPS length needing to be accounted for
R4 = (XPS_th*100 % nodes_XPS)/100

#1st XPS node
HXPS_coords_n[0] = soil_nrthside_th + conc_th + Q4/2 + R4
HXPS_coords_w[0] = soil_th + conc_th + Q4/2 + R4
HXPS_coords_s[0] = soil_nrthside_th + conc_th + XPS_th + sand_L + Q4/2
HXPS_coords_e[0] = soil_th + conc_th + XPS_th + sand_W + Q4/2

#remainder of nodes
for i in range(1,nodes_XPS):
    HXPS_coords_n[i] = HXPS_coords_n[i-1] + Q4
    HXPS_coords_w[i] = HXPS_coords_w[i-1] + Q4
    HXPS_coords_s[i] = HXPS_coords_s[i-1] + Q4
    HXPS_coords_e[i] = HXPS_coords_e[i-1] + Q4


#%%%% SAND, NO-POWER ZONE COORDS (north and west sides)
#============================================================
Hnopwr_coords_n = np.zeros(shape=[nodes_no_pwr])
Hnopwr_coords_w = np.zeros(shape=[nodes_no_pwr])
Hnopwr_coords_s = np.zeros(shape=[nodes_no_pwr])
Hnopwr_coords_e = np.zeros(shape=[nodes_no_pwr])

#dx for no_pwr:
Q5 = math.floor(no_pwr*100/nodes_no_pwr)/100
#remainder of no_pwr length needing to be accounted for
R5 = (no_pwr*100 % nodes_no_pwr)/100

#1st no_pwr node on north and west sides
Hnopwr_coords_n[0] = soil_nrthside_th + conc_th + XPS_th + Q5/2 + R5
Hnopwr_coords_w[0] = soil_th + conc_th + XPS_th + Q5/2 + R5

#remainder of nodes on north and west sides
for i in range(1,nodes_no_pwr):
    Hnopwr_coords_n[i] = Hnopwr_coords_n[i-1] + Q5
    Hnopwr_coords_w[i] = Hnopwr_coords_w[i-1] + Q5


#%%%% SAND, POWER ZONE COORDS
#============================================================
#number of nodes in sand power area, x-dir
Q6 = math.floor(round((sand_L - no_pwr*2)/dx_sand,3))
#remainder of sand length needing to be accounted for
R6 = (Decimal(sand_L) - Decimal(no_pwr)*2) % Decimal(dx_sand)
R6 = check_mod(R6,dx_sand)

#number of nodes in sand power area, y-dir
Q7 = math.floor(round((sand_W - no_pwr*2)/dx_sand,3))
#remainder of sand width needing to be accounted for
R7 = (Decimal(sand_W) - Decimal(no_pwr)*2) % Decimal(dx_sand)
R7 = check_mod(R7,dx_sand)


Hsand_coords_x = np.zeros(shape=[Q6])
Hsand_coords_y = np.zeros(shape=[Q7])

Hsand_coords_x[0] = soil_nrthside_th + conc_th + XPS_th + no_pwr + dx_sand/2 + R6
Hsand_coords_y[0] = soil_th + conc_th + XPS_th + no_pwr + dx_sand/2 + R7

#remainder of nodes
for i in range(1,Q6):
    Hsand_coords_x[i] = Hsand_coords_x[i-1] + dx_sand

for j in range(1,Q7):
    Hsand_coords_y[(j)] = Hsand_coords_y[(j-1)] + dx_sand


#%%%% SAND NO-POWER ZONE COORDS (south and east sides)
#============================================================
#1st no_pwr node on south and east sides
Hnopwr_coords_s[0] = soil_nrthside_th + conc_th + XPS_th + sand_L - no_pwr + Q5/2
Hnopwr_coords_e[0] = soil_th + conc_th + XPS_th + sand_W - no_pwr + Q5/2

#remainder of nodes on south and east sides
for i in range(1,nodes_no_pwr):
    Hnopwr_coords_s[i] = Hnopwr_coords_s[i-1] + Q5
    Hnopwr_coords_e[i] = Hnopwr_coords_e[i-1] + Q5


#%%%% PUTTING X- AND Y- MESH COORDINATES VECTORS TOGETHER
#============================================================
node_xcoords = np.concatenate((Hsoil_coords_n,Hconc_coords_n,HXPS_coords_n,Hnopwr_coords_n,Hsand_coords_x,Hnopwr_coords_s,HXPS_coords_s,Hconc_coords_s,Hsoil_coords_s))

node_ycoords = np.concatenate((Hsoil_coords_w,Hconc_coords_w,HXPS_coords_w,Hnopwr_coords_w,Hsand_coords_y,Hnopwr_coords_e,HXPS_coords_e,Hconc_coords_e,Hsoil_coords_e))


#%%% Vertical Coordinate Vectors
#%%%% SOIL COORDS

'''
#Reminder:
#number of nodes in coarser soil grid = Q1
#remainder of coarse soil length needing to be accounted for = R1

#number of nodes in fine soil grid = Q2
#remainder of fine soil length needing to be accounted for = R2
'''

#SUBVECTOR in which to put Z-COORDINATES OF BOTTOM SOIL NODES
#--------------------------------------------------
Vsoil_coords_b = np.zeros(shape=[1+Q1+Q2])

#far-field bottom boundary z=0 m
Vsoil_coords_b[0] = 0
#1st node's z-coordinate, in metres:
Vsoil_coords_b[1] = (dx_soil/2) + R1 + R2
#starting at index 3, coarse nodes' z-coordinates in metres
for i in range(2,(1+Q1)):
    Vsoil_coords_b[i] = Vsoil_coords_b[i-1] + dx_soil

#1st node in fine grid area:
i = i+1
Vsoil_coords_b[i] = Vsoil_coords_b[i-1] + dx_soil/2 + dx_soilconc_boundary/2
i = i+1
#fine nodes' z-coordinates in metres
for i in range(i,len(Vsoil_coords_b)):
    Vsoil_coords_b[i] = Vsoil_coords_b[i-1] + dx_soilconc_boundary



#SUBVECTOR in which to put Z-COORDINATES OF TOP SOIL NODES
#--------------------------------------------------
#number of nodes in coarser soil grid
Q8 = math.floor((soil_top_th - soilconc_boundary_th)/dz_soil_top)
#remainder of coarse soil length needing to be accounted for
R8 = (soil_top_th - soilconc_boundary_th) % dz_soil_top

Vsoil_coords_t = np.zeros(shape=[1+Q8+Q2])

Vsoil_coords_t[0] = soil_th + conc_th + XPS_th*2 + sand_H + dx_soilconc_boundary/2
#fine nodes' z-coordinates in metres
for i in range(1,Q2):
    Vsoil_coords_t[i] = Vsoil_coords_t[i-1] + dx_soilconc_boundary

#1st node in coarse grid area:
i = i+1
Vsoil_coords_t[i] = Vsoil_coords_t[i-1] + dz_soil_top/2 + dx_soilconc_boundary/2
i = i+1
#coarse nodes' z-coordinates in metres
for i in range(i,len(Vsoil_coords_t)-1):
    Vsoil_coords_t[i] = Vsoil_coords_t[i-1] + dz_soil_top

#far-field south boundary x=domain_L
Vsoil_coords_t[len(Vsoil_coords_t)-1] = domain_H


#%%%% CONCRETE COORDS
#============================================================
Vconc_coords_b = np.zeros(shape=[nodes_conc])

#dx for concrete = Q3:
#remainder of concrete length needing to be accounted for = R3

#1st conc node
Vconc_coords_b[0] = soil_th + Q3/2 + R3

#remainder of nodes
for i in range(1,nodes_conc):
    Vconc_coords_b[i] = Vconc_coords_b[i-1] + Q3


#%%%% XPS COORDS
#============================================================
VXPS_coords_b = np.zeros(shape=[nodes_XPS])
VXPS_coords_t = np.zeros(shape=[nodes_XPS])

#dx for XPS = Q4
#remainder of XPS length needing to be accounted for = R4

#1st XPS node
VXPS_coords_b[0] = soil_th + conc_th + Q4/2 + R4
VXPS_coords_t[0] = soil_th + conc_th + XPS_th + sand_H + Q4/2

#remainder of nodes
for i in range(1,nodes_XPS):
    VXPS_coords_b[i] = VXPS_coords_b[i-1] + Q4
    VXPS_coords_t[i] = VXPS_coords_t[i-1] + Q4


#%%%% SAND COORDS
#============================================================
#dz in space in between pwr_bott layer and bottom of sand store (not including the three nodes around the pwr_bott layer)
dz1 = (pwr_bott - (dz_pwr_th + dz_pwr_th/2))/nodes_dz1_sand
#initialize vector
Vsand1 = np.zeros(shape=[nodes_dz1_sand+3])
#1st node coordinate
Vsand1[0] = soil_th + conc_th + XPS_th + dz1/2
#remainder of nodes
for i in range(1,nodes_dz1_sand):
    Vsand1[i] = Vsand1[i-1] + dz1

#1st node in "3-node pwr layer" 
i = i+1
Vsand1[i] = Vsand1[i-1] + dz1/2 + dz_pwr_th/2
i = i+1
#remainder of nodes in "3-node pwr layer" 
for i in range(i,len(Vsand1)):
    Vsand1[i] = Vsand1[i-1] + dz_pwr_th


#*********

#dz in space #number of nodes in between pwr_mid layer and pwr_bott layer (not including the three nodes around each pwr layer)
dz2 = ((pwr_mid - (dz_pwr_th + dz_pwr_th/2)) - (pwr_bott + (dz_pwr_th + dz_pwr_th/2)))/nodes_dz2_sand
#initialize vector
Vsand2 = np.zeros(shape=[nodes_dz2_sand+3])
#1st node coordinate
Vsand2[0] = soil_th + conc_th + XPS_th + pwr_bott + (dz_pwr_th + dz_pwr_th/2) + dz2/2
#remainder of nodes
for i in range(1,nodes_dz2_sand):
    Vsand2[i] = Vsand2[i-1] + dz2

#1st node in "3-node pwr layer" 
i = i+1
Vsand2[i] = Vsand2[i-1] + dz2/2 + dz_pwr_th/2
i = i+1
#remainder of nodes in "3-node pwr layer" 
for i in range(i,len(Vsand2)):
    Vsand2[i] = Vsand2[i-1] + dz_pwr_th


#*********

#dz in space #number of nodes in between pwr_upr layer and pwr_mid layer (not including the three nodes around each pwr layer)
dz3 = ((pwr_upr - (dz_pwr_th + dz_pwr_th/2)) - (pwr_mid + (dz_pwr_th + dz_pwr_th/2)))/nodes_dz3_sand
#initialize vector
Vsand3 = np.zeros(shape=[nodes_dz3_sand+3])
#1st node coordinate
Vsand3[0] =  soil_th + conc_th + XPS_th + pwr_mid + (dz_pwr_th + dz_pwr_th/2) + dz3/2
#remainder of nodes
for i in range(1,nodes_dz3_sand):
    Vsand3[i] = Vsand3[i-1] + dz3


#1st node in "3-node pwr layer" 
i = i+1
Vsand3[i] = Vsand3[i-1] + dz3/2 + dz_pwr_th/2
i = i+1
#remainder of nodes in "3-node pwr layer" 
for i in range(i,len(Vsand3)):
    Vsand3[i] = Vsand3[i-1] + dz_pwr_th


#*********

#dz in space #number of nodes in between pwr_upr layer and pwr_mid layer (not including the three nodes around each pwr layer)
dz4 = (sand_H - (pwr_upr + (dz_pwr_th + dz_pwr_th/2)))/nodes_dz4_sand
#initialize vector
Vsand4 = np.zeros(shape=[nodes_dz4_sand])
#1st node coordinate
Vsand4[0] =  soil_th + conc_th + XPS_th + pwr_upr + (dz_pwr_th + dz_pwr_th/2) + dz4/2
#remainder of nodes
for i in range(1,nodes_dz4_sand):
    Vsand4[i] = Vsand4[i-1] + dz4



#%%%% PUTTING Z- COORDINATE VECTORS TOGETHER
#============================================================
node_zcoords = np.concatenate((Vsoil_coords_b,Vconc_coords_b,VXPS_coords_b,Vsand1,Vsand2,Vsand3,Vsand4,VXPS_coords_t,Vsoil_coords_t))
# figure
# plot(ones(length(node_zcoords),1),node_zcoords,'o')


#%%% Print out mesh size spacing to screen
#------------------------------------------------------------------------
print('------------')
print('MESH SIZING:')
print('------------')
print('Soil(coarse) dx,dy,dz =',dx_soil, 'm')
print('Soil(fine) dx,dy,dz =',dx_soilconc_boundary, 'm on',soilconc_boundary_th, 'm')
print('Concrete dx,dy,dz =',Q3, 'm')
print('XPS dx,dy,dz =',Q4, 'm')
print('Sand (no power zone) dx =',Q5, 'm')
print('Sand (power zone) dx,dy =',dx_sand, 'm\n')

print('Sand dz on 0 m < z <',pwr_bott,'m of sand store  =',"%.2f" % dz1, 'm')
print('Sand dz on ',pwr_bott,'m < z <',pwr_mid,'m of sand store  =',"%.2f" % dz2, 'm')
print('Sand dz on ',pwr_mid,'m < z <',pwr_upr,'m of sand store  =',"%.2f" % dz3, 'm')
print('Sand dz on ',pwr_upr,'m < z <',sand_H,'m of sand store  =',"%.2f" % dz4, 'm')


#%%%VERTICAL index boundaries (6 layers)
indx_bndrz_z = np.array([0,              #bottom soil
    len(Vsoil_coords_b)-1,      #bottom soil                               
    len(Vsoil_coords_b),                        #bottom conc
    len(Vsoil_coords_b)+len(Vconc_coords_b)-1,    #bottom conc
    len(Vsoil_coords_b)+len(Vconc_coords_b),                    #bottom XPS
    len(Vsoil_coords_b)+len(Vconc_coords_b)+len(VXPS_coords_b)-1, #bottom XPS
    len(Vsoil_coords_b)+len(Vconc_coords_b)+len(VXPS_coords_b), #sand
    len(Vsoil_coords_b)+len(Vconc_coords_b)+len(VXPS_coords_b)+len(Vsand1)+len(Vsand2)+len(Vsand3)+len(Vsand4)-1, #sand
    len(Vsoil_coords_b)+len(Vconc_coords_b)+len(VXPS_coords_b)+len(Vsand1)+len(Vsand2)+len(Vsand3)+len(Vsand4), #top XPS
    len(Vsoil_coords_b)+len(Vconc_coords_b)+len(VXPS_coords_b)+len(Vsand1)+len(Vsand2)+len(Vsand3)+len(Vsand4)+len(VXPS_coords_t)-1,#top XPS
    len(Vsoil_coords_b)+len(Vconc_coords_b)+len(VXPS_coords_b)+len(Vsand1)+len(Vsand2)+len(Vsand3)+len(Vsand4)+len(VXPS_coords_t),#top soil
    len(node_zcoords)-1])#top soil

#Indices for the top node (inclusive) of each "moisture layer boundary"...appx middle points between moisture sensor z-coordinates
moisture_layer_height = np.zeros(shape=[3], dtype=int)
moisture_layer_height[0] = int(len(Vsoil_coords_b)+len(Vconc_coords_b)+len(VXPS_coords_b)+len(Vsand1)+ math.floor(nodes_dz2_sand/2) - 1) # -1 to acct for zero-based arrays
moisture_layer_height[1] = int(len(Vsoil_coords_b)+len(Vconc_coords_b)+len(VXPS_coords_b)+len(Vsand1)+len(Vsand2)+math.floor(nodes_dz3_sand/2) - 1) # -1 to acct for zero-based arrays
moisture_layer_height[2] = int(indx_bndrz_z[7])

vol_slab = (sand_L - 2*no_pwr)*(sand_W - 2*no_pwr)*dz_pwr_th #(m3) volume of each power source slab

#Define number of nodes in each direction in numerical model domain

U = len(node_xcoords)	#number of nodes in x-direction 
V = len(node_ycoords)	#number of nodes  in y-direction 
W = len(node_zcoords)	#number of nodes in z-direction 


pwr_bott_indx = indx_bndrz_z[5] + nodes_dz1_sand + 2	#[m] k-index for bottom layer of PEX tubing
pwr_mid_indx = indx_bndrz_z[5] + len(Vsand1) + nodes_dz2_sand + 2		#[m] k-index for middle layer of PEX tubing
pwr_upr_indx =  indx_bndrz_z[5] + len(Vsand1) + len(Vsand2) + nodes_dz3_sand + 2		#[m] k-index for top layer of PEX tubing

#%%%HORIZONTAL index boundaries [9 layers]
indx_bndrz_x = np.array([0,              #north soil
    len(Hsoil_coords_n)-1,      #north soil
    len(Hsoil_coords_n),                     #north conc
    len(Hsoil_coords_n)+len(Hconc_coords_n)-1, #north conc
    len(Hsoil_coords_n)+len(Hconc_coords_n),                      #north XPS
    len(Hsoil_coords_n)+len(Hconc_coords_n)+len(HXPS_coords_n)-1,   #north XPS
    len(Hsoil_coords_n)+len(Hconc_coords_n)+len(HXPS_coords_n),                       #north sand no power
    len(Hsoil_coords_n)+len(Hconc_coords_n)+len(HXPS_coords_n)+len(Hnopwr_coords_n)-1,  #north sand no power
    len(Hsoil_coords_n)+len(Hconc_coords_n)+len(HXPS_coords_n)+len(Hnopwr_coords_n),                    #sand
    len(Hsoil_coords_n)+len(Hconc_coords_n)+len(HXPS_coords_n)+
        len(Hnopwr_coords_n)+len(Hsand_coords_x)-1,#sand
    len(Hsoil_coords_n)+len(Hconc_coords_n)+len(HXPS_coords_n)+
        len(Hnopwr_coords_n)+len(Hsand_coords_x),                      #south sand no pwr
    len(Hsoil_coords_n)+len(Hconc_coords_n)+len(HXPS_coords_n)+
        len(Hnopwr_coords_n)+len(Hsand_coords_x)+len(Hnopwr_coords_s)-1, #south sand no pwr
    len(Hsoil_coords_n)+len(Hconc_coords_n)+len(HXPS_coords_n)+
        len(Hnopwr_coords_n)+len(Hsand_coords_x)+len(Hnopwr_coords_s),   #south XPS
    len(Hsoil_coords_n)+len(Hconc_coords_n)+len(HXPS_coords_n)+
        len(Hnopwr_coords_n)+len(Hsand_coords_x)+len(Hnopwr_coords_s)+
        len(HXPS_coords_s)-1,   #south XPS
    len(Hsoil_coords_n)+len(Hconc_coords_n)+len(HXPS_coords_n)+
        len(Hnopwr_coords_n)+len(Hsand_coords_x)+len(Hnopwr_coords_s)+
        len(HXPS_coords_s),                     #south conc
    len(Hsoil_coords_n)+len(Hconc_coords_n)+len(HXPS_coords_n)+
        len(Hnopwr_coords_n)+len(Hsand_coords_x)+len(Hnopwr_coords_s)+
        len(HXPS_coords_s)+len(Hconc_coords_s)-1, #south conc
    len(Hsoil_coords_n)+len(Hconc_coords_n)+len(HXPS_coords_n)+
        len(Hnopwr_coords_n)+len(Hsand_coords_x)+len(Hnopwr_coords_s)+
        len(HXPS_coords_s)+len(Hconc_coords_s), #south soil
    len(node_xcoords)-1])                             #south soil
    

indx_bndrz_y = np.array([0,              #west soil
    len(Hsoil_coords_w)-1,      #west soil
    len(Hsoil_coords_w),                     #west conc
    len(Hsoil_coords_w)+len(Hconc_coords_w)-1, #west conc
    len(Hsoil_coords_w)+len(Hconc_coords_w),                      #west XPS
    len(Hsoil_coords_w)+len(Hconc_coords_w)+len(HXPS_coords_w)-1,   #west XPS
    len(Hsoil_coords_w)+len(Hconc_coords_w)+len(HXPS_coords_w),                       #west sand no power
    len(Hsoil_coords_w)+len(Hconc_coords_w)+len(HXPS_coords_w)+len(Hnopwr_coords_w)-1,  #west sand no power
    len(Hsoil_coords_w)+len(Hconc_coords_w)+len(HXPS_coords_w)+
        len(Hnopwr_coords_w),                    #sand
    len(Hsoil_coords_w)+len(Hconc_coords_w)+len(HXPS_coords_w)+
        len(Hnopwr_coords_w)+len(Hsand_coords_y)-1,#sand
    len(Hsoil_coords_w)+len(Hconc_coords_w)+len(HXPS_coords_w)+
        len(Hnopwr_coords_w)+len(Hsand_coords_y),                      #east sand no pwr
    len(Hsoil_coords_w)+len(Hconc_coords_w)+len(HXPS_coords_w)+
        len(Hnopwr_coords_w)+len(Hsand_coords_y)+len(Hnopwr_coords_e)-1, #east sand no pwr
    len(Hsoil_coords_w)+len(Hconc_coords_w)+len(HXPS_coords_w)+
        len(Hnopwr_coords_w)+len(Hsand_coords_y)+len(Hnopwr_coords_e), #east XPS
    len(Hsoil_coords_w)+len(Hconc_coords_w)+len(HXPS_coords_w)+
        len(Hnopwr_coords_w)+len(Hsand_coords_y)+len(Hnopwr_coords_e)+
        len(HXPS_coords_e)-1,   #east XPS
    len(Hsoil_coords_w)+len(Hconc_coords_w)+len(HXPS_coords_w)+
        len(Hnopwr_coords_w)+len(Hsand_coords_y)+len(Hnopwr_coords_e)+
        len(HXPS_coords_e),                     #east conc
    len(Hsoil_coords_w)+len(Hconc_coords_w)+len(HXPS_coords_w)+
        len(Hnopwr_coords_w)+len(Hsand_coords_y)+len(Hnopwr_coords_e)+
        len(HXPS_coords_e)+len(Hconc_coords_e)-1, #east conc
    len(Hsoil_coords_w)+len(Hconc_coords_w)+len(HXPS_coords_w)+
        len(Hnopwr_coords_w)+len(Hsand_coords_y)+len(Hnopwr_coords_e)+
        len(HXPS_coords_e)+len(Hconc_coords_e), #east soil
    len(node_ycoords)-1])                  #east soil


#%%%Indices for points within the "power slab" in the x-y plane
x_pwr_low_indx = indx_bndrz_x[8] #inclusive of this index
x_pwr_high_indx = indx_bndrz_x[9]#inclusive of this index
y_pwr_low_indx = indx_bndrz_y[8] #inclusive of this index
y_pwr_high_indx = indx_bndrz_y[9]#inclusive of this index

#============================================================
#%%%Vectors with CARTESIAN COORDS OF LAYER BOUNDARIES
#north-south
N_S_layr_bounds = np.array([0,          #north boundary
    soil_nrthside_th,          #soil/conc
    soil_nrthside_th + conc_th,             #conc/XPS
    soil_nrthside_th + conc_th + XPS_th ,   #XPS/sand
    soil_nrthside_th + conc_th + XPS_th + sand_L,   #sand/XPS
    soil_nrthside_th + conc_th + XPS_th*2 + sand_L, #XPS/conc
    soil_nrthside_th + conc_th*2 + XPS_th*2 + sand_L,#conc/soil
    soil_nrthside_th + conc_th*2 + XPS_th*2 + sand_L + soil_th])  #south boundary


#west-east
W_E_layr_bounds =np.array([0, #west boundary
    soil_th,          #soil/conc
    soil_th + conc_th,             #conc/XPS
    soil_th + conc_th + XPS_th ,   #XPS/sand
    soil_th + conc_th + XPS_th + sand_W,   #sand/XPS
    soil_th + conc_th + XPS_th*2 + sand_W, #XPS/conc
    soil_th + conc_th*2 + XPS_th*2 + sand_W,#conc/soil
    soil_th*2 + conc_th*2 + XPS_th*2 + sand_W]) #east boundary

#bottom-top
B_T_layr_bounds =np.array([0, #bottom boundary
    soil_th,          #soil/conc
    soil_th + conc_th,             #conc/XPS
    soil_th + conc_th + XPS_th ,   #XPS/sand
    soil_th + conc_th + XPS_th + sand_H,   #sand/XPS
    soil_th + conc_th + XPS_th*2 + sand_H, #XPS/soil
    soil_th + conc_th + XPS_th*2 + sand_H + soil_top_th]) #top boundary

#============================================================
#%%%Vectors holding the x and y coordinates of the TCs in the sand store
TC_xcoords = np.zeros(shape=[6])
TC_ycoords = np.zeros(shape=[6])

TC_xcoords[0] = soil_nrthside_th + conc_th + XPS_th + no_TC
TC_ycoords[0] = soil_th + conc_th + XPS_th + no_TC

for i in range(1,6):
    TC_xcoords[i] = TC_xcoords[i-1] + node_space
    TC_ycoords[i] = TC_ycoords[i-1] + node_space


#Find the indices of the TC locations
sand_TC_node_xindices = np.zeros(shape=[6], dtype=int)

for TC in range(0,6):
    diff = 0
    prev = 100
    for i in range(0,len(node_xcoords)):
        diff = abs(TC_xcoords[TC] - node_xcoords[i])
       #this finds the closest nodes to the real locations of the TCs
        if round(diff,5) >= round(prev,5):
            #then we want the index represented by [i-1]
            sand_TC_node_xindices[TC] = i-1
            break
        prev = diff
        

sand_TC_node_yindices = np.zeros(shape=[6], dtype=int)

for TC in range(0,6):
    diff = 0
    prev = 100
    for j in range(0,len(node_ycoords)):
        diff = abs(TC_ycoords[TC] - node_ycoords[j])
       #this finds the closest nodes to the real locations of the TCs
        if round(diff,5) >= round(prev,5):
            #then we want the index represented by [j-1]
            sand_TC_node_yindices[TC] = j-1
            break
        prev = diff
      

# %% 2. Material Thermal Properties
'''
This script defines the thermal properties of the materials used in the simulation and also builds the 3-D thermal property matrices for the simulation.
'''
#Properties of water
water_Cp = 4.186*1e3  # (J/kgK) specific heat for water
water_rho = 998.2  # (kg/m3) density for water at 293 K https://hypertextbook.com/facts/2007/AllenMa.shtml

#Properties of dry sand -> also used Hamdhan and Clarke (2010) as a reference
drySand_Cp = 830 #(J/kgK) specific heat for dry sand https://www.engineeringtoolbox.com/specific-heat-capacity-d_391.html
drySand_rho = 1850 #(kg/m3) density for dry sand (personal correspondence with Abdulghader Abdulrahman, Research Manager, Dr.Paul Simms' lab)

#volume percents of water layers in sand store
vol_percent = np.zeros(shape=[3])
vol_percent[0] = diff_runs['vol% bott'][run_num] #bottom layer = 33 vol#
vol_percent[1] = diff_runs['vol% mid'][run_num] #middle layer = 11 vol#
vol_percent[2] = diff_runs['vol% top'][run_num] #top layer = 7 vol#

sand_lambda = np.zeros(shape=[3])
sand_lambda[0] = diff_runs['λ_bott'][run_num] #(W/mK) bottom layer - thermal conductivity for saturated sand F. Rad (2009)
sand_lambda[1] = diff_runs['λ_mid'][run_num] #(W/mK) middle layer - moist sand https://www.engineeringtoolbox.com/thermal-conductivity-d_429.html
sand_lambda[2] = diff_runs['λ_top'][run_num] #(W/mK) top layer - moist sand https://www.engineeringtoolbox.com/thermal-conductivity-d_429.html

#Properties of concrete at 300 K (Source - Incropera, 6e, p.939)
conc_Cp = 880 #(J/kgK) specific heat 
conc_rho = 2300  # (kg/m3) density 
conc_lambda = 1.4 #(W/mK) thermal conductivity

#Properties of XPS at 297 K(Source - Owens Corning data sheets and papers - see ppt presentation MCG5157 final project for sources)
XPS_Cp_orig = 1500 #(J/kgK) specific heat source - article by BASF staff
XPS_rho_orig = 30  # (kg/m3) density 

#Properties of XPS
vol_percent_H2O_SETface_XPS = diff_runs['SET(B) XPS Vol % H2O'][run_num]
XPS_lambda_SET = diff_runs['SET(B)'][run_num]  #(W/mK) thermal conductivity
XPS_Cp_SET = ((water_Cp * (water_rho*vol_percent_H2O_SETface_XPS)) + (XPS_Cp_orig*XPS_rho_orig)) / (water_rho * vol_percent_H2O_SETface_XPS + XPS_rho_orig) 
              # (Number of Joules to raise sand in m3 by 1K + Joules to raise water in 1m3 by 1K) / (number of kg in 1m3)           
XPS_rho_SET = water_rho * vol_percent_H2O_SETface_XPS + XPS_rho_orig 


#Properties of soil (soil_rhoCp and soil_lambda) around CHEeR house sand
#store
soil_Cp = 750 #(J/kgK) (A. Wills MASc,Table 6.1)  
soil_rho = 2800 #(kg/m3) (A. Wills MASc,Table 6.1)
soil_lambda = 2   #(W/mK) thermal conductivity (A. Wills MASc,Table 6.1) (TRY PARAMETRIZING FOR CLAY VALUES )

#%% 2. Print user inputs to screen:
print("\nCHECK THESE PARAMETERS  \n")
print("Start and end dates: " + str(date_start) + " to " + str(date_end) + "\n")
print("Soil width surrounding sand store: " + str(soil_th) + " m and " + str(soil_nrthside_th) + " m \n")
print("Kusuda T_avg: " + str(T_avg) + " oC \n")
print("Kusuda ampl: " + str(ampl) + "\n")


#%% 2. Build matrices of thermal properties
# (from outside layers inwards)

#Initialize thermal property matrices
Lambda = np.zeros(shape=[U,V,W])
Cp = np.zeros(shape=[U,V,W])
rho = np.zeros(shape=[U,V,W])

#Soil layer - make entire matrix soil
Lambda[:,:,:] = soil_lambda
Cp[:,:,:] = soil_Cp
rho[:,:,:] = soil_rho

#Concrete layer
for k in range(indx_bndrz_z[2],indx_bndrz_z[9]+1):
    for i in range(indx_bndrz_x[2],indx_bndrz_x[15]+1):
        for j in range(indx_bndrz_y[2],indx_bndrz_y[15]+1):
            Lambda[i,j,k] = conc_lambda
            Cp[i,j,k] = conc_Cp
            rho[i,j,k] = conc_rho
        
    


#Insulation layer
for k in range(indx_bndrz_z[4],indx_bndrz_z[9]+1):
    for i in range(indx_bndrz_x[4],indx_bndrz_x[13]+1):
        for j in range(indx_bndrz_y[4],indx_bndrz_y[13]+1):
            Lambda[i,j,k] = XPS_lambda_SET
            Cp[i,j,k] = XPS_Cp_SET
            rho[i,j,k] = XPS_rho_SET
        
        
#North insulation layer

#volume percent of water layers in XPS north face
vpH20_NXPS = diff_runs['North XPS Vol % H2O'][run_num] # just an acronym of "vol_percent_H2O_Nface_XPS"; to make it easier to change values when face is homogeneous
vol_percent_H2O_Nface_XPS = [ vpH20_NXPS,vpH20_NXPS ,vpH20_NXPS ]
northXPS_lambda = diff_runs['North'][run_num]
# northXPS_lambda = [2, 2, 2]

#1st layer [bottom of insulation to ~halfway between first two PEX layers]
for k in range(indx_bndrz_z[4],moisture_layer_height[0]+1):
    for i in range(indx_bndrz_x[4],indx_bndrz_x[5]+1):
        for j in range(indx_bndrz_y[4],indx_bndrz_y[13]+1):
            Lambda[i,j,k] = northXPS_lambda#[0] 
            Cp[i,j,k] = ((water_Cp * (water_rho*vol_percent_H2O_Nface_XPS[0])) + (XPS_Cp_orig*XPS_rho_orig)) / (water_rho * vol_percent_H2O_Nface_XPS[0] + XPS_rho_orig) 
                          # (Number of Joules to raise sand in m3 by 1K + Joules to raise water in 1m3 by 1K) / (number of kg in 1m3)           
            rho[i,j,k] = water_rho * vol_percent_H2O_Nface_XPS[0] + XPS_rho_orig 
        
#2nd layer [ of 1st layer to ~halfway between next two PEX layers]
for k in range((moisture_layer_height[0]+1),moisture_layer_height[1]+1):
    for i in range(indx_bndrz_x[4],indx_bndrz_x[5]+1):
        for j in range(indx_bndrz_y[4],indx_bndrz_y[13]+1):
            Lambda[i,j,k] = northXPS_lambda#[1]
            Cp[i,j,k] = ((water_Cp * (water_rho*vol_percent_H2O_Nface_XPS[1])) + (XPS_Cp_orig*XPS_rho_orig)) / (water_rho * vol_percent_H2O_Nface_XPS[1] + XPS_rho_orig) 
            rho[i,j,k] = water_rho * vol_percent_H2O_Nface_XPS[1] + XPS_rho_orig         

#3rd layer [ of 2nd layer to top of insulation]
for k in range((moisture_layer_height[1]+1),indx_bndrz_z[9]+1):
    for i in range(indx_bndrz_x[4],indx_bndrz_x[5]+1):
        for j in range(indx_bndrz_y[4],indx_bndrz_y[13]+1):
            Lambda[i,j,k] = northXPS_lambda#[2]
            Cp[i,j,k] = ((water_Cp * (water_rho*vol_percent_H2O_Nface_XPS[2])) + (XPS_Cp_orig*XPS_rho_orig)) / (water_rho * vol_percent_H2O_Nface_XPS[2] + XPS_rho_orig) 
            rho[i,j,k] = water_rho * vol_percent_H2O_Nface_XPS[2] + XPS_rho_orig
            

#West insulation layer

#volume percent of water layers in XPS west face
vpH20_WXPS =diff_runs['West XPS Vol % H2O'][run_num] # just an acronym of "vol_percent_H2O_Nface_XPS"; to make it easier to change values when face is homogeneous
vol_percent_H2O_Wface_XPS = [ vpH20_WXPS, vpH20_WXPS, vpH20_WXPS ]
westXPS_lambda = diff_runs['West'][run_num]
# westXPS_lambda = [0.6, 0.6, 0.6]

#1st layer [bottom of insulation to ~halfway between first two PEX layers]
for k in range(indx_bndrz_z[4],moisture_layer_height[0]+1):
    #this is starting at index 6 instead of 4 to not overwrite the properties already written in the North-face XPS section above
    for i in range(indx_bndrz_x[6],indx_bndrz_x[13]+1): 
        for j in range(indx_bndrz_y[4],indx_bndrz_y[5]+1):
            Lambda[i,j,k] = westXPS_lambda#[0] 
            Cp[i,j,k] = ((water_Cp * (water_rho*vol_percent_H2O_Wface_XPS[0])) + (XPS_Cp_orig*XPS_rho_orig)) / (water_rho * vol_percent_H2O_Wface_XPS[0] + XPS_rho_orig) 
                          # (Number of Joules to raise sand in m3 by 1K + Joules to raise water in 1m3 by 1K) / (number of kg in 1m3)           
            rho[i,j,k] = water_rho * vol_percent_H2O_Wface_XPS[0] + XPS_rho_orig 
        
#2nd layer [ of 1st layer to ~halfway between next two PEX layers]
for k in range((moisture_layer_height[0]+1),moisture_layer_height[1]+1):
    for i in range(indx_bndrz_x[6],indx_bndrz_x[13]+1):
        for j in range(indx_bndrz_y[4],indx_bndrz_y[5]+1):
            Lambda[i,j,k] = westXPS_lambda#[1]
            Cp[i,j,k] = ((water_Cp * (water_rho*vol_percent_H2O_Wface_XPS[1])) + (XPS_Cp_orig*XPS_rho_orig)) / (water_rho * vol_percent_H2O_Wface_XPS[1] + XPS_rho_orig) 
            rho[i,j,k] = water_rho * vol_percent_H2O_Wface_XPS[1] + XPS_rho_orig         

#3rd layer [ of 2nd layer to top of insulation]
for k in range((moisture_layer_height[1]+1),indx_bndrz_z[9]+1):
    for i in range(indx_bndrz_x[6],indx_bndrz_x[13]+1):
        for j in range(indx_bndrz_y[4],indx_bndrz_y[5]+1):
            Lambda[i,j,k] = westXPS_lambda#[2]
            Cp[i,j,k] = ((water_Cp * (water_rho*vol_percent_H2O_Wface_XPS[2])) + (XPS_Cp_orig*XPS_rho_orig)) / (water_rho * vol_percent_H2O_Wface_XPS[2] + XPS_rho_orig) 
            rho[i,j,k] = water_rho * vol_percent_H2O_Wface_XPS[2] + XPS_rho_orig

#Bottom insulation layer

#volume percent of water layers in XPS west face
vpH20_BXPS = diff_runs['Bott XPS Vol % H2O'][run_num] # just an acronym of "vol_percent_H2O_Nface_XPS"; to make it easier to change values when face is homogeneous
vol_percent_H2O_Bface_XPS = [ vpH20_BXPS, vpH20_BXPS, vpH20_BXPS ]
bottomXPS_lambda = diff_runs['Bott'][run_num]

for k in range(indx_bndrz_z[4],indx_bndrz_z[5]+1):
    #this is starting at index 6 instead of 4 to not overwrite the properties already written in the North-face XPS section above
    for i in range(indx_bndrz_x[6],indx_bndrz_x[13]+1): 
        for j in range(indx_bndrz_y[6],indx_bndrz_y[13]+1):
            Lambda[i,j,k] = bottomXPS_lambda#[0] 
            Cp[i,j,k] = ((water_Cp * (water_rho*vol_percent_H2O_Bface_XPS[0])) + (XPS_Cp_orig*XPS_rho_orig)) / (water_rho * vol_percent_H2O_Bface_XPS[0] + XPS_rho_orig) 
                          # (Number of Joules to raise sand in m3 by 1K + Joules to raise water in 1m3 by 1K) / (number of kg in 1m3)           
            rho[i,j,k] = water_rho * vol_percent_H2O_Bface_XPS[0] + XPS_rho_orig 
  

#-----------------------            
            
#1st Sand layer [0m to ~halfway between first two PEX layers]
for k in range(indx_bndrz_z[6],moisture_layer_height[0]+1):
    for i in range(indx_bndrz_x[6],indx_bndrz_x[11]+1):
        for j in range(indx_bndrz_y[6],indx_bndrz_y[11]+1):
            Lambda[i,j,k] = sand_lambda[0] 
            Cp[i,j,k] = ((water_Cp * (water_rho*vol_percent[0])) + (drySand_Cp*drySand_rho)) / (water_rho * vol_percent[0] + drySand_rho) 
                         # (Number of Joules to raise sand in m3 by 1K + Joules to raise water in 1m3 by 1K) / (number of kg in 1m3)           
            rho[i,j,k] = water_rho * vol_percent[0] + drySand_rho 
        
    


#2nd Sand layer [ of 1st layer to ~halfway between next two PEX layers]
for k in range((moisture_layer_height[0]+1),moisture_layer_height[1]+1):
    for i in range(indx_bndrz_x[6],indx_bndrz_x[11]+1):
        for j in range(indx_bndrz_y[6],indx_bndrz_y[11]+1):
            Lambda[i,j,k] = sand_lambda[1]
            Cp[i,j,k] = ((water_Cp * (water_rho*vol_percent[1])) + (drySand_Cp*drySand_rho)) / (water_rho * vol_percent[1] + drySand_rho) 
            rho[i,j,k] = water_rho * vol_percent[1] + drySand_rho         
        
    


#3rd Sand layer [ of 2nd layer to top of sand store]
for k in range((moisture_layer_height[1]+1),moisture_layer_height[2]+1):
    for i in range(indx_bndrz_x[6],indx_bndrz_x[11]+1):
        for j in range(indx_bndrz_y[6],indx_bndrz_y[11]+1):
            Lambda[i,j,k] = sand_lambda[2]
            Cp[i,j,k] = ((water_Cp * (water_rho*vol_percent[2])) + (drySand_Cp*drySand_rho)) / (water_rho * vol_percent[2] + drySand_rho) 
            rho[i,j,k] = water_rho * vol_percent[2] + drySand_rho    

# %% 2. Heat input to Sand Store
# %%% Process heat input data 

#Put heat source data into a new numpy array
solar_to_Sand_W = Exp_heat_transfer_data[['time','sand_H2O']].copy().to_numpy()

#Specific rate of heat transfer (W/m3) to sand store per unit volume of power source slabs
sp_heat_trnsfr=np.array(solar_to_Sand_W[:,1],ndmin=2).transpose()/(3*vol_slab)

#append specific rate of heat transfer to array
solar_to_Sand_W = np.append(solar_to_Sand_W,sp_heat_trnsfr,axis=1)

#%%%% Check for missing data  
if (solar_to_Sand_W.shape[0] < totTimeIntrvls):
    
    #set boolean marker to True
    missing_heat_data = True
    #make new array with orig data
    solar_to_Sand_W_orig_data = solar_to_Sand_W
    #re-initialize solar_to_Sand_W array
    solar_to_Sand_W = np.zeros(shape=(totTimeIntrvls,solar_to_Sand_W.shape[1]))
    
    new=0 #initialize second counter variable for loop below
    solar_to_Sand_W[0,0] = solar_to_Sand_W_orig_data[0,0]   #copy first data point
    
    for orig in range(0,solar_to_Sand_W_orig_data.shape[0]-1):
        
        #check gaps between data points to not be not larger than 2 time step intervals (dt), try 1.5 intervals
        #If gap is larger than 1.5*dt, fill in missing data
        if (solar_to_Sand_W_orig_data[orig+1,0] - solar_to_Sand_W_orig_data[orig,0]) > (1.5*dt/3600):
            #number of data points to add (fill in)
            fill = round((solar_to_Sand_W_orig_data[orig+1,0] - solar_to_Sand_W_orig_data[orig,0])*3600/dt)
            
            # print(solar_to_Sand_W_orig_data[orig+1,0],
            #       solar_to_Sand_W_orig_data[orig,0],
            #       (solar_to_Sand_W_orig_data[orig+1,0]- solar_to_Sand_W_orig_data[orig,0])*60,
            #       fill, 
            #       new, 
            #       orig, 
            #       new-orig)
            
            #fill the missing data with average of points on either side
            for i in range(0,fill-1):
                new = new + 1   #increment second counter variable
                #first column data fill
                solar_to_Sand_W[new,0] = solar_to_Sand_W[new-1,0] + (dt/3600)
                #second column data fill
                solar_to_Sand_W[new,1] = (solar_to_Sand_W_orig_data[orig+1,1] + 
                                          solar_to_Sand_W_orig_data[orig,1])/2
                #third column data fill
                solar_to_Sand_W[new,2] = (solar_to_Sand_W_orig_data[orig+1,2] + 
                                          solar_to_Sand_W_orig_data[orig,2])/2
                
            #copy the data point on the "far side of the data gap" to the new array
            solar_to_Sand_W[new+1,0] = solar_to_Sand_W_orig_data[orig+1,0]
            solar_to_Sand_W[new+1,1] = solar_to_Sand_W_orig_data[orig+1,1]
            solar_to_Sand_W[new+1,2] = solar_to_Sand_W_orig_data[orig+1,2]
            
        else:
            #copy original data over to new array
            solar_to_Sand_W[new+1,0] = solar_to_Sand_W_orig_data[orig+1,0]
            solar_to_Sand_W[new+1,1] = solar_to_Sand_W_orig_data[orig+1,1]
            solar_to_Sand_W[new+1,2] = solar_to_Sand_W_orig_data[orig+1,2]
            
        new = new + 1   #increment second counter variable (regardless of if data has been filled or not)
            
    
        
    
        
    
    print('*****************************************************************\n')
    print('GAPS IN EXPERIMENTAL SAND STORE HEAT DATA - VERIFY solar_to_Sand_W AND solar_to_Sand_W_orig_data ARRAYS MAKE SENSE!!!!\n')
    print('*****************************************************************\n\n')
    
else:
    print('*****************************************************************\n')
    print('NO MISSING SAND STORE HEAT DATA\n')
    print('*****************************************************************\n\n')
    

# %% 2. Heat extracted from Sand Store
# %%% Process discharging data 

#Put discharging data into a new numpy array
Sand_to_SH_W = Exp_heat_transfer_data[['time','sand_2_SH']].copy().to_numpy()

#Specific rate of heat extraction (W/m3) from sand store per unit volume of power source slabs
sp_heat_extr=np.array(Sand_to_SH_W[:,1],ndmin=2).transpose()/(3*vol_slab)

#append specific rate of heat extraction to array
Sand_to_SH_W = np.append(Sand_to_SH_W,sp_heat_extr,axis=1)


#%%%% Check for missing data  
if (Sand_to_SH_W.shape[0] < totTimeIntrvls):
    
    #set boolean marker to True
    missing_heat_data = True
    #make new array with orig data
    Sand_to_SH_W_orig_data = Sand_to_SH_W
    #re-initialize Sand_to_SH_W array
    Sand_to_SH_W = np.zeros(shape=(totTimeIntrvls,Sand_to_SH_W.shape[1]))
    
    new=0 #initialize second counter variable for loop below
    Sand_to_SH_W[0,0] = Sand_to_SH_W_orig_data[0,0]   #copy first data point
    
    for orig in range(0,Sand_to_SH_W_orig_data.shape[0]-1):
        
        #check gaps between data points to not be not larger than 2 time step intervals (dt), try 1.5 intervals
        #If gap is larger than 1.5*dt, fill in missing data
        if (Sand_to_SH_W_orig_data[orig+1,0] - Sand_to_SH_W_orig_data[orig,0]) > (1.5*dt/3600):
            #number of data points to add (fill in)
           
            fill = round((Sand_to_SH_W_orig_data[orig+1,0] - Sand_to_SH_W_orig_data[orig,0])*3600/dt)
            
            #fill the missing data with average of points on either side
            for i in range(0,fill-1):
                new = new + 1   #increment second counter variable
                #first column data fill
                Sand_to_SH_W[new,0] = Sand_to_SH_W[new-1,0] + (dt/3600)
                #second column data fill
                Sand_to_SH_W[new,1] = (Sand_to_SH_W_orig_data[orig+1,1] + 
                                          Sand_to_SH_W_orig_data[orig,1])/2
                #third column data fill
                Sand_to_SH_W[new,2] = (Sand_to_SH_W_orig_data[orig+1,2] + 
                                          Sand_to_SH_W_orig_data[orig,2])/2
                
            #copy the data point on the "far side of the data gap" to the new array
            Sand_to_SH_W[new+1,0] = Sand_to_SH_W_orig_data[orig+1,0]
            Sand_to_SH_W[new+1,1] = Sand_to_SH_W_orig_data[orig+1,1]
            Sand_to_SH_W[new+1,2] = Sand_to_SH_W_orig_data[orig+1,2]
            
        else:
            #copy original data over to new array
            Sand_to_SH_W[new+1,0] = Sand_to_SH_W_orig_data[orig+1,0]
            Sand_to_SH_W[new+1,1] = Sand_to_SH_W_orig_data[orig+1,1]
            Sand_to_SH_W[new+1,2] = Sand_to_SH_W_orig_data[orig+1,2]
            
        new = new + 1   #increment second counter variable (regardless of if data has been filled or not)
            
    
        
    
        
    
    print('*****************************************************************\n')
    print('GAPS IN EXPERIMENTAL SAND STORE HEAT EXTRACTION DATA - VERIFY solar_to_Sand_W AND solar_to_Sand_W_orig_data ARRAYS MAKE SENSE!!!!\n')
    print('*****************************************************************\n\n')
    
else:
    print('*****************************************************************\n')
    print('NO MISSING SAND STORE HEAT EXTRACTION DATA\n')
    print('*****************************************************************\n\n')
    

# %%2. Visualization Output Points

'''
This script is to get the nodes that I will record to visualize the data to check that the FD eqn is performing as expected
'''

#%%%1st SOIL LAYER - starting at y, or z = 0 (west or bottom)
#every 60 cm, including  points
node_dist = 0.6 #(m)
soil_viz_indices = np.zeros(shape=[math.floor(soil_th/node_dist)+1], dtype=int)

soil_viz_indices[0] = 0
for indx in range(1,len(soil_viz_indices)):
    prev = 100
    for i in range(0,len(Hsoil_coords_w)):
        diff = abs((indx)*node_dist - Hsoil_coords_w[i]) #diff = desired position - current index
       #this finds the closest node to the desired position
        if round(diff,5) >= round(prev,5):
            #then we want the index represented by (i-1)
            soil_viz_indices[indx] = i-1
            break
        #if next iteration would be outside the soil domain, put the last node in the soil domain there
        elif i == (len(Hsoil_coords_w)-1):
            soil_viz_indices[indx] = i
        
        prev = diff
        
    

 
#%%%NORTH SOIL LAYER 
soil_nrthside_viz_indices = np.zeros(shape=[math.floor(soil_nrthside_th/node_dist)+1], dtype=int)

soil_nrthside_viz_indices[0] = 0
for indx in range(1,len(soil_nrthside_viz_indices)):
    prev = 100
    for i in range(0,len(Hsoil_coords_n)):
        diff = abs((indx)*node_dist - Hsoil_coords_n[i]) #diff = desired position - current index
       #this finds the closest node to the desired position
        if round(diff,5) >= round(prev,5):
            #then we want the index represented by (i-1)
            soil_nrthside_viz_indices[indx] = i-1
            break
        #if next iteration would be outside the soil domain, put the last node in the soil domain there
        elif i == len(Hsoil_coords_n)-1:
            soil_nrthside_viz_indices[indx] = i
        
        prev = diff
        
    


#%%%EAST SOIL LAYER - ending at y = domain_W
soil_east_viz_indices = np.zeros(shape=[math.floor(soil_th/node_dist)+1], dtype=int)
soil_east_viz_indices[-1] = indx_bndrz_y[-1] #last index in east soil 
c = 0
for indx in range(len(soil_east_viz_indices)-2,-1,-1):
    prev = 100
    c = c+1
    d = 0
    for i in range(len(Hsoil_coords_e)-1,-1,-1):
        d = d+1
        diff = abs((W_E_layr_bounds[-1] - c*node_dist) - Hsoil_coords_e[i-1]) #diff = desired position - current index
        #this finds the closest node to the desired position
        if round(diff,5) >= round(prev,5):
            #then we want the index representing the previous node
            soil_east_viz_indices[indx] = indx_bndrz_y[-1] - d +1
            break
        elif i == 1: #If it's at the end of the loop and nothing has been chosen, choose the last node
            soil_east_viz_indices[indx] = indx_bndrz_y[-2]
        
        prev = diff
    


#%%%SOUTH SOIL LAYER - ending at x = domain_L
soil_south_viz_indices = np.zeros(shape=[math.floor(soil_th/node_dist)+1], dtype=int)
soil_south_viz_indices[-1] = indx_bndrz_x[-1] #last index in south soil 
c = 0
for indx in range(len(soil_south_viz_indices)-2,-1,-1):
    prev = 100
    c = c+1
    d = 0
    for i in range(len(Hsoil_coords_s)-1,-1,-1):
        d = d+1
        diff = abs((N_S_layr_bounds[-1] - c*node_dist) - Hsoil_coords_s[i-1]) #diff = desired position - current index
        #this finds the closest node to the desired position
        if round(diff,5) >= round(prev,5):
            #then we want the index representing the previous node
            soil_south_viz_indices[indx] = indx_bndrz_x[-1] - d +1
            break
        elif i == 1: #If it's at the end of the loop and nothing has been chosen, choose the last node
            soil_south_viz_indices[indx] = indx_bndrz_x[-2]
        
        prev = diff
    


#%%%TOP SOIL LAYER - ending at x = domain_H
#every 50 cm, including  points
node_dist = 0.5 #(m)
soil_top_viz_indices = np.zeros(shape=[math.floor(soil_top_th/node_dist)+1], dtype=int)
c = 0
soil_top_viz_indices[-1] = indx_bndrz_z[-1] #last index in top soil 
for indx in range(len(soil_top_viz_indices)-2,-1,-1):
    c = c+1
    d = 0
    prev = 100
    for i in range(len(Vsoil_coords_t)-1,-1,-1):
        d = d+1
        diff = abs((B_T_layr_bounds[-1] - c*node_dist) - Vsoil_coords_t[i-1]) #diff = desired position - current index
        #this finds the closest node to the desired position
        if round(diff,5) >= round(prev,5):
            #then we want the index representing the previous node
            soil_top_viz_indices[indx] = indx_bndrz_z[-1] - d+1
            break
        elif i == 1: #If it's at the end of the loop and nothing has been chosen, choose the last node
            soil_top_viz_indices[indx] = indx_bndrz_z[-2]
        
        prev = diff
    


#%%%1st CONCRETE LAYER (west or bottom)

if all_wall_layer_nodes==0:
    #one node in concrete layer, closest to centre of layer
    conc_wb_viz_index = 0
    prev = 100
    for i in range(0,len(Vconc_coords_b)):
        diff = abs((W_E_layr_bounds[1]+W_E_layr_bounds[2])/2 - Vconc_coords_b[i])
        #this finds the closest node to the desired position
        if round(diff,5) >= round(prev,5):
        #then we want the index representing the previous node
            conc_wb_viz_index = indx_bndrz_y[2] + (i-1)
            break 
        elif i == len(Vconc_coords_b)-1: #If it's at the end of the loop and nothing has been chosen, choose the last node
            conc_wb_viz_index = indx_bndrz_y[3]
        
        prev = diff
    
    
    #%%%North CONCRETE LAYER
    #one node in concrete layer, closest to centre of layer
    conc_n_viz_index = 0
    prev = 100
    for i in range(0,len(Hconc_coords_n)):
        diff = abs((N_S_layr_bounds[1]+N_S_layr_bounds[2])/2 - Hconc_coords_n[i])
        #this finds the closest node to the desired position
        if round(diff,5) >= round(prev,5):
        #then we want the index representing the previous node
            conc_n_viz_index = indx_bndrz_x[2] + (i-1)
            break      
        elif i == len(Hconc_coords_n)-1: #If it's at the end of the loop and nothing has been chosen, choose the last node
            conc_n_viz_index = indx_bndrz_x[3]
        
        prev = diff
    
    
    #%%%East CONCRETE LAYER
    #one node in concrete layer, closest to centre of layer
    conc_e_viz_index = 0
    prev = 100
    d=0
    for i in range(len(Hconc_coords_e)-1,-1,-1):
        d = d+1
        diff = abs((W_E_layr_bounds[5]+W_E_layr_bounds[6])/2 - Hconc_coords_e[i])
        #this finds the closest node to the desired position
        if round(diff,5) >= round(prev,5):
            #then we want the index representing the previous node
            conc_e_viz_index = indx_bndrz_y[15] - d +2
            break
        elif i == 0: #If it's at the end of the loop and nothing has been chosen, choose the last node
            conc_e_viz_index = indx_bndrz_y[14]
        
        prev = diff
    
    
    #%%%South CONCRETE LAYER
    #one node in concrete layer, closest to centre of layer
    conc_s_viz_index = 0
    prev = 100
    d=0
    for i in range(len(Hconc_coords_s)-1,-1,-1):
        d = d+1
        diff = abs((N_S_layr_bounds[5]+N_S_layr_bounds[6])/2 - Hconc_coords_s[i])
        #this finds the closest node to the desired position
        if round(diff,5) >= round(prev,5):
            #then we want the index representing the previous node
            conc_s_viz_index = indx_bndrz_x[15] - d +2
            break
        elif i == 0: #If it's at the end of the loop and nothing has been chosen, choose the last node
            conc_s_viz_index = indx_bndrz_x[14]
        
        prev = diff
    
    
    #%%%1st XPS LAYER
    #one node in insulation layer, closest to centre of layer
    XPS_wb_viz_index = 0
    prev = 100
    for i in range(0,len(VXPS_coords_b)):
        diff = abs((W_E_layr_bounds[2]+W_E_layr_bounds[3])/2 - VXPS_coords_b[i])
        #this finds the closest node to the desired position
        if round(diff,5) >= round(prev,5):
            #then we want the index representing the 
            XPS_wb_viz_index = indx_bndrz_y[4] + (i-1)
            break      
        elif i == len(VXPS_coords_b)-1: #If it's at the end of the loop and nothing has been chosen, choose the last node
            XPS_wb_viz_index = indx_bndrz_y[5]
        
        prev = diff
    
    
    #%%%North XPS LAYER
    #one node in insulation layer, closest to centre of layer
    XPS_n_viz_index = 0
    prev = 100
    for i in range(0,len(HXPS_coords_n)):
        diff = abs((N_S_layr_bounds[2] + N_S_layr_bounds[3])/2 - HXPS_coords_n[i])
        #this finds the closest node to the desired position
        if round(diff,5) >= round(prev,5):
            #then we want the index representing the 
            XPS_n_viz_index = indx_bndrz_x[4] + (i-1)
            break        
        elif i == len(HXPS_coords_n)-1: #If it's at the end of the loop and nothing has been chosen, choose the last node
            XPS_n_viz_index = indx_bndrz_x[5]
        
        prev = diff
    
    
    #%%%Top XPS LAYER
    #one node in insulation layer, closest to centre of layer
    XPS_t_viz_index = 0
    prev = 100
    d=0
    for i in range(len(VXPS_coords_t)-1,-1,-1):
        d=d+1
        diff = abs((B_T_layr_bounds[4] + B_T_layr_bounds[5])/2 - VXPS_coords_t[i])
        #this finds the closest node to the desired position
        if round(diff,5) >= round(prev,5):
            #then we want the index representing the previous node
            XPS_t_viz_index = indx_bndrz_z[9] - d +2
            break
        elif i == 0: #If it's at the end of the loop and nothing has been chosen, choose the last node
            XPS_t_viz_index = indx_bndrz_z[8]
        
        prev = diff
    
    
    #%%%East XPS LAYER
    #one node in insulation layer, closest to centre of layer
    XPS_e_viz_index = 0
    prev = 100
    d=0
    for i in range(len(HXPS_coords_e)-1,-1,-1):
        d=d+1
        diff = abs((W_E_layr_bounds[4] + W_E_layr_bounds[5])/2 - HXPS_coords_e[i])
        #this finds the closest node to the desired position
        if round(diff,5) >= round(prev,5):
            #then we want the index representing the previous node
            XPS_e_viz_index = indx_bndrz_y[13] - d +2
            break
        elif i == 0: #If it's at the end of the loop and nothing has been chosen, choose the last node
            XPS_e_viz_index = indx_bndrz_y[12]
        
        prev = diff
    
    
    #%%%South XPS LAYER
    #one node in insulation layer, closest to centre of layer
    XPS_s_viz_index = 0
    prev = 100
    d=0
    for i in range(len(HXPS_coords_s)-1,-1,-1):
        d=d+1
        diff = abs((N_S_layr_bounds[4] + N_S_layr_bounds[5])/2 - HXPS_coords_s[i])
        #this finds the closest node to the desired position
        if round(diff,5) >= round(prev,5):
            #then we want the index representing the previous node
            XPS_s_viz_index = indx_bndrz_x[13] - d +2
            break
        elif i == 0: #If it's at the end of the loop and nothing has been chosen, choose the last node
            XPS_s_viz_index = indx_bndrz_x[12]
        
        prev = diff


#%%%East SAND "no power" node
#Find the indices that correspond to the nodes in the sand store in the no TC zone
sand_noTCe_viz_index = 0 
prev = 100
d=0
for i in range(len(Hnopwr_coords_e)-1,-1,-1):
    d=d+1
    diff = abs((W_E_layr_bounds[4] + W_E_layr_bounds[4]-no_pwr)/2 - Hnopwr_coords_e[i])
    #this finds the closest node to the desired position
    if round(diff,5) >= round(prev,5):
        #then we want the index representing the previous node
        sand_noTCe_viz_index = indx_bndrz_y[11] - d + 2
        break
    elif i == 0: #If it's at the end of the loop and nothing has been chosen, choose the last node
        sand_noTCe_viz_index = indx_bndrz_y[10]
    
    prev = diff


#%%%South SAND "no power" node
#Find the indices that correspond to the nodes in the sand store in the no TC zone
sand_noTCs_viz_index = 0 
prev = 100
d=0
for i in range(len(Hnopwr_coords_s)-1,-1,-1):
    d=d+1
    diff = abs((N_S_layr_bounds[4] + N_S_layr_bounds[4]-no_pwr)/2 - Hnopwr_coords_s[i])
    #this finds the closest node to the desired position
    if round(diff,5) >= round(prev,5):
        #then we want the index representing the previous node
        sand_noTCs_viz_index = indx_bndrz_x[11] - d + 2
        break
    elif i == 0: #If it's at the  of the loop and nothing has been chosen, choose the last node
        sand_noTCs_viz_index = indx_bndrz_x[10]
    
    prev = diff


#%%%North SAND "no power" node
#Find the indices that correspond to the nodes in the sand store in the no TC zone
sand_noTCn_viz_index = 0 
prev = 100
for i in range(0,len(Hnopwr_coords_n)):
    diff = abs((N_S_layr_bounds[3] + N_S_layr_bounds[3]+no_pwr)/2 - Hnopwr_coords_n[i])
    #this finds the closest node to the desired position
    if round(diff,5) >= round(prev,5):
        #then we want the index representing the 
        sand_noTCn_viz_index = indx_bndrz_x[6] + (i-1)
        break        
    elif i == len(Hnopwr_coords_n)-1: #If it's at the  of the loop and nothing has been chosen, choose the last node
        sand_noTCn_viz_index = indx_bndrz_x[6]+ (i)
    
    prev = diff


#%%%WEST SAND "no power" node
#Find the indices that correspond to the nodes in the sand store in the no TC zone
sand_noTCw_viz_index = 0 
prev = 100
for i in range(0,len(Hnopwr_coords_w)):
    diff = abs((W_E_layr_bounds[3] + W_E_layr_bounds[3]+no_pwr)/2 - Hnopwr_coords_w[i])
    #this finds the closest node to the desired position
    if round(diff,5) >= round(prev,5):
        #then we want the index representing the 
        sand_noTCw_viz_index = indx_bndrz_y[6] + (i-1)
        break        
    elif i == len(Hnopwr_coords_n)-1: #If it's at the  of the loop and nothing has been chosen, choose the last node
        sand_noTCw_viz_index = indx_bndrz_y[6]+ i
    
    prev = diff


#%%%N-S SAND "power" nodes
node_dist = 0.5
sand_x_viz_indices = np.zeros(shape=[math.floor((sand_L-no_pwr*2)/node_dist) + 2], dtype=int) 
#If the number of nodes to be visualized is the same as or greater than the number of FDE nodes...
if len(sand_x_viz_indices) >= len(Hsand_coords_x):
    sand_x_viz_indices = np.zeros(shape=(len(Hsand_coords_x)), dtype=int)
    sand_x_viz_indices[0] = indx_bndrz_x[8]
    for i in range(1,len(sand_x_viz_indices)):
        sand_x_viz_indices[i] = sand_x_viz_indices[i-1]+1
else:
    prev = 100
    sand_x_viz_indices[0] = indx_bndrz_x[8]
    for indx in range(1,len(sand_x_viz_indices)):
        prev = 100
        for i in range(0,len(Hsand_coords_x)):
            diff = abs(N_S_layr_bounds[3]+no_pwr + indx*node_dist - Hsand_coords_x[i]) #diff = desired position - current index
            #this finds the closest node to the desired position
            if round(diff,5) >= round(prev,5):
                #then we want the index represented by (i-1)
                sand_x_viz_indices[indx] = indx_bndrz_x[8]+ i-1
                break
            #if next iteration would be outside the soil domain, put the last node in the soil domain there
            elif i == len(Hsand_coords_x)-1:
                sand_x_viz_indices[indx] = indx_bndrz_x[9]
            
            prev = diff
        
    


#%%%W-E SAND "power" nodes
sand_y_viz_indices = np.zeros(shape=[math.floor((sand_W-no_pwr*2)/node_dist) + 2], dtype=int) 
#If the number of nodes to be visualized is the same as or greater than the number of FDE nodes...
if len(sand_y_viz_indices) >= len(Hsand_coords_y):
    sand_y_viz_indices = np.zeros(shape=(len(Hsand_coords_y)), dtype=int)
    sand_y_viz_indices[0] = indx_bndrz_y[8]
    for i in range(1,len(sand_y_viz_indices)):
        sand_y_viz_indices[i] = sand_y_viz_indices[i-1]+1
else:
    prev = 100
    sand_y_viz_indices[0] = indx_bndrz_y[8]
    for indx in range(1,len(sand_y_viz_indices)):
        prev = 100
        for i in range(0,len(Hsand_coords_y)):
            diff = abs(W_E_layr_bounds[3]+no_pwr + indx*node_dist - Hsand_coords_y[i]) #diff = desired position - current index
            #this finds the closest node to the desired position
            if round(diff,5) >= round(prev,5):
                #then we want the index represented by (i-1)
                sand_y_viz_indices[indx] = indx_bndrz_y[8]+ i-1
                break
            #if next iteration would be outside the soil domain, put the last node in the soil domain there
            elif i == len(Hsand_coords_y)-1:
                sand_y_viz_indices[indx] = indx_bndrz_y[9]
            
            prev = diff
        
    


#%%%VERT SAND nodes
#for the nodes between the bottom of sand store and the "first power"
#layer,pick the one closest to centre
sand_z_viz_indices1 = 0 
prev = 100
for i in range(0,len(Vsand1)-3):
    diff = abs((B_T_layr_bounds[3] + B_T_layr_bounds[3]+ pwr_bott)/2 - Vsand1[i])
    #this finds the closest node to the desired position
    if round(diff,5) >= round(prev,5):
        #then we want the index representing the 
        sand_z_viz_indices1 = indx_bndrz_z[6] + (i-1)
        break        
    elif i == (len(Vsand1)-4): #If it's at the  of the loop and nothing has been chosen, choose the last node not in the power slab
        sand_z_viz_indices1 = indx_bndrz_z[6]+ len(Vsand1)- 4
    
    prev = diff


#for the nodes between the first power layer and the next layer, keep them 0.3m apart    
#of the vector 
node_dist = 0.3
sand_z_viz_indices2 = np.zeros(shape=[math.floor((pwr_mid-pwr_bott)/node_dist)-1], dtype=int) 
for indx in range(0,len(sand_z_viz_indices2)):
    prev = 100
    for i in range (0,(len(Vsand2)-3)):
        diff = abs(B_T_layr_bounds[3]+pwr_bott + (indx+1)*node_dist - Vsand2[i]) #diff = desired position - current index
        #this finds the closest node to the desired position
        if round(diff,5) >= round(prev,5):
            #then we want the index represented by (i-1)
            sand_z_viz_indices2[indx] = pwr_bott_indx + i+1
            break
        #if next iteration would be outside the soil domain, put the last node in the soil domain there
        elif i == len(Vsand2)-4:
            sand_z_viz_indices2[indx] = pwr_bott_indx + len(Vsand2)- 2
        
        prev = diff
        
    
#keep same distance for next layer (mid and upr pwr layers)
sand_z_viz_indices3 = np.zeros(shape=[math.floor((pwr_upr-pwr_mid)/node_dist)-1], dtype=int) 
for indx in range(0,len(sand_z_viz_indices3)):
    prev = 100
    for i in range(0,len(Vsand3)-3):
        diff = abs(B_T_layr_bounds[3]+pwr_mid+ (indx+1)*node_dist - Vsand3[i]) #diff = desired position - current index
        #this finds the closest node to the desired position
        if round(diff,5) >= round(prev,5):
            #then we want the index represented by (i-1)
            sand_z_viz_indices3[indx] = pwr_mid_indx + i+1
            break
        #if next iteration would be outside the soil domain, put the last node in the soil domain there
        elif i == len(Vsand3)-4:
            sand_z_viz_indices3[indx] = pwr_mid_indx + len(Vsand3)- 2
        
        prev = diff
        
    


#keep same distance for last layer (upr layer and top of sand store)
sand_z_viz_indices4 = np.zeros(shape=[math.floor((sand_H-pwr_upr)/node_dist)], dtype=int) 
for indx in range(0,len(sand_z_viz_indices4)):
    prev = 100
    for i in range(0,len(Vsand4)):
        diff = abs(B_T_layr_bounds[3] + pwr_upr+(indx+1)*node_dist - Vsand4[i]) #diff = desired position - current index
        #this finds the closest node to the desired position
        if round(diff,5) >= round(prev,5):
            #then we want the index represented by (i-1)
            sand_z_viz_indices4[indx] = pwr_upr_indx + i+1
            break
        #if next iteration would be outside the soil domain, put the last node in the soil domain there
        elif i == len(Vsand4)-1:
            sand_z_viz_indices4[indx] = pwr_upr_indx + len(Vsand4)+1
        
        prev = diff
        
    


# Put sand indices together
sand_z_viz_indices = np.concatenate(([indx_bndrz_z[6]],[sand_z_viz_indices1],[pwr_bott_indx],sand_z_viz_indices2,[pwr_mid_indx],sand_z_viz_indices3, [pwr_upr_indx],sand_z_viz_indices4,[indx_bndrz_z[7]]))


#%%%PUT AXIS INDICES TOGETHER

#If the nodes from the concrete and XPS layers should all be output:
if all_wall_layer_nodes == 1:
    VIZ_X_index = np.concatenate((soil_nrthside_viz_indices,np.arange(soil_nrthside_viz_indices[-1]+1,sand_noTCn_viz_index),[sand_noTCn_viz_index],sand_x_viz_indices,[sand_noTCs_viz_index],np.arange(sand_noTCs_viz_index+1,soil_south_viz_indices[0]),soil_south_viz_indices))

    VIZ_Y_index = np.concatenate((soil_viz_indices,np.arange(soil_viz_indices[-1]+1,sand_noTCw_viz_index),[sand_noTCw_viz_index],sand_y_viz_indices,[sand_noTCe_viz_index],np.arange(sand_noTCe_viz_index+1,soil_east_viz_indices[0]), soil_east_viz_indices))

    VIZ_Z_index = np.concatenate((soil_viz_indices,np.arange(soil_viz_indices[-1]+1,sand_z_viz_indices[0]),sand_z_viz_indices,np.arange(sand_z_viz_indices[-1]+1,soil_top_viz_indices[0]),soil_top_viz_indices))
#If only one node from the concrete and XPS layers should be output:
else:
    VIZ_X_index = np.concatenate((soil_nrthside_viz_indices,[conc_n_viz_index],[XPS_n_viz_index],[sand_noTCn_viz_index],sand_x_viz_indices,[sand_noTCs_viz_index],[XPS_s_viz_index],[conc_s_viz_index],soil_south_viz_indices))

    VIZ_Y_index = np.concatenate((soil_viz_indices,[conc_wb_viz_index],[XPS_wb_viz_index],[sand_noTCw_viz_index],sand_y_viz_indices,[sand_noTCe_viz_index],[XPS_e_viz_index], [conc_e_viz_index], soil_east_viz_indices))

    VIZ_Z_index = np.concatenate((soil_viz_indices,[conc_wb_viz_index],[XPS_wb_viz_index],sand_z_viz_indices,[XPS_t_viz_index],soil_top_viz_indices))


#%% 3. FDE SOLVER
#%%% Initial conditions
#%%%% Initialize Variables for FDE Solver

#Initialize temperature matrix
Temps = np.ones(shape=(U,V,W,2))     #Matrix for temperatures (two time points)

#Side wall boundaries - initialize matrices
T_SEWbound = np.zeros(shape=(len(node_zcoords)))

#in case the north wall needs to be made different b/c of its proximity to the CHEeR house, an extra variable was created
T_Nbound = np.zeros(shape=(len(node_zcoords)))

#Initial avg temperatures of 3 sand store layers
IC_SSlayers = np.zeros(shape=(3))
IC_SSlayers[0]=avg_expmtl_temps[0,0] #Layer C (bottom)
IC_SSlayers[1]=avg_expmtl_temps[0,1] #Layer B (middle)
IC_SSlayers[2]=avg_expmtl_temps[0,2] #Layer A (top)

#variable that will hold temps of nodes closest to TCs, and will be printed out for comparison with experimental data
toprintTCs = np.zeros(shape=(len(sand_TC_node_xindices), len(sand_TC_node_yindices), 3))

# #if graphs of the entire num. domain are wanted, temps for visualization will be printed out
# toprintViz = np.zeros(shape=(len(VIZ_X_index), len(VIZ_Y_index), len(VIZ_Z_index)))


#Total time elapsed in days, and hour of the day
day = 0
hour = 0

#%%%% Set Initial Conditions

#Set initial temperature conditions for the very first timestep (construct layers from outside in)
n = 0 # first timestep

#Calculate/update boundary condition temps
[T_Nbound,T_SEWbound] = boundaryConditions(DOY,day,node_zcoords, domain_H, T_Nbound, T_SEWbound)

#Initial temperature of the concrete and XPS layers -> the average of the sand store average temp and the average of the Kusuda array
XPSConc_IC = np.mean([np.mean(IC_SSlayers), np.mean(T_SEWbound)])

#Initialize LOWER Soil layers - set temperatures in LOWER soil layer
#gradient in lower soil layer
gradient_z = (XPSConc_IC - T_SEWbound[0]) / (indx_bndrz_z[1]-indx_bndrz_z[0])

for k in range(1,indx_bndrz_z[1]+1):
    Temps[:,:,k,n] = T_SEWbound[0] + gradient_z*k
    

#Concrete and XPS layers (entire x-y block from bottom of concrete to top of XPS)
Temps[:,:,indx_bndrz_z[2]:indx_bndrz_z[9]+1,0] = XPSConc_IC

#Initialize TOP Soil layers - set temperatures in TOP soil layer
#gradient in top soil layer
gradient_z = (XPSConc_IC - T_SEWbound[-1]) / (indx_bndrz_z[11]-indx_bndrz_z[10])

for k in range(indx_bndrz_z[11],indx_bndrz_z[10]-1,-1):
    Temps[:,:,k,n] = T_SEWbound[-1] + gradient_z*(indx_bndrz_z[11]-k)
    

#Initialize MIDDLE Soil layers - set temperatures in MIDDLE soil layer, at elevation of concrete and XPS
for k in range(indx_bndrz_z[2]-2,indx_bndrz_z[9]+3):

    #gradient in x-direction from north boundary wall and from south boundary wall
    gradient_x = (XPSConc_IC - T_SEWbound[k]) / (indx_bndrz_x[1]-indx_bndrz_x[0])

    for i in range(1,indx_bndrz_x[1]+1):
        #for temps nearest to north boundary wall
        Temps[i,:,k,0] = T_SEWbound[k] + gradient_x*i
        #for temps nearest to south boundary wall
        Temps[indx_bndrz_x[-1]-i,:,k,0] = T_SEWbound[k] + gradient_x*i
        
    #gradient in y-direction from west boundary wal and from east boundary walll
    gradient_y = (XPSConc_IC - T_SEWbound[k]) / (indx_bndrz_y[1]-indx_bndrz_y[0])

    for j in range(1,indx_bndrz_y[1]+1):
        #for temps nearest to west boundary wall
        Temps[indx_bndrz_x[2]-2:indx_bndrz_x[15]+3,j,k,0] = T_SEWbound[k] + gradient_y*j
        #for temps nearest to east boundary wall
        Temps[indx_bndrz_x[2]-2:indx_bndrz_x[15]+3,indx_bndrz_y[-1]-j,k,0] = T_SEWbound[k] + gradient_y*j
  
#Apply BCs again
updateBCs(T_Nbound, T_SEWbound,0)

#1st Sand layer (0m to 0.9m from bottom of sand store)
for k in range(indx_bndrz_z[6],moisture_layer_height[0]+1):
    for i in range(indx_bndrz_x[6],indx_bndrz_x[11]+1):
        for j in range(indx_bndrz_y[6],indx_bndrz_y[11]+1):
            Temps[i,j,k,n] = IC_SSlayers[0]
        

#2nd Sand layer (0.9m to 1.8m from bottom of sand store)
for k in range((moisture_layer_height[0]+1),moisture_layer_height[1]+1):
    for i in range(indx_bndrz_x[6],indx_bndrz_x[11]+1):
        for j in range(indx_bndrz_y[6],indx_bndrz_y[11]+1):
            Temps[i,j,k,n] = IC_SSlayers[1]
        

#3rd Sand layer (1.8m to 3.0m from bottom of sand store)
for k in range((moisture_layer_height[1]+1),moisture_layer_height[2]+1):
    for i in range(indx_bndrz_x[6],indx_bndrz_x[11]+1):
        for j in range(indx_bndrz_y[6],indx_bndrz_y[11]+1):
            Temps[i,j,k,n] = IC_SSlayers[2]
              
# #Copy the initial boundary conditions to next timestep (since now they are "ones" and they are 
# #only updated every week in the simulation)
# updateBCs(T_Nbound, T_SEWbound,1)

#Copy the initial conditions to next timestep (since the IC_SS function will run next, and doesn't iterate
#over the middle of the sand store)
Temps[:,:,:,1] = Temps[:,:,:,0]    



#%%%% Print initial conditions to file

#Copy temperature values that need to be output to compare to experimental data
for i in range(0,np.size(toprintTCs,0)):
    for j in range(0,np.size(toprintTCs,1)):
        toprintTCs[i,j,0] = Temps[sand_TC_node_xindices[i],sand_TC_node_yindices[j],pwr_bott_indx,n]
        toprintTCs[i,j,1] = Temps[sand_TC_node_xindices[i],sand_TC_node_yindices[j],pwr_mid_indx,n]
        toprintTCs[i,j,2] = Temps[sand_TC_node_xindices[i],sand_TC_node_yindices[j],pwr_upr_indx,n]
    


# #Copy intial temperature values to be visualized
# for i in range(0,np.size(toprintViz,0)):
#     for j in range(0,np.size(toprintViz,1)):
#         for k in range(0,np.size(toprintViz,2)):
#             toprintViz[i,j,k] = Temps[VIZ_X_index[i], VIZ_Y_index[j], VIZ_Z_index[k],n]


#Output the first timestep of data (initial condition) to output files
#This file will hold temperature values that need to be compared to experimental data
fileID = 'r' + str(run_num) + '_1Sand_TC_Temps.txt'
print2newfile(fileID,day,hour,toprintTCs,len(sand_TC_node_xindices),len(sand_TC_node_yindices),3)
   

# #This file will hold temperature values that will be visualized
# fileID2='r' + str(run_num) + '_2Sand_Viz_Temps.txt'
# temparray = np.concatenate(([day], [hour], toprintViz.flatten(order='F')))
# print2newfile(fileID2,day,hour,toprintViz,len(VIZ_X_index),len(VIZ_Y_index),len(VIZ_Z_index),True)



#%%% Run function to get initial conditions to steady state (pre-conditioning period)
#create variable to exit the solver in case FDM model becomes unstable
get_out=0

#iterations variable, nn
nn=0
diff_tol = IC_SS(dt) #FUNCTION

print("IC_SS iterations: nn = " + str(nn) + ", diff = " + str(diff_tol))

#%%%% Print steady state initial conditions to file
if get_out==0: #if previous loop had get_out==1, skip this
    
    n=0
    #Copy temperature values that need to be output to compare to experimental data
    for i in range(0,np.size(toprintTCs,0)):
        for j in range(0,np.size(toprintTCs,1)):
            toprintTCs[i,j,0] = Temps[sand_TC_node_xindices[i],sand_TC_node_yindices[j],pwr_bott_indx,n]
            toprintTCs[i,j,1] = Temps[sand_TC_node_xindices[i],sand_TC_node_yindices[j],pwr_mid_indx,n]
            toprintTCs[i,j,2] = Temps[sand_TC_node_xindices[i],sand_TC_node_yindices[j],pwr_upr_indx,n]
        
    
    
    # #Copy intial temperature values to be visualized
    # for i in range(0,np.size(toprintViz,0)):
    #     for j in range(0,np.size(toprintViz,1)):
    #         for k in range(0,np.size(toprintViz,2)):
    #             toprintViz[i,j,k] = Temps[VIZ_X_index[i], VIZ_Y_index[j], VIZ_Z_index[k],n]
    
    #Output the data with a fake hour value of 0.001h (something close to zero, since we already have values output at t=0s)
    append2file(fileID, day, 0.001, toprintTCs)
    # append2file(fileID2, day, 0.001, toprintViz)

#%%% Solver
for n in range(1,totTimeIntrvls):
    if get_out==1:  #if previous loop had get_out==1
        break
    
    #Total time elapsed in days, and hour of the day
    day = n*dt / 86400
    hour = (day - math.floor(day))*24
    
    #%%%% Boundary Conditions Update
    if day%7 == 0: #If modulus (remainder) of division of day by 7 is zero
        #Calculate/update boundary condition temps every week
        [T_Nbound,T_SEWbound] = boundaryConditions(DOY,day,node_zcoords,domain_H, T_Nbound, T_SEWbound)
        
        #Apply newly calculated B.C.'s to boundary temps
        updateBCs(T_Nbound, T_SEWbound,1)
        
    
    
    #%%%% Governing Eqn 
    GoverningEqn(dt,n)

    
    #%%%% Print temperatures to output file at requested timestep interval 
    if n%output_temp_freq_int == 0: #checking modulus (remainder) of operation
        #first check if model is stable, if it's not, exit simulation
        if np.max(Temps)>1000 or np.min(Temps)<-1000 or np.isinf(Temps).sum()>0 or np.isnan(Temps).sum()>0:
            get_out=1
            print("Exited for loop - unstable FDM model")
            break
        
        #Copy temperature values that need to be output to compare to experimental data
        for i in range(0,np.size(toprintTCs,0)):
            for j in range(0,np.size(toprintTCs,1)):
                toprintTCs[i,j,0] = Temps[sand_TC_node_xindices[i],sand_TC_node_yindices[j],pwr_bott_indx,1]
                toprintTCs[i,j,1] = Temps[sand_TC_node_xindices[i],sand_TC_node_yindices[j],pwr_mid_indx,1]
                toprintTCs[i,j,2] = Temps[sand_TC_node_xindices[i],sand_TC_node_yindices[j],pwr_upr_indx,1]
        #write temperatures from that timestep to output file
        append2file(fileID, day, hour, toprintTCs)
    
    #print out last timestep of simulation     
    elif n == (totTimeIntrvls - 1):    
        #Copy temperature values that need to be output to compare to experimental data
        for i in range(0,np.size(toprintTCs,0)):
            for j in range(0,np.size(toprintTCs,1)):
                toprintTCs[i,j,0] = Temps[sand_TC_node_xindices[i],sand_TC_node_yindices[j],pwr_bott_indx,1]
                toprintTCs[i,j,1] = Temps[sand_TC_node_xindices[i],sand_TC_node_yindices[j],pwr_mid_indx,1]
                toprintTCs[i,j,2] = Temps[sand_TC_node_xindices[i],sand_TC_node_yindices[j],pwr_upr_indx,1]
        #write temperatures from that timestep to output file
        append2file(fileID, day, hour, toprintTCs)
            
            
    # #save variables to output viz file at the requested timestep interval
    # if (n%viz_output_freq_int == 0 and n!=0): #modulus
    #     #Print out temperature values that will be visualized
    #     for i in range(0,np.size(toprintViz,0)):
    #         for j in range(0,np.size(toprintViz,1)):
    #             for k in range(0,np.size(toprintViz,2)):
    #                 toprintViz[i,j,k] = Temps[VIZ_X_index[i], VIZ_Y_index[j], VIZ_Z_index[k],1]
    #     #print temps to file
    #     append2file(fileID2, day, hour, toprintViz)
        
    # elif n == (totTimeIntrvls - 1): #print out last timestep 
    #     #Print out temperature values that will be visualized
    #     for i in range(0,np.size(toprintViz,0)):
    #         for j in range(0,np.size(toprintViz,1)):
    #             for k in range(0,np.size(toprintViz,2)):
    #                 toprintViz[i,j,k] = Temps[VIZ_X_index[i], VIZ_Y_index[j], VIZ_Z_index[k],1]
    #     #print temps to file
    #     append2file(fileID2, day, hour, toprintViz)
        
#%%%%Prepare for next iteration
    #Put temps from just-calculated timestep into "previous" timestep
    Temps[:,:,:,0]= Temps[:,:,:,1]

            
#%% 4. PARAMETER OUTPUTS TO FILE

if get_out ==1:
    filename ='r' + str(run_num) +'_SimParameters_MODEL UNSTABLE.txt'
    with open(filename,'w') as f:
        f.write("Run number "+ str(run_num)+ "\n\n")   
        f.write("Did not finish simulation, model was unstable. Solver loop exited.")
        
        f.write("Run number "+ str(run_num)+ "\n\n")   
        
        f.write("Number of pre-conditioning iterations: "+ str(nn)+ "\n")
        f.write("Largest temperature difference between final pre-conditioning iterations: "+ str(diff_tol)+ "\n\n") 
        
        f.write(filename1 + "\n")
        f.write(filename2 + "\n\n")
        
        if north_BC_diff == 1:
            f.write("North wall BC different? YES \n")
            f.write("T_avg_N = " + str(T_avg_N) + "\n")
            f.write("ampl_N = " + str(ampl_N) + "\n")
        else:
            f.write("North wall BC different? NO \n")
        f.write("T_avg = " + str(T_avg) + "\n")
        f.write("ampl = " + str(ampl) + "\n\n")
        
        f.write("water_Cp = "+ str(water_Cp)+ "\n")   
        f.write("water_rho = "+ str(water_rho)+ "\n")
        f.write("drySand_Cp = "+ str(drySand_Cp)+ "\n")
        f.write("drySand_rho = "+ str(drySand_rho)+ "\n")
        f.write("vol_percent = "+ str(vol_percent)+ "\n")
        f.write("sand_lambda = "+ str(sand_lambda)+ "\n")
        f.write("conc_Cp = "+ str(conc_Cp)+ "\n")
        f.write("conc_rho = "+ str(conc_rho)+ "\n")
        f.write("conc_lambda = "+ str(conc_lambda)+ "\n")
        
        f.write("XPS_Cp_orig = "+ str(XPS_Cp_orig)+ "\n")
        f.write("XPS_rho_orig = "+ str(XPS_rho_orig)+ "\n")
        f.write("XPS_lambda_SET = "+ str(XPS_lambda_SET)+ "\n")
        
        f.write("vol_percent_H2O_SETface_XPS = "+ str(vol_percent_H2O_SETface_XPS)+ "\n")
        f.write("XPS_Cp_SET = "+ str(XPS_Cp_SET)+ "\n")
        f.write("XPS_rho_SET = "+ str(XPS_rho_SET)+ "\n")
        f.write("vol_percent_H2O_Nface_XPS = "+ str(vol_percent_H2O_Nface_XPS)+ "\n")
        f.write("northXPS_lambda = "+ str(northXPS_lambda)+ "\n")
        f.write("vol_percent_H2O_Wface_XPS = "+ str(vol_percent_H2O_Wface_XPS)+ "\n")
        f.write("westXPS_lambda = "+ str(westXPS_lambda)+ "\n")
        f.write("vol_percent_H2O_Bface_XPS = "+ str(vol_percent_H2O_Bface_XPS)+ "\n")
        f.write("bottomXPS_lambda = "+ str(bottomXPS_lambda)+ "\n")
        f.write("XPS Cp, north wall = "+ str([Cp[indx_bndrz_x[4],indx_bndrz_y[4],indx_bndrz_z[4]],
                                              Cp[indx_bndrz_x[4],indx_bndrz_y[4],(moisture_layer_height[0]+1)],
                                              Cp[indx_bndrz_x[4],indx_bndrz_y[4],(moisture_layer_height[1]+1)]]) + "\n")
        f.write("XPS rho, north wall = "+ str([rho[indx_bndrz_x[4],indx_bndrz_y[4],indx_bndrz_z[4]],
                                              rho[indx_bndrz_x[4],indx_bndrz_y[4],(moisture_layer_height[0]+1)],
                                              rho[indx_bndrz_x[4],indx_bndrz_y[4],(moisture_layer_height[1]+1)]]) + "\n")
        f.write("XPS Cp, west wall = "+ str([Cp[indx_bndrz_x[6],indx_bndrz_y[4],indx_bndrz_z[4]],
                                              Cp[indx_bndrz_x[6],indx_bndrz_y[4],(moisture_layer_height[0]+1)],
                                              Cp[indx_bndrz_x[6],indx_bndrz_y[4],(moisture_layer_height[1]+1)]]) + "\n")
        f.write("XPS rho, west wall = "+ str([rho[indx_bndrz_x[6],indx_bndrz_y[4],indx_bndrz_z[4]],
                                              rho[indx_bndrz_x[6],indx_bndrz_y[4],(moisture_layer_height[0]+1)],
                                              rho[indx_bndrz_x[6],indx_bndrz_y[4],(moisture_layer_height[1]+1)]]) + "\n")
        f.write("XPS Cp, bottom face = "+ str(Cp[indx_bndrz_x[6],indx_bndrz_y[6],indx_bndrz_z[4]])+ "\n")
        f.write("XPS rho, bottom face = "+ str(rho[indx_bndrz_x[6],indx_bndrz_y[6],indx_bndrz_z[4]]) + "\n")
 
        f.write("soil_Cp = "+ str(soil_Cp)+ "\n")
        f.write("soil_rho = "+ str(soil_rho)+ "\n")
        f.write("soil_lambda = "+ str(soil_lambda) + "\n\n")
        
        f.write("simulation timestep, dt = "+ str(dt)+ "\n")
        f.write("output_temp_freq_h = "+ str(output_temp_freq_h)+ "\n")
        f.write("visualization_output_freq_h = "+ str(visualization_output_freq_h) +"\n\n")
    
        f.write("sand_L = "+ str(sand_L)+ "\n")
        f.write("sand_W = "+ str(sand_W)+ "\n")
        f.write("sand_H = "+ str(sand_H)+ "\n")
        f.write("XPS_th = "+ str(XPS_th)+ "\n")
        f.write("conc_th = "+ str(conc_th)+ "\n")
        f.write("soil_top_th = "+ str(soil_top_th)+ "\n")
        f.write("soil_nrthside_th = "+ str(soil_nrthside_th)+ "\n")
        f.write("soil_th = "+ str(soil_th)+ "\n")
        f.write("no_TC = "+ str(no_TC)+ "\n")
        f.write("node_space = "+ str(node_space)+ "\n")
        f.write("no_pwr = "+ str(no_pwr)+ "\n")
        f.write("pwr_bott = "+ str(pwr_bott)+ "\n")
        f.write("pwr_mid = "+ str(pwr_mid)+ "\n")
        f.write("pwr_upr = "+ str(pwr_upr)+ "\n\n")
        
        f.write("------------"+ "\n")
        f.write("MESH SIZING:"+ "\n")
        f.write("------------"+ "\n")
        f.write("Soil(coarse) dx,dy,dz = " + str(dx_soil) + "m" + "\n")
        f.write("Soil(fine) dx,dy,dz = " + str(dx_soilconc_boundary)+ "m on " + str(soilconc_boundary_th)+  "m"+ "\n")
        f.write("Concrete dx,dy,dz = "+ str(Q3) + "m"+ "\n")
        f.write("XPS dx,dy,dz = "+ str(Q4) + "m"+ "\n")
        f.write("Sand (no power zone) dx = "+ str(Q5) + "m"+ "\n")
        f.write("Sand (power zone) dx,dy = "+ str(dx_sand) + "m\n"+ "\n")
        
        f.write("Sand dz on 0<z<0.4m of sand store = "+ str(dz1) + "m"+ "\n")
        f.write("Sand dz on 0.4m<z<1.3m of sand store  = "+ str(dz2) + "m"+ "\n")
        f.write("Sand dz on 1.3m<z<2.2m of sand store  = "+ str(dz3) + "m"+ "\n")
        f.write("Sand dz on 2.2m<z<3.0m of sand store  = "+ str(dz4) + "m"+ "\n")

elif get_out != 1:    #if model was stable, print file

    filename ='r' + str(run_num) +'_SimParameters.txt'
    
    with open(filename,'w') as f:
        f.write("Run number "+ str(run_num)+ "\n\n")   
        
        f.write("Number of pre-conditioning iterations: "+ str(nn)+ "\n")
        f.write("Largest temperature difference between final pre-conditioning iterations: "+ str(diff_tol)+ "\n\n") 
        
        f.write(filename1 + "\n")
        f.write(filename2 + "\n\n")
        
        if north_BC_diff == 1:
            f.write("North wall BC different? YES \n")
            f.write("T_avg_N = " + str(T_avg_N) + "\n")
            f.write("ampl_N = " + str(ampl_N) + "\n")
        else:
            f.write("North wall BC different? NO \n")
        f.write("T_avg = " + str(T_avg) + "\n")
        f.write("ampl = " + str(ampl) + "\n\n")
        
        f.write("water_Cp = "+ str(water_Cp)+ "\n")   
        f.write("water_rho = "+ str(water_rho)+ "\n")
        f.write("drySand_Cp = "+ str(drySand_Cp)+ "\n")
        f.write("drySand_rho = "+ str(drySand_rho)+ "\n")
        f.write("vol_percent = "+ str(vol_percent)+ "\n")
        f.write("sand_lambda = "+ str(sand_lambda)+ "\n")
        f.write("conc_Cp = "+ str(conc_Cp)+ "\n")
        f.write("conc_rho = "+ str(conc_rho)+ "\n")
        f.write("conc_lambda = "+ str(conc_lambda)+ "\n")
        f.write("XPS_Cp_orig = "+ str(XPS_Cp_orig)+ "\n")
        f.write("XPS_rho_orig = "+ str(XPS_rho_orig)+ "\n")
        f.write("XPS_lambda_SET = "+ str(XPS_lambda_SET)+ "\n")
        
        f.write("vol_percent_H2O_SETface_XPS = "+ str(vol_percent_H2O_SETface_XPS)+ "\n")
        f.write("XPS_Cp_SET = "+ str(XPS_Cp_SET)+ "\n")
        f.write("XPS_rho_SET = "+ str(XPS_rho_SET)+ "\n")
        f.write("vol_percent_H2O_Nface_XPS = "+ str(vol_percent_H2O_Nface_XPS)+ "\n")
        f.write("northXPS_lambda = "+ str(northXPS_lambda)+ "\n")
        f.write("vol_percent_H2O_Wface_XPS = "+ str(vol_percent_H2O_Wface_XPS)+ "\n")
        f.write("westXPS_lambda = "+ str(westXPS_lambda)+ "\n")
        f.write("vol_percent_H2O_Bface_XPS = "+ str(vol_percent_H2O_Bface_XPS)+ "\n")
        f.write("bottomXPS_lambda = "+ str(bottomXPS_lambda)+ "\n")
        f.write("XPS Cp, north wall = "+ str([Cp[indx_bndrz_x[4],indx_bndrz_y[4],indx_bndrz_z[4]],
                                              Cp[indx_bndrz_x[4],indx_bndrz_y[4],(moisture_layer_height[0]+1)],
                                              Cp[indx_bndrz_x[4],indx_bndrz_y[4],(moisture_layer_height[1]+1)]]) + "\n")
        f.write("XPS rho, north wall = "+ str([rho[indx_bndrz_x[4],indx_bndrz_y[4],indx_bndrz_z[4]],
                                              rho[indx_bndrz_x[4],indx_bndrz_y[4],(moisture_layer_height[0]+1)],
                                              rho[indx_bndrz_x[4],indx_bndrz_y[4],(moisture_layer_height[1]+1)]]) + "\n")
        f.write("XPS Cp, west wall = "+ str([Cp[indx_bndrz_x[6],indx_bndrz_y[4],indx_bndrz_z[4]],
                                              Cp[indx_bndrz_x[6],indx_bndrz_y[4],(moisture_layer_height[0]+1)],
                                              Cp[indx_bndrz_x[6],indx_bndrz_y[4],(moisture_layer_height[1]+1)]]) + "\n")
        f.write("XPS rho, west wall = "+ str([rho[indx_bndrz_x[6],indx_bndrz_y[4],indx_bndrz_z[4]],
                                              rho[indx_bndrz_x[6],indx_bndrz_y[4],(moisture_layer_height[0]+1)],
                                              rho[indx_bndrz_x[6],indx_bndrz_y[4],(moisture_layer_height[1]+1)]]) + "\n")
        f.write("XPS Cp, bottom face = "+ str(Cp[indx_bndrz_x[6],indx_bndrz_y[6],indx_bndrz_z[4]])+ "\n")
        f.write("XPS rho, bottom face = "+ str(rho[indx_bndrz_x[6],indx_bndrz_y[6],indx_bndrz_z[4]]) + "\n")
 
        f.write("soil_Cp = "+ str(soil_Cp)+ "\n")
        f.write("soil_rho = "+ str(soil_rho)+ "\n")
        f.write("soil_lambda = "+ str(soil_lambda) + "\n\n")
        
        f.write("simulation timestep, dt = "+ str(dt)+ "\n")
        f.write("output_temp_freq_h = "+ str(output_temp_freq_h)+ "\n")
        f.write("visualization_output_freq_h = "+ str(visualization_output_freq_h) +"\n\n")
    
        f.write("sand_L = "+ str(sand_L)+ "\n")
        f.write("sand_W = "+ str(sand_W)+ "\n")
        f.write("sand_H = "+ str(sand_H)+ "\n")
        f.write("XPS_th = "+ str(XPS_th)+ "\n")
        f.write("conc_th = "+ str(conc_th)+ "\n")
        f.write("soil_top_th = "+ str(soil_top_th)+ "\n")
        f.write("soil_nrthside_th = "+ str(soil_nrthside_th)+ "\n")
        f.write("soil_th = "+ str(soil_th)+ "\n")
        f.write("no_TC = "+ str(no_TC)+ "\n")
        f.write("node_space = "+ str(node_space)+ "\n")
        f.write("no_pwr = "+ str(no_pwr)+ "\n")
        f.write("pwr_bott = "+ str(pwr_bott)+ "\n")
        f.write("pwr_mid = "+ str(pwr_mid)+ "\n")
        f.write("pwr_upr = "+ str(pwr_upr)+ "\n\n")
        
        f.write("------------"+ "\n")
        f.write("MESH SIZING:"+ "\n")
        f.write("------------"+ "\n")
        f.write("Soil(coarse) dx,dy,dz = " + str(dx_soil) + "m" + "\n")
        f.write("Soil(fine) dx,dy,dz = " + str(dx_soilconc_boundary)+ "m on " + str(soilconc_boundary_th)+  "m"+ "\n")
        f.write("Concrete dx,dy,dz = "+ str(Q3) + "m"+ "\n")
        f.write("XPS dx,dy,dz = "+ str(Q4) + "m"+ "\n")
        f.write("Sand (no power zone) dx = "+ str(Q5) + "m"+ "\n")
        f.write("Sand (power zone) dx,dy = "+ str(dx_sand) + "m\n"+ "\n")
        
        f.write("Sand dz on 0<z<0.4m of sand store = "+ str(dz1) + "m"+ "\n")
        f.write("Sand dz on 0.4m<z<1.3m of sand store  = "+ str(dz2) + "m"+ "\n")
        f.write("Sand dz on 1.3m<z<2.2m of sand store  = "+ str(dz3) + "m"+ "\n")
        f.write("Sand dz on 2.2m<z<3.0m of sand store  = "+ str(dz4) + "m"+ "\n")
