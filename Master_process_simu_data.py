# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 13:55:10 2021

@author: Rebecca
"""
# %% Import Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import math
import re
import os
#import xarray as xr
from datetime import datetime, timedelta
from pathlib import Path
from decimal import Decimal
#import my functions from other Python files in the same folder
import func_readSerrData as func_serr

# import pdb
# pdb.set_trace()
    
plt.close('all')

#%% DEFINE DIRECTORIES

#Define Directory Paths
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

#%% FUNCTIONS
def strToArr(comments):
    tempstr = comments.split("[",1)[1] #remove everything to the left of '['
    tempstr = tempstr.rsplit("]")[0]  #remove everything to the right of ']'
    tempstr = tempstr.split()    # split the remaining along white spaces
    temparr = np.array(tempstr)
    return temparr

def check_mod(rem,divisor):
    rem = round(rem,3)
    rem = float(rem)

    if rem == divisor:
        #if the remainder is equal to the divisor, then the remainder should really 
        #be zero, and it's a particularity of floating point arithmetic and binary numbers
        #and the modulus operator that have made it not equal to zero
        rem = 0
        
    return rem

def plotAllSndTemps(arrayname, marker='o'):
#plot all columns on y-axis, with first two columns ignored
    # x-axis is number of days starting from start of simulation (column 0 in array)
    fig1,ax1 = plt.subplots()
    ax1.set_title("Simulated sand temperatures of all TCs")
    ax1.set_xlabel("Days since start of simulation")
    ax1.set_ylabel("Temperature (\N{DEGREE SIGN}C)")
    
    for col in range(2,np.size(arrayname,1)):
        ax1.plot(arrayname[:,0],arrayname[:,col],marker)                

    
def plotTimeSeriesEntireArray(timearray,y_array, fig1, ax1, labels, colors=['g','b','r']):
#plot all columns on y-axis, with first columns as time series
    # x-axis is number of days starting from start of simulation (column 0 in array)
    
    for col in range(np.size(y_array,1)-1,-1,-1):
        #ax2.plot(timearray[:,0],y_array[:,col],marker) #Use to get x-axis "days after simulation"
        ax1.plot(timearray,y_array[:,col], label=labels[col], color = colors[col], linestyle = '--')                

    
    return fig1, ax1

def DateArrayFromSimDayHoursArray(time_in_days_array,start_date_in_datetime): 
    #create an empty list for the simulation time array
    date_array = [0]*time_in_days_array.shape[0]
    #put start date in top row of list
    date_array[0] = start_date_in_datetime
    for d in range(1,time_in_days_array.shape[0]):
        date_array[d] = date_array[d-1] + timedelta(days=(time_in_days_array[d]-time_in_days_array[d-1]))
        
    return date_array
    
def readOutputSimuDatafile(filename, plotall=False):
    
    run_number = filename.split('_',1)
    run_number = run_number[0][1:]

    #read in dimensions of array 
    with open(filename, 'r') as f:
        for line in range(0,2):
            comments = f.readline()
            if line==0:    #read in dimensions of array
                dims=comments.split('#')[1]
            elif line==1:   #read in start and end dates
                dates=comments.split('#')[1]   
        
        #Two possible headers for the output files...check the header
        if dims[0].isdigit() == False:   #If the first character after the # is a digit, then it's the old type of header
            c=0
            #find the first character that's a digit
            while (dims[c].isdigit()==False):
                c=c+1
            #get rid of everything to the left of that character    
            dims = dims[c:]
            
        [X,Y,Z] =dims.split(',')
        X=int(X)
        Y=int(Y)
        Z=int(Z)

        if dates[0].isdigit() == False:   #If the first character after the # is a digit, then it's the old type of header
            c=0
            #find the first character that's a digit
            while (dates[c].isdigit()==False):
                c=c+1
            #get rid of everything to the left of that character    
            dates = dates[c:]
            
        dates = dates.split(',')
        start_date = datetime.strptime(dates[0],'%Y-%m-%d')
        end_date = dates[1][:dates[1].rindex('\n')] #slices from the beginning to the '\n'
        # end_date = datetime.strptime(end_date,'%Y-%m-%d')
        end_date = datetime.strptime(end_date,'%Y-%m-%d')
        
            
    #Read data from simulation into holding variable C
    C = np.loadtxt(filename, delimiter=',', comments='#')
    #Gets number of rows in file
    num_rowsC = np.size(C,0)
    #extract time data from file
    timearrayout = C[:,0:2]  
    #extract other data from file and "unflatten/reshape" into an array
    mtrxOut = np.reshape(C[:,2:],(num_rowsC,X,Y,Z),order='F') 
    
    if plotall == True:
        #Plot sand Temps closest to TCs (delete this once the code works)       
        plotAllSndTemps(C,'-')

    
    return mtrxOut, timearrayout, start_date, end_date, run_number

def readOutputSimuVizDatafile(filename):
    
    run_number = filename.split('_',1)
    run_number = run_number[0][1:]
    
    #read in dimensions of array
    with open(filename, 'r') as f:
        df_out = pd.DataFrame()
        for line in range(0,11):
            comments = f.readline()
            if line==0:    #read in dimensions of array
                dims=comments.split(" ")[1]
            elif line==1:   #read in start and end dates
                dates=comments.split(" ")[1] 
            elif line==2:   #read in number of simulated days
                simu_days= int(comments.split(" ")[-1]) 
            elif line>=3:   #read in the rest of the lines into a dataframe
                df = pd.DataFrame({ line: strToArr(comments)})  
                        #put each line from 4-11 into a new dataframe (column of strings) 
                        #make the line number the column heading
                df_out = pd.concat([df_out,df], axis=1)
                        #vertically concatenate each new column to the main DataFrame, df_out

                
                
    [X,Y,Z] =dims.split(',')
    X=int(X)
    Y=int(Y)
    Z=int(Z)
    
    dates = dates.split(',')
    start_date = datetime.strptime(dates[0],'%Y-%m-%d')
    end_date = dates[1][:dates[1].rindex('\n')] #slices from the beginning to the '\n'
    end_date = datetime.strptime(end_date,'%Y-%m-%d')
      
    #Read data from simulation into holding variable C
    C = np.loadtxt(filename, delimiter=',', comments='#')
    #Gets number of rows in file
    num_rowsC = np.size(C,0)
    #extract time data from file
    timearrayout = C[:,0:2]  
    #extract other data from file and "unflatten/reshape" into an array
    mtrxOut = np.reshape(C[:,2:],(num_rowsC,X,Y,Z),order='F') 
    
    
    # xcoords = mtrxOut.shape[1]
    # ycoords = mtrxOut.shape[2]
    # zcoords = mtrxOut.shape[3]
    return mtrxOut, timearrayout, start_date, end_date, df_out[3].dropna(), df_out[4].dropna(), df_out[5].dropna(), df_out[6].dropna(), df_out[7].dropna(), df_out[8].dropna(), df_out[9].dropna(), df_out[10].dropna(), simu_days, run_number

def readSerranoPerformanceMetricsData(filename1, dat_headers=dat_headers, main_dir=main_dir, data_dir=data_dir):
    
   
    #FILENAMES OF DATA FILES     
    #(NB! filename1 MUST have a start AND end date, even if they are equal, ie. only one day of data)
       
    #Parse out dates between which to run the simulation
    #find year 1
    f1_name, f1_2, f1_3 = re.split('(_20)',filename1)

    st1 = filename1.find('_20')
    date_start = filename1[st1+1:st1+11]
    exp_start_date = datetime.date(datetime.strptime(date_start,'%Y-%m-%d'))

    end1 = filename1.find('to-')
    date_end = filename1[end1+3:-4]
    exp_end_date = datetime.date(datetime.strptime(date_end,'%Y-%m-%d'))
    
    #Read in sand temps data into a DataFrame
    Perf_metrics = func_serr.read_performance_metrics_data(filename1, dat_headers, main_dir, data_dir)
    
    return Perf_metrics, exp_start_date, exp_end_date


def readSerranoSandMultipleDates(filename1, dat_headers=dat_headers, main_dir=main_dir, data_dir=data_dir):
    
   
    #FILENAMES OF DATA FILES     
    #(NB! filename1 MUST have a start AND end date, even if they are equal, ie. only one day of data)
       
    #Parse out dates between which to run the simulation
    #find year 1
    f1_name, f1_2, f1_3 = re.split('(_20)',filename1)

    st1 = filename1.find('_20')
    date_start = filename1[st1+1:st1+11]
    exp_start_date = datetime.date(datetime.strptime(date_start,'%Y-%m-%d'))

    end1 = filename1.find('to-')
    date_end = filename1[end1+3:-4]
    exp_end_date = datetime.date(datetime.strptime(date_end,'%Y-%m-%d'))
    
    #Read in sand temps data into a DataFrame
    Exp_sand_temps = func_serr.read_sand_temps(filename1, dat_headers, main_dir, data_dir)
    
    return Exp_sand_temps, exp_start_date, exp_end_date

def readSerranoSandSingleDate(filename1, dat_headers=dat_headers, main_dir=main_dir, data_dir=data_dir):
    
    #Define Directory Paths
    #get current directory
    main_dir = Path(os.getcwd())
    #define parent directory
    parent = main_dir.parents[0]

    #define directory holding input data
    data_dir = parent.joinpath('Exp_Data_for_models')
       
    #Parse out date from filename
    f1_name, f1_2, f1_3 = re.split('(_20)',filename1)
    
    date_start = '20' + f1_3
    date_start = date_start.rstrip('.dat')
    exp_start_date = datetime.strptime(date_start,'%Y-%m-%d')
     
    
    #Read in Headers file
    dat_headers = pd.read_csv(data_dir.joinpath('Serrano_dat_headers.csv'))
    
    #Read in sand temps data into a DataFrame
    Exp_sand_temps = func_serr.read_sand_temps(filename1, dat_headers, main_dir, data_dir)
    
    return Exp_sand_temps, exp_start_date   

def processExpTCDataToArrays(exp_TC_data,exp_start_date):
    # Process Experimental Sand Temp data into Numpy arrays
    """
    This script processes the experimental data, output by Serrano, from a .dat 
    or .csv file into matrix form, so that it can be used later
    
    RECALL!!!! k = 0 is bottom of sand store, while C-layer is bottom-most plane
    """
    
    #Put time data into a new numpy array (shows the time in hours from the start of the year)
    time_h = exp_TC_data['time'].copy().to_numpy()
    
    #rows and columns of sand temp data
    [A, B] = exp_TC_data.shape
    
    #create 4-D array to hold temp data
    Exp_sand_temps_4d = np.zeros(shape=[A,6,6,3])
    
    #create loop to map expt'l data to 4-D matrix
    for indx1 in range(0,A):
        
        indx2 = 1 #index for the number of columns in the matrix. It starts at 1 since the first column is the time-stamp (zero-indexed)
        
        for k in range (2,-1,-1):
            for i in range (0,6):
                for j in range (0,6):
                    Exp_sand_temps_4d[indx1,i,j,k] = exp_TC_data.iloc[indx1,indx2]
                    
                    #increment the column to be read next time
                    indx2 = indx2 + 1
                    #check that it exists, and that it's not the end of the line
                    if indx2 >= B:
                        #break out of the innermost loop    
                        break 
                    
             
      
    #Create a time list for the experimental time array
    #create empty list
    exp_time_array = [0]*A
    #put start date (rounded down to midnight) in top row of list
    exp_time_array[0] = datetime(year=exp_start_date.year,month=1,day=1) + timedelta(days=np.floor((time_h[0]/24))) 
    #get difference between start datetime and midnight of that day
    start_offset = timedelta(days=(time_h[0]/24)) - timedelta(days=np.floor((time_h[0]/24)))
    
    start_year = exp_start_date.year
    for d in range(1,A):
        #if the next entry is less than the previous entry, and their absolute difference is many months (>5000h)
        if (((time_h[d]-time_h[d-1]) < 0) and (abs(time_h[d]-time_h[d-1]) > 8000)):
            start_year += 1
            exp_time_array[d] = datetime(year=start_year,month=1,day=1) + timedelta(days=(time_h[d]/24)) - start_offset
        else:    
            exp_time_array[d] =  datetime(year=start_year,month=1,day=1) + timedelta(days=(time_h[d]/24)) - start_offset
    
    return Exp_sand_temps_4d, exp_time_array

def plotLayersofExpSandTempData_Sbplts(exp_time_array,Exp_sand_temps):
    
    titles = ['Experimental top layer sand temperatures',
              'Experimental middle layer sand temperatures',
              'Experimental bottom layer sand temperatures']
    layer_name = ['A','B','C']
    
    for layer in range(0,3):
        fig4,axs = plt.subplots(3,2,sharex=True, sharey=True)  #6 SUBPLOTS
        fig4.suptitle(titles[layer])
        
        sbplt = 0
        #plot top layer of exp sand temps
        for a in range(0,2):
            for b in range (0,3):
                for j in range(0,6):
                    col = (sbplt*6 + j) + 36*layer
                    axs[b,a].plot(exp_time_array,Exp_sand_temps.iloc[:,col+1],label=Exp_sand_temps.columns[col+1]) #plot top layer of sand store
                    axs[b,a].grid(True)
                    axs[b,a].set_title(layer_name[layer] + '-' + str(sbplt+1) + '-X', fontsize=8)
                    axs[b,a].set_ylim(20,60)
                    # axs[b,a].legend()
                sbplt = sbplt + 1    
        fig4.legend(['X-X-1',
                     'X-X-2',
                     'X-X-3',
                     'X-X-4',
                     'X-X-5',
                     'X-X-6'],fontsize=8)
        fig4.autofmt_xdate() #automatically makes the x-labels rotate
                    # locator = mdates.AutoDateLocator()
                    # formatter = mdates.ConciseDateFormatter(locator)
        fig4.text(0.06, 0.5, 'Temperature ($\degree$C)', ha='center', va='center', rotation='vertical')
    return

def plotTopLayerofExpSandData_IndPlts(exp_time_array,Exp_sand_temps):
#this function plots 6 graphs with all the indivual TC data, 6 lines per graph, representing each row of TCs from north to south
    global exp_start_date,exp_end_date
    
    #plot top layer of exp sand temps
    for a in range(0,6):
        fig,ax = plt.subplots()
        for j in range(0,6):
            col = a*6 + j
            ax.plot(exp_time_array,Exp_sand_temps.iloc[:,col+1], label=Exp_sand_temps.columns[col+1]) #plot top layer of sand store
                
        #locator = mdates.DayLocator(interval=2)
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.legend()
        ax.grid(True)
        ax.set_title("Sand store top layer TCs: "+ str(exp_start_date) + " to " + str(exp_end_date))
        ax.set_ylabel("Temperature ($\degree$C)")
        ax.xaxis.set_major_locator(locator)   
        #ax.xaxis.set_minor_locator(mdates.DayLocator())   
        ax.xaxis.set_major_formatter(formatter)
        ax.set_ylim(20,60)
        fig.autofmt_xdate() #automatically makes the x-labels rotate
    return

def plotMiddleLayerofExpSandData_IndPlts(exp_time_array,Exp_sand_temps):
#this function plots 6 graphs with all the indivual TC data, 6 lines per graph, representing each row of TCs from north to south

    global exp_start_date,exp_end_date
    
    #plot middle layer of exp sand temps
    for a in range(0,6):
        fig,ax = plt.subplots()
        for j in range(0,6):
            col = (a*6 + j) + (36)
            ax.plot(exp_time_array,Exp_sand_temps.iloc[:,col+1], label=Exp_sand_temps.columns[col+1]) #plot top layer of sand store
                
        #locator = mdates.DayLocator(interval=2)
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.legend()
        ax.grid(True)
        ax.set_title("Sand store middle layer TCs: "+ str(exp_start_date) + " to " + str(exp_end_date))
        ax.set_ylabel("Temperature ($\degree$C)")
        ax.xaxis.set_major_locator(locator)   
        #ax.xaxis.set_minor_locator(mdates.DayLocator())   
        ax.xaxis.set_major_formatter(formatter)
        ax.set_ylim(20,60)
        fig.autofmt_xdate() #automatically makes the x-labels rotate
    return

def plotBottomLayerofExpSandData_IndPlts(exp_time_array,Exp_sand_temps):
#this function plots 6 graphs with all the indivual TC data, 6 lines per graph, representing each row of TCs from north to south
    
    global exp_start_date,exp_end_date
    
    #plot middle layer of exp sand temps
    for a in range(0,6):
        fig,ax = plt.subplots()
        for j in range(0,6):
            col = (a*6 + j) + (36*2)
            ax.plot(exp_time_array,Exp_sand_temps.iloc[:,col+1], label=Exp_sand_temps.columns[col+1]) #plot top layer of sand store
                
        #locator = mdates.DayLocator(interval=2)
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.legend()
        ax.grid(True)
        ax.set_title("Sand store middle layer TCs: "+ str(exp_start_date) + " to " + str(exp_end_date))
        ax.set_ylabel("Temperature ($\degree$C)")
        ax.xaxis.set_major_locator(locator)   
        #ax.xaxis.set_minor_locator(mdates.DayLocator())   
        ax.xaxis.set_major_formatter(formatter)
        ax.set_ylim(20,60)
        fig.autofmt_xdate() #automatically makes the x-labels rotate
    return

def plotExpSandDataAvgofEWRows_IndPlts(exp_time_array,Exp_sand_temps_4d):
#this function plots 1 graph with 6 lines of data, each line is the average TC from each row of TCs (north to south)
    
    global exp_start_date,exp_end_date
    
    TC_EW_row_avgs = np.mean(Exp_sand_temps_4d,axis=2)
    # layer_name = ["Bottom",
    #               "Middle",
    #               "Top"]
    layer_name = [", Level C",
                  ", Level B",
                  ", Level A"]
    
    #plot top layer of exp sand temps
    for layer in range(2,-1,-1):
        fig,ax = plt.subplots()
        lgnd = ["Row 1, north edge" + layer_name[layer],
                    "Row 2"+ layer_name[layer],
                    "Row 3"+ layer_name[layer],
                    "Row 4"+ layer_name[layer],
                    "Row 5"+ layer_name[layer],
                    "Row 6, south edge"+ layer_name[layer]]
        lines = ["--",
                 "dotted",
                 "-",
                 "-",
                 "-",
                 "-"]
        for j in range(0,6):
            
            ax.plot(exp_time_array,TC_EW_row_avgs[:,j,layer], label=lgnd[j], linestyle=lines[j]) #plot top layer of sand store
                
        #locator = mdates.DayLocator(interval=2)
        # locator = mdates.AutoDateLocator()
        locator = mdates.MonthLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.legend()
        # ax.legend([layer_name[layer] +" Row 1 (north)",
        #             layer_name[layer] +" Row 2",
        #             layer_name[layer] +" Row 3",
        #             layer_name[layer] +" Row 4",
        #             layer_name[layer] +" Row 5",
        #             layer_name[layer] +" Row 6 (south)"])
        # ax.legend(["Row 1 (northern edge" + layer_name[layer],
        #            "Row 2",
        #             "Row 3",
        #             "Row 4",
        #             "Row 5",
        #             "Row 6 (southern edge"+ layer_name[layer]])          
        ax.grid(True)
        #ax.set_title("Sand store "+ layer_name[layer] +" layer, average TC row temps: "+ str(exp_start_date) + " to " + str(exp_end_date))
        ax.set_ylabel("Temperature ($\degree$C)")
        ax.xaxis.set_major_locator(locator)   
       # ax.xaxis.set_minor_locator(mdates.MonthLocator())   
        ax.xaxis.set_major_formatter(formatter)
        ax.set_ylim(20,60)
        fig.autofmt_xdate() #automatically makes the x-labels rotate
    return

def plotExpSandDataAvgofNSRows_IndPlts(exp_time_array,Exp_sand_temps_4d):
#this function plots 1 graph with 6 lines of data, each line is the average TC from each row of TCs (north to south)
    
    global exp_start_date,exp_end_date
    
    TC_NS_row_avgs = np.mean(Exp_sand_temps_4d,axis=1)
    # layer_name = ["Bottom",
    #               "Middle",
    #               "Top"]
    layer_name = [", Level C",
                  ", Level B",
                  ", Level A"]
    
    #plot top layer of exp sand temps
    for layer in range(2,-1,-1):
        fig,ax = plt.subplots()
        lgnd = ["Col 1, west edge" + layer_name[layer],
                   "Col 2"+ layer_name[layer],
                    "Col 3"+ layer_name[layer],
                    "Col 4"+ layer_name[layer],
                    "Col 5"+ layer_name[layer],
                    "Col 6, east edge"+ layer_name[layer]]
        lines = ["--",
                 "dotted",
                 "-",
                 "-",
                 "-",
                 "-"]
        for j in range(0,6):
            ax.plot(exp_time_array,TC_NS_row_avgs[:,j,layer], label=lgnd[j], linestyle = lines[j]) #plot top layer of sand store
                
        #locator = mdates.DayLocator(interval=2)
        locator = mdates.AutoDateLocator()
        # locator = mdates.MonthLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        # ax.legend([layer_name[layer] + " Column 1 (west)",
        #            layer_name[layer] + " Column 2",
        #             layer_name[layer] + " Column 3",
        #             layer_name[layer] + " Column 4",
        #             layer_name[layer] + " Column 5",
        #             layer_name[layer] + " Column 6 (east)"])
        ax.legend()        
        ax.grid(True)
        #ax.set_title("Sand store "+ layer_name[layer] +" layer, average TC row temps: "+ str(exp_start_date) + " to " + str(exp_end_date))
        ax.set_ylabel("Temperature ($\degree$C)")
        ax.xaxis.set_major_locator(locator)   
        #ax.xaxis.set_minor_locator(mdates.DayLocator())   
        ax.xaxis.set_major_formatter(formatter)
        ax.set_ylim(20,60)
        fig.autofmt_xdate() #automatically makes the x-labels rotate
    return


def plotExpSandDataXSections(Exp_sand_temps_4d,min_ylim,max_ylim):
    
    sand_xcoords = [0.5 ,1.5 ,2.5 ,3.5 ,4.5 ,5.5]
    sand_ycoords =  [0.5 ,1.5 ,2.5 ,3.5 ,4.5 ,5.5]
    #plot x-section of all exp sand temp layers
    #TOP LAYER
    for j in range(0,6):
        fig,ax = plt.subplots()
        for time in range(0, Exp_sand_temps_4d.shape[0],288): #<- every 288 intervals (5-minutes each) is 1 day
            ax.plot(sand_xcoords,Exp_sand_temps_4d[time,:,j,2],label="Day "+str(int(time/288))) #plot k=2 (top of sand store)
                            
        ax.grid(True)
        ax.set_title("Expt'l data: Cross-section of top-layer Sand Store TCs, at y=" + str(sand_ycoords[j]) + "m and z=2.2m")
        ax.set_ylim(min_ylim,max_ylim)
        ax.set_ylabel("Temperature ($\degree$C)")
        ax.set_xlabel("Sand store x-coordinate (m), north wall XPS/sand boundary = 0m ")
        ax.legend()
    
    #MIDDLE LAYER
    for j in range(0,6):
        fig,ax = plt.subplots()
        for time in range(0, Exp_sand_temps_4d.shape[0],288): #<- every 288 intervals (5-minutes each) is 1 day
            ax.plot(sand_xcoords,Exp_sand_temps_4d[time,:,j,1],label="Day "+str(int(time/288))) #plot k=1 (middle layer of sand store)
                            
        ax.grid(True)
        ax.set_title("Expt'l data: Cross-section of middle-layer Sand Store TCs, at y=" + str(sand_ycoords[j]) + "m and z=1.3m")
        ax.set_ylim(min_ylim,max_ylim)
        ax.set_ylabel("Temperature ($\degree$C)")
        ax.set_xlabel("Sand store x-coordinate (m), where north wall XPS/sand boundary = 0m ")
        ax.legend()
    
    #BOTTOM LAYER
    for j in range(0,6):
        fig,ax = plt.subplots()
        for time in range(0, Exp_sand_temps_4d.shape[0],288): #<- every 288 intervals (5-minutes each) is 1 day
            ax.plot(sand_xcoords,Exp_sand_temps_4d[time,:,j,0],label="Day "+str(int(time/288))) #plot k=0 (bottom layer of sand store)
                            
        ax.grid(True)
        ax.set_title("Expt'l data: Cross-section of bottom-layer Sand Store TCs, at y=" + str(sand_ycoords[j]) + "m and z=0.4m")
        ax.set_ylim(min_ylim,max_ylim)
        ax.set_ylabel("Temperature ($\degree$C)")
        ax.set_xlabel("Sand store x-coordinate (m), where north wall XPS/sand boundary = 0m ")
        ax.legend()
    
    return

def plotExpSandDataYSections(Exp_sand_temps_4d,min_ylim,max_ylim):
    
     sand_xcoords = [0.5 ,1.5 ,2.5 ,3.5 ,4.5 ,5.5]
     sand_ycoords =  [0.5 ,1.5 ,2.5 ,3.5 ,4.5 ,5.5]
     #plot y-section of all exp sand temp layers
     #TOP LAYER
     for i in range(0,6):
         fig,ax = plt.subplots()
         for time in range(0, Exp_sand_temps_4d.shape[0],288): #<- every 288 intervals (5-minutes each) is 1 day
             ax.plot(sand_ycoords,Exp_sand_temps_4d[time,i,:,2],label="Day "+str(int(time/288))) #plot k=2 (top of sand store)
                             
         ax.grid(True)
         ax.set_title("Expt'l data: Cross-section of top-layer Sand Store TCs, at x=" + str(sand_xcoords[i]) + "m and z=2.2m")
         ax.set_ylim(min_ylim,max_ylim)
         ax.set_ylabel("Temperature ($\degree$C)")
         ax.set_xlabel("Sand store y-coordinate (m), west wall XPS/sand boundary = 0m ")
         ax.legend()
     
     #MIDDLE LAYER
     for i in range(0,6):
         fig,ax = plt.subplots()
         for time in range(0, Exp_sand_temps_4d.shape[0],288): #<- every 288 intervals (5-minutes each) is 1 day
             ax.plot(sand_ycoords,Exp_sand_temps_4d[time,i,:,1],label="Day "+str(int(time/288))) #plot k=1 (middle layer of sand store)
                             
         ax.grid(True)
         ax.set_title("Expt'l data: Cross-section of middle-layer Sand Store TCs, at x=" + str(sand_xcoords[i]) + "m and z=1.3m")
         ax.set_ylim(min_ylim,max_ylim)
         ax.set_ylabel("Temperature ($\degree$C)")
         ax.set_xlabel("Sand store y-coordinate (m), where west wall XPS/sand boundary = 0m ")
         ax.legend()
     
     #BOTTOM LAYER
     for i in range(0,6):
         fig,ax = plt.subplots()
         for time in range(0, Exp_sand_temps_4d.shape[0],288): #<- every 288 intervals (5-minutes each) is 1 day
             ax.plot(sand_ycoords,Exp_sand_temps_4d[time,i,:,0],label="Day "+str(int(time/288))) #plot k=0 (bottom layer of sand store)
                             
         ax.grid(True)
         ax.set_title("Expt'l data: Cross-section of bottom-layer Sand Store TCs, at x=" + str(sand_xcoords[i]) + "m and z=0.4m")
         ax.set_ylim(min_ylim,max_ylim)
         ax.set_ylabel("Temperature ($\degree$C)")
         ax.set_xlabel("Sand store y-coordinate (m), where west wall XPS/sand boundary = 0m ")
         ax.legend()
         
     return


def PlotMaxandMean(num_mod_temps,num_mod_time, run_num):
    
    # Plot max temp over time
    
    #initialize array
    max_temps = np.zeros(shape=(num_mod_time.shape[0]))
    
    #put max temps at each timestep in the new array
    for time in range(0,num_mod_time.shape[0]):
        max_temps[time] = np.max(num_mod_temps[time,:,:,:])
        
    #Plot the max temps
    fig3,ax3 = plt.subplots() 
    ax3.plot(num_mod_time[:,0],max_temps, label = 'max')
   
    # Plot mean temp over time
    #initialize array
    mean_temps = np.zeros(shape=(num_mod_time.shape[0]))
    
    #put max temps at each timestep in the new array
    for time in range(0,num_mod_time.shape[0]):
        mean_temps[time] = np.mean(num_mod_temps[time,:,:,:])
        
  
    ax3.plot(num_mod_time[:,0],mean_temps, label ='mean')
        
    ax3.set_title("Run "+ run_num + ": Max and Mean temps during cooldown")
    ax3.set_xlabel('Time (days)')
    ax3.set_ylabel("Temperature (\N{DEGREE SIGN}C)")   
    ax3.legend()
    return max_temps,mean_temps

def sim_round2hr(dtm, Nmin):
    #this function rounds a datetime to the nearest 60 minutes, based on user input.

    rounding = timedelta(minutes=dtm.minute % Nmin,
                               seconds=dtm.second,
                               microseconds=dtm.microsecond)
    
    dtm = dtm - rounding
    
    #if needed, round up
    if rounding >= timedelta(minutes=Nmin/2):
        dtm = dtm + timedelta(minutes=Nmin)

    return dtm



def exp_round2hr():
    #this function rounds a datetime to the nearest 60 minutes, based on user input.
    #It also puts the corresponding average experimental temps into another array, avg_expmtl_temps4resid
    
    global exp_time_array, sndTemps_datetimes_rnd, exp_time_array_rn
    global avg_expmtl_temps4resid, Exp_sand_temps4resid108, avg_expmtl_temps
    
    
    
    #first data point in exp arrays
    exp_time_array_rnd[0] = exp_time_array[0]
    avg_expmtl_temps4resid[0,:] = avg_expmtl_temps[0,:]
    Exp_sand_temps4resid108 [0,:,:,:] = Exp_sand_temps_4d[0,:,:,:]
    
    idx = 1
    #now, for the rest of the array
    for u in range(2,len(sndTemps_datetimes_rnd)):
        for v in range(idx,len(exp_time_array)):
            #for the very last value in the array
            if u == (len(sndTemps_datetimes_rnd) - 1):
                if abs(exp_time_array[-1] - sndTemps_datetimes_rnd[-1]) <= timedelta(minutes=15):
                    exp_time_array_rnd[-1] = exp_time_array[-1]
                    avg_expmtl_temps4resid[-1,:] = avg_expmtl_temps[-1,:]
                    Exp_sand_temps4resid108 [-1,:,:,:] =  Exp_sand_temps_4d[-1,:,:,:]
                    
            #as the loop is going through all the times in the expmt'l array, is it less than 15min on either side of the hour?
            elif abs(exp_time_array[v] - sndTemps_datetimes_rnd[u]) <= timedelta(minutes=15):
                #assign difference between experimental timestamp and simultd timestamp
                curr_ent = abs(exp_time_array[v] - sndTemps_datetimes_rnd[u])
                next_ent = abs(exp_time_array[v+1] - sndTemps_datetimes_rnd[u])
                if next_ent > curr_ent:
                    #we want the current v to be stored in the RMSD array
                    exp_time_array_rnd[u-1] = exp_time_array[v]
                    avg_expmtl_temps4resid[u-1,:] = avg_expmtl_temps[v,:]
                    Exp_sand_temps4resid108 [u-1,:,:,:] =  Exp_sand_temps_4d[v,:,:,:]
                    
                    #and bring up the lower limit that the inner loop loops through
                    idx = v + 1
                    break
                    
    return





#%% 0. ASK USER WHICH DATA TO PROCESS
#Excel sheet with different run descriptions
diff_runs_file = graph_dir.joinpath('2021-07-13 Description of different runs.xlsx')
diff_runs = pd.read_excel(diff_runs_file, header=1, index_col=0)
    

#%%% Which run numbers to process?

#single run
# runs = [399]
# i=0
# for r in runs:
#     runs[i] = str(r)
#     i += 1

#range of runs
run_alpha = 399  #first run
run_om = 404 #last run
except_these = []   #list of runs to exclude


runs = [None]*(run_om - run_alpha + 1 - len(except_these))
i=0
for r in range(run_alpha,run_om+1):
    
    skip = 0 #-> variable to check if run needs to be skipped
    
    #check if r is any of the exceptions
    for unwanted in except_these:
        if r == unwanted:
            skip = 1
            break
    if skip == 1:
        pass
    #if not, do the usual - writing the run to the list and incrementing the index
    else:
        runs[i] = str(r)
        i=i+1


print('For runs: '); print(runs)
# breakpoint()
#%%% Names of simulation files with import data
df = pd.DataFrame(columns=['runs','fileID','fileID2'])


for i in range(0,len(runs)):
    df.loc[i,'runs']    = runs[i]
    df.loc[i,'fileID']  = 'r'+ runs[i] +'_1Sand_TC_Temps.txt'
    df.loc[i,'fileID2']  = 'r'+ runs[i] +'_2Sand_Viz_Temps.txt'


#%%% Variables needed for simulation
precon_days = 10    #Number of preconditioning days in each "rewind"

#%%% Ask user for inputs
#Initialize variables to False
ExpTCimport = 0
SimTCImport = 0
VizImport = 0  
SimExpPlot = 0
VizXSection=0 
ExpTCGraph = 0
crossrowsNcols = 0
TdiffExpVSsim = 0
dE_ExpVSsim = 0
PerfMetr = 0
PerfMetrGrTCs = 0
Save_graphs=0
plot_individ_SS_graphs = 0
plot_multiple_on_same = 0

#EXPERIMENTAL DATA

ExpTCGraph = int(input("Only graph experimental TC data? (1/0)"))

   
#VIZ DATA
SimExpPlot = int(input("Import sim Sand TC data and compare to exp data (avg of each level)? (1/0)")) 
if SimExpPlot == 1:
    SimTCImport = 1
VizXSection = int(input("Plot temperature cross-sections slices of Viz data? (1/0)")) 
if VizXSection == 1:
   VizImport = 1 

#CROSS-ROWS AND COLUMN PLOTS
crossrowsNcols = int(input("Plot cross-rows and cross-columns of sim and exp data for the sand store? (1/0)")) 
if crossrowsNcols == 1:
   SimTCImport = 1 
   
#CALCULATE TEMPERATURE DIFFERENCES B/T EXPERIMENTAL AND SIMULATED DATA
TdiffExpVSsim = int(input("Calc. RMSD b/t sim and expmtl temperatures in sand store and plot entire_SS graph?"))

if TdiffExpVSsim == 1:
    #Plot each entire_SS individually?
    plot_individ_SS_graphs = int(input("Plot each entire_SS graph individually?"))

    #Plot multiple graphs of entire_SS on same plot?
    plot_multiple_on_same = int(input("Plot multiple graphs of entire_SS on same plot?"))
    if plot_multiple_on_same == 1:
        fig5, ax5 = plt.subplots()
        labels5 = [None]*len(runs)
        for run in range(0,len(runs)):
            labels5[run] = input("What is the legend label for Run " + runs[run] + "?")

Save_graphs=int(input("Save output graphs? (1/0)")) 

#CALCULATE DIFFERENCE IN TOTAL ENERGY STORED IN SAND
dE_ExpVSsim = int(input("Calc. KM3 metric?"))

#PERFORMANCE METRICS
# PerfMetr = int(input("Import Performance Metrics data (ground TCs,etc)? (1/0)")) 
# if PerfMetr ==1:
#     PerfMetrGrTCs = int(input("Plot ground TCs against simulated ground temp data? (1/0)")) 
#     if PerfMetrGrTCs ==1:
#         VizImport = 1

for run in range(0,len(runs)):
    #%%% EXPERIMENTAL FILENAMES with import data

    #check the period to get the experimental file names accordingly
    if diff_runs.Period[int(runs[run])].rstrip() == 'cooldown June 2021':
        
        #%%%% Cooldown
        mode = 1
        #sand store temperatures file
        filename1 = 'Sand_temperatures_2021-05-28-to-2021-07-06.dat' 
        #loops heat transfer file
        filename2 ='Loops_heat_transfer_2021-05-28-to-2021-07-06.dat'

    elif diff_runs.Period[int(runs[run])].rstrip() == 'charging Mar 2021':

        #%%%% Charging
        mode = 2
        #sand store temperatures file
        filename1 = 'Sand_temperatures_2021-03-21-to-2021-03-23.dat' 
        #loops heat transfer file
        filename2 ='Loops_heat_transfer_2021-03-21-to-2021-03-23.dat'

    elif diff_runs.Period[int(runs[run])].rstrip() == 'discharging Nov 2020':

        #%%%% Discharging
        mode = 3
        #sand store temperatures file
        filename1 = 'Sand_temperatures_2020-11-25-to-2020-11-27.dat' 
        #loops heat transfer file
        filename2 ='Loops_heat_transfer_2020-11-25-to-2020-11-27.dat'

    elif diff_runs.Period[int(runs[run])].rstrip() == 'charging + discharging Nov 29':
    
        #%%%% Charging and Discharging (mode 4)
        mode = 4
        #sand store temperatures file
        filename1 = 'Sand_temperatures_2020-11-28-to-2020-11-30.dat' 
        #loops heat transfer file
        filename2 ='Loops_heat_transfer_2020-11-28-to-2020-11-30.dat'
        
    elif diff_runs.Period[int(runs[run])].rstrip() == 'charging + discharging Nov 23':
    
        #%%%% Charging and Discharging 2 (mode 5)
        mode = 5
        #sand store temperatures file
        filename1 = 'Sand_temperatures_2020-11-23-to-2020-11-23.dat' 
        #loops heat transfer file
        filename2 ='Loops_heat_transfer_2020-11-23-to-2020-11-23.dat'
    elif diff_runs.Period[int(runs[run])].rstrip() == 'heating season 2021':

        #%%%% Entire heating season (mode 6)
        mode = 6
        #sand store temperatures file
        filename1 = 'Sand_temperatures_2020-11-19-to-2021-04-30.dat' 
        #loops heat transfer file
        filename2 ='Loops_heat_transfer_2020-11-19-to-2021-04-30.dat'
        
    #%% 3. GRAPH EXPERIMENTAL TC DATA
    if ExpTCGraph == 1:
        
        #%%% Import data
        Exp_sand_temps, exp_start_date, exp_end_date = readSerranoSandMultipleDates(filename1)
        
        #%%% Process Experimental Sand Temp data into Numpy arrays
        Exp_sand_temps_4d, exp_time_array  = processExpTCDataToArrays(Exp_sand_temps,exp_start_date)
       
        #%%% Plot all layers of TCs over time, in sub plots
        plotLayersofExpSandTempData_Sbplts(exp_time_array,Exp_sand_temps)
       
       #%%%% Plot layers of TCs over time, in individual plots
        plotTopLayerofExpSandData_IndPlts(exp_time_array,Exp_sand_temps)
        plotMiddleLayerofExpSandData_IndPlts(exp_time_array,Exp_sand_temps)
        plotBottomLayerofExpSandData_IndPlts(exp_time_array,Exp_sand_temps)
        
        #%%%% Plot average of TCs in north/south rows over time, in individual plots, for each layer
        plotExpSandDataAvgofNSRows_IndPlts(exp_time_array,Exp_sand_temps_4d)
        # Plot average of TCs in west/east rows over time, in individual plots, for each layer
        plotExpSandDataAvgofEWRows_IndPlts(exp_time_array,Exp_sand_temps_4d)
        
        #%%% Plot expmtl data cross-sections     
        # plotExpSandDataXSections(Exp_sand_temps_4d,0,60)
        # plotExpSandDataYSections(Exp_sand_temps_4d,0,60)
 
        #%%% Plot average temps of sand store data (exp.) only

        #take averages of sand temps (lowest index is bottom-most layer of sand store)
        avg_expmtl_temps = np.mean(np.mean(Exp_sand_temps_4d,axis=1),axis=1)
        
        #plot
        fig1, ax1 = plt.subplots()
        for col in range(0,np.size(avg_expmtl_temps,1)):
            ax1.plot(exp_time_array,avg_expmtl_temps[:,col],marker=',')  
            
        ax1.set_title("Mean temperatures of sand store layers")
        ax1.set_ylabel("Temperature (\N{DEGREE SIGN}C)")
     
        ax1.legend(['Exp-bott','Exp-mid','Exp-top',],loc='best')
        
        
        #Documentation on date locators
        #https://matplotlib.org/stable/api/dates_api.html
        locator = mdates.AutoDateLocator()
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_minor_locator(mdates.DayLocator())
        
        loc = ticker.MultipleLocator(base=2.0)
        ax1.yaxis.set_major_locator(loc)
        ax1.yaxis.set_minor_locator(ticker.MultipleLocator(base=1.0))
        
        # #set the major locator on x-axis to every 6 days
        # locator.intervald[mdates.DAILY]=[6] #https://matplotlib.org/stable/api/dates_api.html#matplotlib.dates.AutoDateLocator
        
        #Formatting - make the x-axis labels more consices
        #https://matplotlib.org/stable/gallery/ticks_and_spines/date_concise_formatter.html
        formatter = mdates.ConciseDateFormatter(locator)
        ax1.xaxis.set_major_formatter(formatter)
        fig1.autofmt_xdate() #automatically makes the x-labels rotate
        
        #gridlines on
        ax1.grid(True)
        #Set max and min values for axes
        ax1.set_xlim(exp_time_array[0], exp_time_array[-1])
        if mode == 6:
            ax1.set_ylim(20,60)
        else:
            ax1.set_ylim(30,60)
   
    
        
    #%% 4. SIMULATION TC DATA
    #%%% Import simulation data into variables
    
    if SimTCImport == 1:
           
        #Import sand temp data
        sndTemps,sndTemps_time, start_date, end_date, run_number = readOutputSimuDatafile(df.loc[run,'fileID'])
        
        
        
        
        #%%% Process Simulated Sand Temp data into Numpy arrays 
        avg_sim_temps_snd_layer = np.mean(np.mean(sndTemps,axis=1),axis=1)
        
        #take average of all sand temps per timestep
        avg_sim_temps_snd = np.mean(np.mean(np.mean(sndTemps,axis=1),axis=1),axis=1)

        
        #%%%Print final avg Sand store temps to screen
        print("Final avg sim temperatures in sand store layers for Run", run_number, ":\n", avg_sim_temps_snd_layer[-1,:])
        print("Bottom        Middle       Top")
        
        #%%% Make list of dates for x-axis of simulation plots
        sndTemps_datetimes = DateArrayFromSimDayHoursArray(sndTemps_time[:,0], start_date)
        sndTemps_datetimes[1] += timedelta(hours = sndTemps_time[1,1])  #the offset of 0.001h needs to be put back into the array -> it was lost with thw function DateArrayFromSimDayHoursArray
        
    
        # %% 5. EXPERIMENTAL TC DATA
        #%%%Import Experimental Data Files
        if (ExpTCGraph != 1): 
            #it's the first run or the experimental data needed isn't the same as the last run 
            if (run == 0) or (diff_runs.Period[int(runs[run-1])] != diff_runs.Period[int(runs[run])]): 
                ##IMPORT MULTIPLE DATES OF SERRANO DATA
                # import data
                Exp_sand_temps, exp_start_date, exp_end_date = readSerranoSandMultipleDates(
                    filename1)

                # #IMPORT SINGLE DATE OF SERRANO DATA
                # filename1 = 'Sand_temperatures_2021-06-04.dat'
                # # import data
                # Exp_sand_temps, exp_start_date = readSerranoSandSingleDate(filename1)
                
                # Process Experimental Sand Temp data into Numpy arrays
                Exp_sand_temps_4d,exp_time_array  = processExpTCDataToArrays(Exp_sand_temps,exp_start_date)
            
        
                #take averages of sand temps (lowest index is bottom-most layer of sand store)
                avg_expmtl_temps = np.mean(np.mean(Exp_sand_temps_4d,axis=1),axis=1)
            
            
        if SimExpPlot == 1:    
            #%% 6. PLOT SIM & EXPT'L TC DATA
            #%%% Plot simulated and experimental data
            
            fig1, ax1 = plt.subplots()
            colors=['g','blue','r']

            # Plot experimental data 
            labels = ['Exp Level C','Exp Level B','Exp Level A']
            
            for col in range(np.size(avg_expmtl_temps,1)-1,-1,-1):
                ax1.plot(exp_time_array,avg_expmtl_temps[:,col],color=colors[col], label=labels[col])  
                
            #Plot simulated data on same figure
            labels = ['Sim Level C','Sim Level B','Sim Level A']
            #less than 115 has rewinds, so we don't want to graph them...start at line 964 in TC sand temp data
            if int(run_number) < 115:
                fig1, ax1 = plotTimeSeriesEntireArray(sndTemps_datetimes[964:],
                                                                 avg_sim_temps_snd_layer[964:],
                                                                 fig1, 
                                                                 ax1, 
                                                                 labels, 
                                                                 colors)
            else:
                fig1, ax1 = plotTimeSeriesEntireArray(sndTemps_datetimes,
                                                                 avg_sim_temps_snd_layer,
                                                                 fig1, 
                                                                 ax1, 
                                                                 labels, 
                                                                 colors)


            #plt.plot(sndTemps_datetimes,avg_sim_temps_snd_layer[:,2],marker='*')
            #ax1.set_xlabel("Days since start of simulation")
            ax1.set_ylabel("Temperature (\N{DEGREE SIGN}C)")
            
            #rename figure and change legend to include experimental data
            ax1.set_title("Mean temperatures of sand store layers, Run " + run_number)
            ax1.legend()
            
            
            #Documentation on date locators
            #https://matplotlib.org/stable/api/dates_api.html
            locator = mdates.AutoDateLocator()
            ax1.xaxis.set_major_locator(locator)
            ax1.xaxis.set_minor_locator(mdates.DayLocator())
            
            loc = ticker.MultipleLocator(base=2.0)
            ax1.yaxis.set_major_locator(loc)
            ax1.yaxis.set_minor_locator(ticker.MultipleLocator(base=1.0))
            
            # #set the major locator on x-axis to every 6 days
            # locator.intervald[mdates.DAILY]=[6] #https://matplotlib.org/stable/api/dates_api.html#matplotlib.dates.AutoDateLocator
            
            #Formatting - make the x-axis labels more consices
            #https://matplotlib.org/stable/gallery/ticks_and_spines/date_concise_formatter.html
            formatter = mdates.ConciseDateFormatter(locator)
            ax1.xaxis.set_major_formatter(formatter)
            fig1.autofmt_xdate() #automatically makes the x-labels rotate
            
            #gridlines on
            ax1.grid(True)
            
            #Set max and min values for axes
            ax1.set_xlim(exp_time_array[0], exp_time_array[-1])
            if mode == 6:
                ax1.set_ylim(20,58)
            else:
                ax1.set_ylim(30,58)
            
            if Save_graphs == 1:
                #save graph to file with proper naming scheme
                plt.savefig(graph_dir.joinpath("r" + df.loc[run,'runs'] + ".png"), dpi=150, bbox_inches='tight')
            
            #%%% Plot expmtl sand TC data (top layer)
            
            # #SUBPLOTS
            # plotLayersofExpSandTempData_Sbplts(exp_time_array,Exp_sand_temps)
            
            # #INDIVIDUAL PLOTS
            # plotTopLayerofExpSandData_IndPlts(exp_time_array,Exp_sand_temps)
            
            
            
                
    
    #%% 7. IMPORT VISUALIZATION DATA
    if VizImport == 1:
        
        #%%% Import sand temp data
        sndVizTemps,sndVizTemps_time, start_date_viz, end_date_viz, xcoords, x_viz_indices, ycoords, y_viz_indices, zcoords, z_viz_indices, TCx_coords, TCy_coords, simu_days, run_number_ID2 = readOutputSimuVizDatafile(df.loc[run,'fileID2'])
        
        
        #%%% Convert arrays of strings to integers
        x_viz_indices = x_viz_indices.astype(int)
        y_viz_indices = y_viz_indices.astype(int)
        z_viz_indices = z_viz_indices.astype(int)
        
        xcoords = xcoords.astype(float)
        ycoords = ycoords.astype(float)
        zcoords = zcoords.astype(float)
        TCx_coords = TCx_coords.astype(float)
        TCy_coords = TCy_coords.astype(float)
    
        #%%% Get Cartesian coords of indices that will be plotted
        x_coords2plt = np.zeros(shape=(x_viz_indices.shape[0],1))   #initialize array
        for i in range(0,x_coords2plt.shape[0]):
            x_coords2plt[i] = xcoords[x_viz_indices[i]]
        
        y_coords2plt = np.zeros(shape=(y_viz_indices.shape[0],1))   #initialize array
        for j in range(0,y_coords2plt.shape[0]):
            y_coords2plt[j] = ycoords[y_viz_indices[j]]
         
        z_coords2plt = np.zeros(shape=(z_viz_indices.shape[0],1))   #initialize array
        for k in range(0,z_coords2plt.shape[0]):
            z_coords2plt[k] = zcoords[z_viz_indices[k]]
            
    #%% 8. PLOT DOMAIN VISUALIZATION DATA
       
    if VizXSection == 1:
        
        # Plot temperature cross-sections slices    
        
        # %%%% Fixed variables for plotting
        # Index values for the 3 axes to plot temps at.
        aa=round(len(x_viz_indices)/2)
        cc=round(len(z_viz_indices)/2)
        #due to simulations with N/S symmetry about y-axis b/t runs 63 and 120 (inclusive)
        if int(run_number_ID2) >=63 and int(run_number_ID2) <121:
            bb=round(len(y_viz_indices))-1
        else:
            bb=round(len(y_viz_indices)/2)
        
        #data points will be plotted every _(data_pt_freq)__ days
        data_pt_freq = 10
        
        #x-, y-, and z-coordinates for concrete
        x_conc_coord1 = np.mean([x_coords2plt[5,0],x_coords2plt[6,0]])
        x_conc_coord2 = np.mean([x_coords2plt[7,0],x_coords2plt[8,0]])
        x_conc_coord3 = np.mean([x_coords2plt[25,0],x_coords2plt[26,0]])
        x_conc_coord4 = np.mean([x_coords2plt[27,0],x_coords2plt[28,0]])
        
        y_conc_coord1 = np.mean([x_coords2plt[5,0],x_coords2plt[6,0]])
        y_conc_coord2 = np.mean([x_coords2plt[7,0],x_coords2plt[8,0]])
        y_conc_coord3 = np.mean([x_coords2plt[25,0],x_coords2plt[26,0]])
        y_conc_coord4 = np.mean([x_coords2plt[27,0],x_coords2plt[28,0]])
            
        z_conc_coord1 = np.mean([z_coords2plt[5,0],z_coords2plt[6,0]])
        z_conc_coord2 = np.mean([z_coords2plt[7,0],z_coords2plt[8,0]])
        
        #x-, y-, and z-coordinates for XPS
        x_XPS_coord1 = np.mean([x_coords2plt[7,0],x_coords2plt[8,0]])
        x_XPS_coord2 = np.mean([x_coords2plt[9,0],x_coords2plt[10,0]])
        x_XPS_coord3 = np.mean([x_coords2plt[23,0],x_coords2plt[24,0]])
        x_XPS_coord4 = np.mean([x_coords2plt[25,0],x_coords2plt[26,0]])
        
        y_XPS_coord1 = np.mean([x_coords2plt[7,0],x_coords2plt[8,0]])
        y_XPS_coord2 = np.mean([x_coords2plt[9,0],x_coords2plt[10,0]])
        y_XPS_coord3 = np.mean([x_coords2plt[23,0],x_coords2plt[24,0]])
        y_XPS_coord4 = np.mean([x_coords2plt[25,0],x_coords2plt[26,0]])
            
        z_XPS_coord1 = np.mean([z_coords2plt[7,0],z_coords2plt[8,0]])
        z_XPS_coord2 = np.mean([z_coords2plt[9,0],z_coords2plt[10,0]])
        z_XPS_coord3 = np.mean([z_coords2plt[21,0],z_coords2plt[22,0]])
        z_XPS_coord4 = np.mean([z_coords2plt[23,0],z_coords2plt[24,0]])
        
        
        # %%%% Plot across z, one snapshot in time
        # plot one temp at x=0,y=16, vertical cross-section 
        # fig2,ax2 = plt.subplots()
        # ax2.plot(x_coords2plt,sndVizTemps[0,16,:,13])
        
        
        # %%%% Plot multiple graphs across x-coords over time
        # aa=round(len(x_viz_indices)/2)
        # cc=round(len(z_viz_indices)/2)
        # # due to N/S symmetry about y-axis:
        # if int(run_number_ID2) >=63 and int(run_number_ID2) <121:
        #     bb=round(len(y_viz_indices))-1
        # else:
        #     bb=round(len(y_viz_indices)/2)
        
        # #plot temperatures at ~mid-point in z-direction, across all x-coords, and at the ~mid-point of y
        # #plot hourly figures at each time step
        # for time in range(0,sndVizTemps_time.shape[0]):
        #     fig2,ax2 = plt.subplots()
        #     ax2.plot(x_coords2plt[:,0],sndVizTemps[time,:,bb,cc],'+')    
        #     #plot node coordinate spacing on figure
        #     ax2.plot(x_coords2plt,np.ones(shape=(x_coords2plt.shape),dtype=int)*80,'+') 
        #     ax2.set_title("Run "+run_number_ID2+ ": Temps at t=" + "{:.2f}".format(sndVizTemps_time[time,0]) + "days across x-axis, at y=" + str(y_viz_indices[bb]) + " and z=" + str(z_viz_indices[cc]))
        #     ax2.set_xlabel('Length of num. domain (m)')
        #     ax2.set_ylabel('Temperature (\N{DEGREE SIGN}C)')
        #     ax2.set_ylim(0,70)
        #     #plot vertical shading for XPS and conc
        #     ax2.axvspan(x_conc_coord1,x_conc_coord2,color='grey', alpha=0.5, lw=0)  
        #     ax2.axvspan(x_conc_coord3,x_conc_coord4,color='grey', alpha=0.5, lw=0) 
        #     ax2.axvspan(x_XPS_coord1,x_XPS_coord2,color='pink', alpha=0.5, lw=0) 
        #     ax2.axvspan(x_XPS_coord3,x_XPS_coord4,color='pink', alpha=0.5, lw=0) 
        
        # %%%% Plot multiple graphs across y-coords over time
        # # plot temperatures at ~mid-point in x-direction, across all y-coords, and at the ~mid-point of z
        # #plot hourly figures at each time step
        # aa=19
        # cc=22
        # for time in range(0,sndVizTemps_time.shape[0]):
        #     fig2,ax2 = plt.subplots()
        #     ax2.plot(y_coords2plt[:,0],sndVizTemps[time,aa,:,cc])    
        #     ax2.set_title("Run "+run_number_ID2+ ": Temps at t=" + "{:.2f}".format(sndVizTemps_time[time,0]) + "days across y-axis, at x=" + str(x_viz_indices[aa]) + " and z=" + str(z_viz_indices[cc]))
        #     ax2.set_xlabel('Length of num. domain (m)')
        #     ax2.set_ylabel('Temperature (\N{DEGREE SIGN}C)')
        #     ax2.set_ylim(0,70)
        #     #plot vertical shading for XPS and conc
        #     ax2.axvspan(x_conc_coord1,x_conc_coord2,color='grey', alpha=0.5, lw=0)  
        #     ax2.axvspan(x_conc_coord3,x_conc_coord4,color='grey', alpha=0.5, lw=0) 
        #     ax2.axvspan(x_XPS_coord1,x_XPS_coord2,color='pink', alpha=0.5, lw=0) 
        #     ax2.axvspan(x_XPS_coord3,x_XPS_coord4,color='pink', alpha=0.5, lw=0) 
        
        
        # %%%% Plot multiple graphs across z-coords over time
        # #plot temperatures at ~mid-point in x-direction, across all z-coords, and at the ~mid-point of y
        # #plot hourly figures at each time step
        # for time in range(0,sndVizTemps_time.shape[0]):
        #     fig2,ax2 = plt.subplots()
        #     ax2.plot(z_coords2plt[:,0],sndVizTemps[time,aa,bb,:])    
        #     ax2.set_title("Run "+run_number_ID2+ ": Temps at t=" + "{:.2f}".format(sndVizTemps_time[time,0]) + "days across z-axis, at x=" + str(x_viz_indices[aa]) + " and y=" + str(y_viz_indices[bb]))
        #     ax2.set_xlabel('Length of num. domain (m)')
        #     ax2.set_ylabel('Temperature (\N{DEGREE SIGN}C)')
        #     ax2.set_ylim(0,60)

        
        #     #plot node coordinate spacing on figure
        #     ax2.plot(z_coords2plt,np.ones(shape=(z_coords2plt.shape),dtype=int)*50,'+')    
            
        #     #plot vertical shading for XPS and conc
        #     ax2.axvspan(z_conc_coord1,z_conc_coord2,color='grey', alpha=0.5, lw=0)  
        #     ax2.axvspan(z_XPS_coord1,z_XPS_coord2,color='pink', alpha=0.5, lw=0) 
        #     ax2.axvspan(z_XPS_coord3,z_XPS_coord4,color='pink', alpha=0.5, lw=0)       
        
        
        # %%%% Plot one temp distribution per day across x-axis
        fig1,ax1 = plt.subplots()
        
        #rewinds no longer being used at run 115 and above
        if int(run_number_ID2) >= 115:
            for time in range(0,sndVizTemps_time.shape[0],data_pt_freq):
                print("Graphed point (day,hour): " + str(sndVizTemps_time[time]))
                ax1.plot(x_coords2plt[:,0],sndVizTemps[time,:,bb,cc])  
            #plot second temp distribution in array, steady state temp distribution after IC
            ax1.plot(x_coords2plt[:,0],sndVizTemps[1,:,bb,cc])   
            print("Graphed point (day,hour): " + str(sndVizTemps_time[1]))
        else:
            for time in range(0,sndVizTemps_time.shape[0],data_pt_freq):
                print("Graphed point (day,hour): " + str(sndVizTemps_time[time]))
                ax1.plot(x_coords2plt[:,0],sndVizTemps[time,:,bb,cc])    
        
        #plot last temp distribution in array, just before midnight
        ax1.plot(x_coords2plt[:,0],sndVizTemps[-1,:,bb,cc])   
        print("Graphed point (day,hour): " + str(sndVizTemps_time[-1]))
        # #plot node coordinate spacing on figure
        # ax1.plot(x_coords2plt,np.ones(shape=(x_coords2plt.shape),dtype=int)*50,'+')    
        
        #plot vertical shading for XPS and conc
        ax1.axvspan(x_conc_coord1,x_conc_coord2,color='grey', alpha=0.5, lw=0)  
        ax1.axvspan(x_conc_coord3,x_conc_coord4,color='grey', alpha=0.5, lw=0) 
        ax1.axvspan(x_XPS_coord1,x_XPS_coord2,color='pink', alpha=0.5, lw=0) 
        ax1.axvspan(x_XPS_coord3,x_XPS_coord4,color='pink', alpha=0.5, lw=0) 
        
        ax1.set_title("Run "+run_number_ID2+ ": Daily temp distribution for " + str(round(sndVizTemps_time[-1,0])) + "days across x-axis, at y=" + str(y_viz_indices[bb]) + " and z=" + str(z_viz_indices[cc]))
        ax1.set_ylim(0,60)
        ax1.set_xlabel('Length of num. domain (m)')
        ax1.set_ylabel("Temperature (\N{DEGREE SIGN}C)")
        
        if Save_graphs == 1:
            #save graph to file with proper naming scheme
            plt.savefig(graph_dir.joinpath("r" + df.loc[run,'runs'] + "_viz_x.png"), dpi=150, bbox_inches='tight')
    
              
        # %%%% Plot one temp distribution per day across y-axis

        fig1,ax1 = plt.subplots()

        #rewinds no longer being used at run 115 and above, domain symmetry still used
        if int(run_number_ID2) >= 115 and int(run_number_ID2) < 121:
            for time in range(0,sndVizTemps_time.shape[0],data_pt_freq):
                print("Graphed point (day,hour): " + str(sndVizTemps_time[time]))
                ax1.plot(x_coords2plt[:,0],
                         np.concatenate((sndVizTemps[time,aa,:,cc],np.flip(sndVizTemps[time,aa,:,cc],0)),axis=0))   
            #plot second temp distribution in array, steady state temp distribution after IC
            ax1.plot(x_coords2plt[:,0],
                     np.concatenate((sndVizTemps[1,aa,:,cc],np.flip(sndVizTemps[1,aa,:,cc],0)),axis=0))   
            print("Graphed point (day,hour): " + str(sndVizTemps_time[1]))
        
        #rewinds no longer being used at run 115 (and 121) and above, domain symmetry no longer used
        elif int(run_number_ID2) >= 121: 
            for time in range(0,sndVizTemps_time.shape[0],data_pt_freq):  #,(precon_days+1)):
                print("Graphed point (day,hour): " + str(sndVizTemps_time[time]))
                ax1.plot(y_coords2plt[:,0],sndVizTemps[time,aa,:,cc]) 
            #plot second temp distribution in array, steady state temp distribution after IC
            ax1.plot(y_coords2plt[:,0],sndVizTemps[1,aa,:,cc])   
            print("Graphed point (day,hour): " + str(sndVizTemps_time[1]))
                
                   
        #rewinds being used, domain symmetry used at run 63 and above
        elif int(run_number_ID2) >=63:
            for time in range(0,sndVizTemps_time.shape[0],data_pt_freq):  #,(precon_days+1)):
                print("Graphed point (day,hour): " + str(sndVizTemps_time[time]))
                ax1.plot(x_coords2plt[:,0],
                     np.concatenate((sndVizTemps[time,aa,:,cc],np.flip(sndVizTemps[time,aa,:,cc],0)),axis=0))   
        #rewinds being used, domain symmetry not used at run 63 and below
        else:
            for time in range(0,sndVizTemps_time.shape[0],data_pt_freq):  #,(precon_days+1)):
                print("Graphed point (day,hour): " + str(sndVizTemps_time[time]))
                ax1.plot(y_coords2plt[:,0],sndVizTemps[time,aa,:,cc])  
        
        #plot last temp distribution in array, just before midnight
        if int(run_number_ID2) >=63 and int(run_number_ID2) <121:  
            #due to simulations with N/S symmetry about y-axis starting at run 63 and above
            ax1.plot(x_coords2plt[:,0],
                     np.concatenate((sndVizTemps[-1,aa,:,cc],np.flip(sndVizTemps[-1,aa,:,cc],0)),axis=0)) 
            print("Graphed point (day,hour): " + str(sndVizTemps_time[-1]))
            # #plot node coordinate spacing on figure    
            # ax1.plot(x_coords2plt,np.ones(shape=(x_coords2plt.shape),dtype=int)*50,'+')
        else:
            #no N/S symmetry
            ax1.plot(y_coords2plt[:,0],sndVizTemps[-1,aa,:,cc])
            print("Graphed point (day,hour): " + str(sndVizTemps_time[-1]))
            # #plot node coordinate spacing on figure    
            # ax1.plot(y_coords2plt,np.ones(shape=(y_coords2plt.shape),dtype=int)*50,'+')
       
        #plot vertical shading for XPS and conc
        ax1.axvspan(x_conc_coord1,x_conc_coord2,color='grey', alpha=0.5, lw=0)  
        ax1.axvspan(x_conc_coord3,x_conc_coord4,color='grey', alpha=0.5, lw=0) 
        ax1.axvspan(x_XPS_coord1,x_XPS_coord2,color='pink', alpha=0.5, lw=0) 
        ax1.axvspan(x_XPS_coord3,x_XPS_coord4,color='pink', alpha=0.5, lw=0) 
     
        ax1.set_title("Run "+run_number_ID2+ ": Daily temp distribution for " + str(round(sndVizTemps_time[-1,0])) + "days across y-axis, at x=" + str(x_viz_indices[aa]) + " and z=" + str(z_viz_indices[cc]))
        ax1.set_xlabel('Length of num. domain (m)')
        ax1.set_ylabel("Temperature (\N{DEGREE SIGN}C)")
        ax1.set_ylim(0,60)
        
        if Save_graphs == 1:
            #save graph to file with proper naming scheme
            plt.savefig(graph_dir.joinpath("r" + df.loc[run,'runs'] + "_viz_y.png"), dpi=150, bbox_inches='tight')
        
        # %%%% Plot one temp distribution per day across z-axis
        
        fig1,ax1 = plt.subplots()
        
        #rewinds no longer being used at run 115 and above
        if int(run_number_ID2) >= 115:
            for time in range(0,sndVizTemps_time.shape[0],data_pt_freq):
                print("Graphed point (day,hour): " + str(sndVizTemps_time[time]))
                ax1.plot(z_coords2plt[:,0],sndVizTemps[time,aa,bb,:]) 
            #plot second temp distribution in array, steady state temp distribution after IC
            ax1.plot(z_coords2plt[:,0],sndVizTemps[1,aa,bb,:])   
            print("Graphed point (day,hour): " + str(sndVizTemps_time[1]))
        else:

            for time in range(0,sndVizTemps_time.shape[0],data_pt_freq):
                print("Graphed point (day,hour): " + str(sndVizTemps_time[time]))
                ax1.plot(z_coords2plt[:,0],sndVizTemps[time,aa,bb,:])  
                
        
        #plot last temp distribution in array, just before midnight
        ax1.plot(z_coords2plt[:,0],sndVizTemps[-1,aa,bb,:])  
        print("Graphed point (day,hour): " + str(sndVizTemps_time[-1]))
        # #plot node coordinate spacing on figure
        # ax1.plot(z_coords2plt,np.ones(shape=(z_coords2plt.shape),dtype=int)*50,'+')    
        
        #plot vertical shading for XPS and conc
        ax1.axvspan(z_conc_coord1,z_conc_coord2,color='grey', alpha=0.5, lw=0)  
        ax1.axvspan(z_XPS_coord1,z_XPS_coord2,color='pink', alpha=0.5, lw=0) 
        ax1.axvspan(z_XPS_coord3,z_XPS_coord4,color='pink', alpha=0.5, lw=0) 
        
        
        ax1.set_title("Run "+run_number_ID2+ ": Daily temp distribution for " + str(round(sndVizTemps_time[-1,0])) + "days across z-axis, at x=" + str(x_viz_indices[aa]) + " and y=" + str(y_viz_indices[bb]))
        ax1.set_xlabel('Length of num. domain (m)')
        ax1.set_ylabel("Temperature (\N{DEGREE SIGN}C)")
        ax1.set_ylim(0,60)
        
        if Save_graphs == 1:
            #save graph to file with proper naming scheme
            plt.savefig(graph_dir.joinpath("r" + df.loc[run,'runs'] + "_viz_z.png"), dpi=150, bbox_inches='tight')
        
    #%% 8a. PLOT VIZ DATA AT SPECIFIED TIMESTEP -> A=sndVizTemps[timestep,:,:,:]
    #Plot all node temps in y-z cross sectional plane at time = timestep
    
    # timestep = 0
    # A=sndVizTemps[timestep,:,:,:]
    
    #%%%% ONE LINE PER GRAPH, MANY GRAPHS
    # # To graph N->S temps along x-axis of sand store, at a fixed z-value, going through all y-points    
    # z= 31  #z_viz_indices index value
    # for y in range(0,A.shape[1]):
    #     fig1,ax1 = plt.subplots()
    #     ax1.plot(x_coords2plt[:,0],A[:,y,z])
    #     ax1.set_title("N->S temps along x-axis, at Viz timestep " +str(timestep)+ ", at y=" + str(y_viz_indices[y]) +", z="+ str(z_viz_indices[z]))
    #     ax1.set_xlabel('Length of num. domain in x-direction (m)')
    #     ax1.set_ylabel("Temperature (\N{DEGREE SIGN}C)")
    #     ax1.set_ylim(0,80)
        
    # #To graph W->E temps along y-axis of sand store, at a fixed x-value, going up through all z-points    
    # x=30
    # for z in range(0,A.shape[2]):
    #     fig1,ax1 = plt.subplots()
    #     ax1.plot(y_coords2plt[:,0],A[x,:,z])
    #     ax1.set_title("W->E temps along y-axis, at Viz timestep " +str(timestep)+ ", at x=" + str(x_coords2plt[x,0]) +"m, z="+ str(z_coords2plt[z,0]) + "m")
    #     ax1.set_xlabel('Length of num. domain in y-direction(m)')
    #     ax1.set_ylabel("Temperature (\N{DEGREE SIGN}C)")
    #     ax1.set_ylim(0,80)
    # #To graph N->S temps along x-axis of sand store, at a fixed y-value, going up through all z-points    
    # y=30
    # for z in range(0,A.shape[2]):
    #     fig1,ax1 = plt.subplots()
    #     ax1.plot(x_coords2plt[:,0],A[:,y,z])
    #     ax1.set_title("N->S temps along x-axis, at Viz timestep " +str(timestep)+ ", at y=" + str(y_coords2plt[y,0]) +"m, z="+ str(z_coords2plt[z,0]) + "m")
    #     ax1.set_xlabel('Length of num. domain in x-direction (m)')
    #     ax1.set_ylabel("Temperature (\N{DEGREE SIGN}C)")
    #     ax1.set_ylim(0,80)
    
    #%%%% MANY LINES PER GRAPH, ONE GRAPH
    # #To graph N->S temps along x-axis of sand store, for all y-nodes, at a fixed z-value 
    # z= 24  #z_viz_indices index value
    # fig1,ax1 = plt.subplots()
    # for y in range(0,A.shape[1]):
    #     ax1.plot(x_coords2plt[:,0],A[:,y,z])
    #     ax1.set_title("N->S temps (x-axis), at Viz timestep " +str(timestep)+ ", at each y output node, at z-node "+ str(z_viz_indices[z]))
    #     ax1.set_xlabel('Length of num. domain in x-direction (m)')
    #     ax1.set_ylabel("Temperature (\N{DEGREE SIGN}C)")
    #     ax1.set_ylim(0,80)
    #%%%%MANY LINES PER GRAPH, ONE GRAPH -> used to troubleshoot while running solver
    # #To graph N->S temps along x-axis of sand store, for all y-nodes, at a fixed z-value 
    # A=Temps[:,:,:,0]
    # z= 24  #z_viz_indices index value
    # fig1,ax1 = plt.subplots()
    # for y in range(0,A.shape[1]):
    #     ax1.plot(node_xcoords,A[:,y,z])
    #     ax1.set_title("N->S temps (x-axis), at each y output node, at z-node "+ str(z))
    #     ax1.set_xlabel('Length of num. domain in x-direction (m)')
    #     ax1.set_ylabel("Temperature (\N{DEGREE SIGN}C)")
    #     ax1.set_ylim(0,80)
        
    #%%%%MANY LINES PER GRAPH, MANY GRAPHS -> used to troubleshoot while running solver
    #To graph N->S temps along x-axis of sand store, for all y-nodes, at each z-value
    # A=Temps[:,:,:,0]
    # for z in range(0,A.shape[2]):
    #     fig1,ax1 = plt.subplots()
    #     for y in range(0,A.shape[1]):
    #         ax1.plot(node_xcoords,A[:,y,z])
    #         ax1.set_title("N->S temps (x-axis), at each y output node, at z-node "+ str(z))
    #         ax1.set_xlabel('Length of num. domain in x-direction (m)')
    #         ax1.set_ylabel("Temperature (\N{DEGREE SIGN}C)")
    #         ax1.set_ylim(0,80)


    # for z in range(0,A.shape[2]):
    #     fig1,ax1 = plt.subplots()
    #     for x in range(0,A.shape[0]):
    #         ax1.plot(node_ycoords,A[x,:,z])
    #         ax1.set_title("E->W temps (y-axis), at each x output node, at z-node "+ str(z))
    #         ax1.set_xlabel('Length of num. domain in y-direction (m)')
    #         ax1.set_ylabel("Temperature (\N{DEGREE SIGN}C)")
    #         ax1.set_ylim(0,80)        
        
    #%%%%MANY LINES PER GRAPH, MANY GRAPHS
    # #To graph N->S temps along x-axis of sand store, for all y-nodes, at each z-value
    # for z in range(0,A.shape[2]):
    #     fig1,ax1 = plt.subplots()
    #     for y in range(0,A.shape[1]):
    #         ax1.plot(x_coords2plt[:,0],A[:,y,z])
    #         ax1.set_title("N->S temps (x-axis), at Viz timestep " +str(timestep)+ ", at each y output node, at z-node "+ str(z_viz_indices[z]))
    #         ax1.set_xlabel('Length of num. domain in x-direction (m)')
    #         ax1.set_ylabel("Temperature (\N{DEGREE SIGN}C)")
    #         ax1.set_ylim(0,80)
        
    
    # #MANY LINES PER GRAPH, MANY GRAPHS
    # #To graph W->E temps along y-axis of sand store, for all x-nodes, at each z-value
    # for z in range(0,A.shape[2]):
    #     fig1,ax1 = plt.subplots()
    #     for x in range(0,A.shape[0]):
    #         ax1.plot(y_coords2plt[:,0],A[x,:,z])
    #         ax1.set_title("W->E temps (y-axis), at Viz timestep " +str(timestep)+ ", at each x output node, at z-node "+ str(z_viz_indices[z]))
    #         ax1.set_xlabel('Length of num. domain in y-direction (m)')
    #         ax1.set_ylabel("Temperature (\N{DEGREE SIGN}C)")
    #         ax1.set_ylim(0,80)
        
    # #MANY LINES PER GRAPH, MANY GRAPHS
    # #To graph B->T temps along z-axis of sand store, for all x-nodes, at each y-value
    # for y in range(0,A.shape[1]):
    #     fig1,ax1 = plt.subplots()
    #     for x in range(0,A.shape[0]):
    #         ax1.plot(z_coords2plt[:,0],A[x,y,:])
    #         ax1.set_title("B->T temps (z-axis), at Viz timestep " +str(timestep)+ ", at each x output node, at y-node "+ str(y_viz_indices[y]))
    #         ax1.set_xlabel('Height of num. domain in z-direction (m)')
    #         ax1.set_ylabel("Temperature (\N{DEGREE SIGN}C)")
    #         ax1.set_ylim(0,80)    
        
        
        
        
    #%% 9. PLOT SAND SIMU AND EXP. DATA N/S AND E/W AVERAGES
    
    '''this function plots 1 graph with 6 lines of data, each line is the average TC from each row of TCs (north to south)
    It will compare the simulated data to the experimental data    
    '''
    if crossrowsNcols == 1:
        TC_exp_EW_row_avgs = np.mean(Exp_sand_temps_4d,axis=2)
        TC_sim_EW_row_avgs = np.mean(sndTemps,axis=2)
        
        TC_exp_NS_row_avgs = np.mean(Exp_sand_temps_4d,axis=1)
        TC_sim_NS_row_avgs = np.mean(sndTemps,axis=1)
        
        layer_name = [", bottom",
                  ", middle",
                  ", top"]
        
        clrs = ['blue',
                'darkorange',
                'g',
                'r',
                'dodgerblue',
                'darkorchid']
        
        #%%% Plot all sim and exp rows on one plot
        
        shift_EW = np.zeros(shape=[3,6])
        #because the averages of the simltd and expmtl rows did not start at the same temperatures, find the difference in starting temperatures b/t these arrays
        for layer in range(2,-1,-1):
            for i in range(0,6):
                shift_EW[layer,i] = TC_exp_EW_row_avgs[0,i,layer] - TC_sim_EW_row_avgs[0,i,layer]
        
        for layer in range(2,-1,-1):
            #E/W rows, from northmost to southmost
            fig,ax = plt.subplots()
            
            
            lgnd = ["R1, N",# + layer_name[layer],
                        "R2",#+ layer_name[layer],
                        "R3",#+ layer_name[layer],
                        "R4",#+ layer_name[layer],
                        "R5",#+ layer_name[layer],
                        "R6, S"]#+ layer_name[layer]]
            
            #plot
            for i in range(0,6):
                ax.plot(exp_time_array,TC_exp_EW_row_avgs[:,i,layer], label="Exp "+ lgnd[i], color=clrs[i]) #plot top layer of sand store    
                #plot the simulated rows with the shift between starting exp and sim avg temps
                ax.plot(sndTemps_datetimes,TC_sim_EW_row_avgs[:,i,layer] + shift_EW[layer,i], label="Sim " + lgnd[i], linestyle='--',color=clrs[i]) #plot top layer of sand store
        
            ax.set_title("Sand store"+ layer_name[layer] +" layer, avg TC row temps, " + "Run " + str(runs[run]))#: "+ str(exp_start_date) + " to " + str(exp_end_date))
            ax.legend()
            ax.set_ylim(20,60)

            #locator = mdates.DayLocator(interval=2)
            locator = mdates.AutoDateLocator()
            # locator = mdates.MonthLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            ax.grid(True)
            ax.set_ylabel("Temperature ($\degree$C)")
            ax.xaxis.set_major_locator(locator)   
            # ax.xaxis.set_minor_locator(mdates.MonthLocator())   
            ax.xaxis.set_major_formatter(formatter)
            ax.set_ylim(20,60)
            fig.autofmt_xdate() #automatically makes the x-labels rotate
 
            if Save_graphs == 1:
                #save graph to file with proper naming scheme
                plt.savefig(graph_dir.joinpath("r" + df.loc[run,'runs'] + "N2S" + layer_name[layer] + ".png"), dpi=150, bbox_inches='tight')
            
            
            
            
            
            shift_NS = np.zeros(shape=[3,6])
            #because the averages of the simltd and expmtl rows did not start at the same temperatures, find the difference in starting temperatures b/t these arrays
            for layer in range(2,-1,-1):
                for j in range(0,6):
                    shift_NS[layer,j] = TC_exp_NS_row_avgs[0,j,layer] - TC_sim_NS_row_avgs[0,j,layer]            
                
                
            
            
            #N/S rows, from westmost to eastmost        
            fig1,ax1 = plt.subplots()    
            lgnd = ["C1, W",# + layer_name[layer],
                       "C2",#+ layer_name[layer],
                        "C3",#+ layer_name[layer],
                        "C4",#+ layer_name[layer],
                        "C5",#+ layer_name[layer],
                        "C6, E"]#+ layer_name[layer]]
            
            #plot
            for j in range(0,6):
                ax1.plot(exp_time_array,TC_exp_NS_row_avgs[:,j,layer], label="Exp "+ lgnd[j], color=clrs[j]) #plot top layer of sand store    
                #plot the simulated rows with the shift between starting exp and sim avg temps
                ax1.plot(sndTemps_datetimes,TC_sim_NS_row_avgs[:,j,layer] + shift_NS[layer,j], label="Sim " + lgnd[j], linestyle='--',color=clrs[j]) #plot top layer of sand store
        
            ax1.set_title("Sand store"+ layer_name[layer] +" layer, avg TC column temps, " + "Run " + str(runs[run]))#: "+ str(exp_start_date) + " to " + str(exp_end_date))
            ax1.legend()
            ax1.set_ylim(20,60)
            
            ax1.grid(True)
            ax1.set_ylabel("Temperature ($\degree$C)")
            ax1.xaxis.set_major_locator(locator)   
            # ax.xaxis.set_minor_locator(mdates.MonthLocator())   
            ax1.xaxis.set_major_formatter(formatter)
            ax1.set_ylim(20,60)
            fig1.autofmt_xdate() #automatically makes the x-labels rotate
            
            if Save_graphs == 1:
                #save graph to file with proper naming scheme
                plt.savefig(graph_dir.joinpath("r" + df.loc[run,'runs'] + "W2E" + layer_name[layer] + ".png"), dpi=150, bbox_inches='tight')
                
        #%%% Plot only simulated data, all rows in each layer on one plot
        
        # for layer in range(2,-1,-1):
        #     #E/W rows, from northmost to southmost
        #     fig,ax = plt.subplots()
        #     lgnd = ["Row 1, northern edge" + layer_name[layer],
        #                 "Row 2"+ layer_name[layer],
        #                 "Row 3"+ layer_name[layer],
        #                 "Row 4"+ layer_name[layer],
        #                 "Row 5"+ layer_name[layer],
        #                 "Row 6, southern edge"+ layer_name[layer]]
        #     for j in range(0,6):
                
        #         ax.plot(sndTemps_datetimes,TC_sim_EW_row_avgs[:,j,layer], label=lgnd[j], color=clrs[j]) #plot top layer of sand store
            
        #     #N/S rows, from westmost to eastmost
        #     fig1,ax1 = plt.subplots()
        #     lgnd = ["Col 1, western edge" + layer_name[layer],
        #                "Col 2"+ layer_name[layer],
        #                 "Col 3"+ layer_name[layer],
        #                 "Col 4"+ layer_name[layer],
        #                 "Col 5"+ layer_name[layer],
        #                 "Col 6, eastern edge"+ layer_name[layer]]
        #     for j in range(0,6):
        #         ax1.plot(sndTemps_datetimes,TC_sim_NS_row_avgs[:,j,layer], label=lgnd[j],color=clrs[j]) #plot top layer of sand store
                
        #     #locator = mdates.DayLocator(interval=2)
        #     locator = mdates.AutoDateLocator()
        #     # locator = mdates.MonthLocator()
        #     formatter = mdates.ConciseDateFormatter(locator)
            
            
        #     ax.legend()      
        #     ax.grid(True)
        #     ax.set_title("Sand store"+ layer_name[layer] +" layer, avg sim TC row temps, " + "Run " + str(runs[run])) #: "+ str(exp_start_date) + " to " + str(exp_end_date))
        #     ax.set_ylabel("Temperature ($\degree$C)")
        #     ax.xaxis.set_major_locator(locator)   
        #    # ax.xaxis.set_minor_locator(mdates.MonthLocator())   
        #     ax.xaxis.set_major_formatter(formatter)
        #     ax.set_ylim(20,60)
        #     fig.autofmt_xdate() #automatically makes the x-labels rotate

        #     ax1.legend()        
        #     ax1.grid(True)
        #     ax1.set_title("Sand store"+ layer_name[layer] +" layer, avg sim TC col temps, " + "Run " + str(runs[run]))#: "+ str(exp_start_date) + " to " + str(exp_end_date))
        #     ax1.set_ylabel("Temperature ($\degree$C)")
        #     ax1.xaxis.set_major_locator(locator)   
        #     #ax1.xaxis.set_minor_locator(mdates.DayLocator())   
        #     ax1.xaxis.set_major_formatter(formatter)
        #     ax1.set_ylim(20,60)
        #     fig1.autofmt_xdate() #automatically makes the x-labels rotate
        

        #%%% Plot each row separately (sim and viz)
        
        # #plot E/W north to south layers of sand temps
        # for layer in range(2,-1,-1):   
        #     lgnd = ["Row 1, northern edge" + layer_name[layer],
        #                 "Row 2"+ layer_name[layer],
        #                 "Row 3"+ layer_name[layer],
        #                 "Row 4"+ layer_name[layer],
        #                 "Row 5"+ layer_name[layer],
        #                 "Row 6, southern edge"+ layer_name[layer]]
        #     for j in range(0,6):
        #         fig,ax = plt.subplots()
        #         ax.plot(exp_time_array,TC_exp_EW_row_avgs[:,j,layer], label="Exp",color=clrs[j]) #plot top layer of sand store
        #         ax.plot(sndTemps_datetimes,TC_sim_EW_row_avgs[:,j,layer], label="Sim", linestyle='--',color=clrs[j]) #plot top layer of sand store
                    
        #         #locator = mdates.DayLocator(interval=2)
        #         locator = mdates.AutoDateLocator()
        #         # locator = mdates.MonthLocator()
        #         formatter = mdates.ConciseDateFormatter(locator)
        #         # ax.legend([layer_name[layer] +" Row 1 (north)",
        #         #             layer_name[layer] +" Row 2",
        #         #             layer_name[layer] +" Row 3",
        #         #             layer_name[layer] +" Row 4",
        #         #             layer_name[layer] +" Row 5",
        #         #             layer_name[layer] +" Row 6 (south)"])
        #         ax.legend()          
        #         ax.grid(True)
        #         ax.set_title("Sand "+ lgnd[j] + " layer, avg TC row temps, " + "Run " + str(runs[run]))#: "+ str(exp_start_date) + " to " + str(exp_end_date))
        #         ax.set_ylabel("Temperature ($\degree$C)")
        #         ax.xaxis.set_major_locator(locator)   
        #             # ax.xaxis.set_minor_locator(mdates.MonthLocator())   
        #         ax.xaxis.set_major_formatter(formatter)
        #         ax.set_ylim(20,60)
        #         fig.autofmt_xdate() #automatically makes the x-labels rotate
        
        # #N/S rows, from westmost to eastmost
        # for layer in range(2,-1,-1):   
        #     lgnd = ["Col 1, western edge" + layer_name[layer],
        #                 "Col 2"+ layer_name[layer],
        #                 "Col 3"+ layer_name[layer],
        #                 "Col 4"+ layer_name[layer],
        #                 "Col 5"+ layer_name[layer],
        #                 "Col 6, eastern edge"+ layer_name[layer]]
        #     for j in range(0,6):
        #         fig,ax = plt.subplots()
        #         ax.plot(exp_time_array,TC_exp_NS_row_avgs[:,j,layer], label="Exp",color=clrs[j]) #plot top layer of sand store
        #         ax.plot(sndTemps_datetimes,TC_sim_NS_row_avgs[:,j,layer], label="Sim", linestyle='--',color=clrs[j]) #plot top layer of sand store
                    
        #         #locator = mdates.DayLocator(interval=2)
        #         locator = mdates.AutoDateLocator()
        #         # locator = mdates.MonthLocator()
        #         formatter = mdates.ConciseDateFormatter(locator)
        #         # ax.legend([layer_name[layer] +" Row 1 (north)",
        #         #             layer_name[layer] +" Row 2",
        #         #             layer_name[layer] +" Row 3",
        #         #             layer_name[layer] +" Row 4",
        #         #             layer_name[layer] +" Row 5",
        #         #             layer_name[layer] +" Row 6 (south)"])
        #         ax.legend()          
        #         ax.grid(True)
        #         ax.set_title("Sand "+ lgnd[j] + " layer, avg TC col temps, " + "Run " + str(runs[run]))#: "+ str(exp_start_date) + " to " + str(exp_end_date))
        #         ax.set_ylabel("Temperature ($\degree$C)")
        #         ax.xaxis.set_major_locator(locator)   
        #             # ax.xaxis.set_minor_locator(mdates.MonthLocator())   
        #         ax.xaxis.set_major_formatter(formatter)
        #         ax.set_ylim(20,60)
        #         fig.autofmt_xdate() #automatically makes the x-labels rotate
                
#%% 10. PLOTS & RMSD OF EXP vs SIM
   
    '''This will calculate the difference between temperature nodes of experimental vs simulated data, 
    to give overall metrics for "fit" of the model
'''
    
    #does this section need to be done?
    if TdiffExpVSsim == 1:
        ExpTCimport = 0
        
        #%%% Check if data has been imported
        #were sim TC temps already imported?
        if (SimExpPlot == 1 or crossrowsNcols == 1):    
            pass
        else: 
            
            #Check if sim file exists
            try: 
                open(df.loc[run,'fileID'], 'r')
            except FileNotFoundError:   #if it doesn't exist, print error and move on to next run
                print('**************************')    
                print("File " + df.loc[run,'fileID'] + " does not exist!!!")
                print('**************************')    
                #skip the rest of that run
                continue
            
            #Import sim sand temp data
            sndTemps,sndTemps_time, start_date, end_date, run_number = readOutputSimuDatafile(df.loc[run,'fileID'])
          
            
            #Process Simulated Sand Temp data into Numpy arrays 
            avg_sim_temps_snd_layer = np.mean(np.mean(sndTemps,axis=1),axis=1)
            
            #take average of all sand temps per timestep
            avg_sim_temps_snd = np.mean(np.mean(np.mean(sndTemps,axis=1),axis=1),axis=1)

            
            # Make list of dates for x-axis of simulation plots
            sndTemps_datetimes = DateArrayFromSimDayHoursArray(sndTemps_time[:,0], start_date)
            sndTemps_datetimes[1] += timedelta(hours = sndTemps_time[1,1])  #the offset of 0.001h needs to be put back into the array -> it was lost with thw function DateArrayFromSimDayHoursArray
        
        #were exp temps already imported?
        if SimExpPlot == 1:    
            pass
        else: 
            #it's the first run or the experimental data needed isn't the same as the last run 
            if (run == 0) or (diff_runs.Period[int(runs[run-1])] != diff_runs.Period[int(runs[run])]): 
                ##IMPORT MULTIPLE DATES OF SERRANO DATA
                # import data
                Exp_sand_temps, exp_start_date, exp_end_date = readSerranoSandMultipleDates(filename1)
                
                # Process Experimental Sand Temp data into Numpy arrays
                Exp_sand_temps_4d,exp_time_array  = processExpTCDataToArrays(Exp_sand_temps,exp_start_date)
                
                #take layer averages of sand temps (lowest index is bottom-most layer of sand store)
                avg_expmtl_temps = np.mean(np.mean(Exp_sand_temps_4d,axis=1),axis=1)  
                
                
        #%%% Round time arrays to nearest hour
        #initialize arrays
        sndTemps_datetimes_rnd = [0]*len(sndTemps_datetimes)
        exp_time_array_rnd= [0]*(len(sndTemps_datetimes) - 1)
        avg_expmtl_SStemp4resid = np.full((len(exp_time_array_rnd)), np.NaN)
        avg_expmtl_temps4resid = np.full((len(exp_time_array_rnd),3), np.NaN)
        Exp_sand_temps4resid108 = np.full((len(exp_time_array_rnd),6,6,3), np.NaN)
        
        #round time to nearest hour for simulated data array
        for time in range(2,len(sndTemps_datetimes)):   # -> it starts at 2 because the value at index 1 is the preconditioning period time value
            #sndTemps_datetimes is hourly data, make sure it rounds to the hour
            sndTemps_datetimes_rnd[time] = sim_round2hr(sndTemps_datetimes[time], 60)
            
        #round time to nearest hour for experimental data array        
        exp_round2hr()
        
        #get average temps at each of the timesteps for calculating residuals
        avg_expmtl_SStemp4resid = np.mean(avg_expmtl_temps4resid, axis=1)

        
        #%%% Residuals of 3 avg layer temps and entire SS
        
        #initialize array 
        residSimVsExpLayers = pd.DataFrame(exp_time_array_rnd,columns=['datetime'])
        col_list = ['LayerC','LayerB','LayerA', 'entireSS']
        residSimVsExpLayers[col_list] = np.nan
        #remove first row, bc the 1st row of temperatures in both exp. and sim. arrays should be the same...that's how I initialized the simulation) 
        residSimVsExpLayers.drop(index=0, axis=0, inplace=True)
        
        #compare temps residuals and alert if the times aren't exactly the same 
        for time in range(0,len(sndTemps_datetimes_rnd)-2):
            #layers:
            residSimVsExpLayers.iloc[time,1] =  avg_sim_temps_snd_layer[time+2,0] - avg_expmtl_temps4resid[(time+1),0]
            residSimVsExpLayers.iloc[time,2] = avg_sim_temps_snd_layer[time+2,1] - avg_expmtl_temps4resid[(time+1),1] 
            residSimVsExpLayers.iloc[time,3] = avg_sim_temps_snd_layer[time+2,2] - avg_expmtl_temps4resid[(time+1),2]
            #entire sand store:
            residSimVsExpLayers.iloc[time,4] = avg_sim_temps_snd[time+2] - avg_expmtl_SStemp4resid[time+1]
            
            if (sndTemps_datetimes_rnd[time+2] != exp_time_array_rnd[time+1]):
                #unless it's the very last timestep
                if exp_time_array_rnd[time+1] != exp_time_array_rnd[-1]:
                    print("ERROR! There is an issue lining up the simulation and experimental temp data when calculating the RMSD in run " +str(runs[run]) + ","+ str(time))

        #count the number of rows that are NaN (bc of experimental data not existing at the time stamp)
        rows_nan = residSimVsExpLayers['LayerC'].isna().sum()       

        #%%% RMSD of avg of 3 layers; total RMSD for run
        
        #using RSMD (forecasted - observed)^2        
        residSimVsExpLayers = pd.concat([residSimVsExpLayers, 
                                        residSimVsExpLayers[col_list].pow(2, axis=0).add_prefix('sq_')], 
                                       axis=1)
        
        #initialize empty list
        sq_col_list =['']*4
        #Make new column names and put into a list
        for i in range(0,4): sq_col_list[i] = ('sq_'+col_list[i])
        
        #sum the squares and divide by the number of entries, then take the sqrt
        RMSD = residSimVsExpLayers[sq_col_list].sum().div(residSimVsExpLayers.shape[0] - rows_nan,axis=0).pow(0.5, axis=0)
        
        #make a row of the RMSD results -> RMSD of each of Layer C, B, A, RMSD of SSavg temp, and  also the average RMSD of all the layers' RMSDs
        temp_row = np.array([[int(runs[run]),
                              RMSD[0],  #Layer C
                              RMSD[1],  #Layer B
                              RMSD[2],  #Layer A
                              RMSD[3],  #entire SS RMSD
                              np.mean(RMSD[0:3])]]) #avg of layer RMSDs
        


        #%%% RMSD for 108 TCs 
        
        # #initial temperature conditions are not the same for the simulated temps as for the exp arrays...
        # #so calculate the difference in initial conditions between sim and exp for each TC
        # delta_init = np.zeros(shape=(6,6,3))
        # res108 =  np.zeros(shape=(len(exp_time_array_rnd)-1,6,6,3)) #make the length one row shorter bc we aren't going to take the resid at t=0s
        # RMSD108 = np.zeros(shape=(6,6,3))
        
        # for c in range(0,3):
        #     for b in range (0,6):
        #         for a in range (0,6):
        #             delta_init[a,b,c] = sndTemps[1,a,b,c] - Exp_sand_temps4resid108[0,a,b,c]
        
        # # calculate the residuals for each TC at each time step (except t=0), with the "corrected" sim temp
        # # also start calculating the RMSD for each TC by summing the squares of all the residuals as the outer "for loop" runs
        # nan_count = 0
        # for time in range(0,len(exp_time_array_rnd)-1):
        #     for c in range(0,3):
        #         for b in range (0,6):
        #             for a in range (0,6):            
        #                 res108[time,a,b,c] = (sndTemps[time+2,a,b,c] - delta_init[a,b,c]) - Exp_sand_temps4resid108[time+1,a,b,c]                        
        #                 #check if residuals for that time step exist or if the values are NaN. Do not add the NaN values to the RMSD108 calcs
        #                 if np.isnan(res108[time,a,b,c]) == True:
        #                     nan_count += 1
        #                 else:
        #                     RMSD108[a,b,c] = RMSD108[a,b,c] + res108[time,a,b,c]**2
        
        # # calculate RMSD for each of the 108 TCs
        # RMSD108 = RMSD108/(res108.shape[0] - nan_count/108)
        # RMSD108 = np.sqrt(RMSD108)
        
        # #Calculate average RMSD for 108 TCs
        # temp_row = np.append(temp_row ,[[np.mean(RMSD108)]], axis=1)
        

        #%%% Plots
        
        if plot_individ_SS_graphs == 1: #if user wants each graph inidividually
            
            #%%%% Plots of exp and sim averages of entire SS individually
            fig1, ax1 = plt.subplots()
            colors='black'
            labels = 'Exp data'
            
            # Plot experimental data 
            ax1.plot(exp_time_array_rnd,avg_expmtl_SStemp4resid,color=colors, label=labels)  
                
            #Plot simulated data on same figure
            colors='crimson'
            labels = 'Sim data'
            linestyle='--'
            #less than r115 has rewinds, so we don't want to graph them...start at line 964 in TC sand temp data
            if int(run_number) < 115:
                ax1.plot(sndTemps_datetimes_rnd[964:],avg_sim_temps_snd[964:], color=colors, label=labels, linestyle=linestyle)
            else:
                ax1.plot(sndTemps_datetimes_rnd[2:],avg_sim_temps_snd[2:], color=colors, label=labels, linestyle=linestyle)
    
    
            #plt.plot(sndTemps_datetimes,avg_sim_temps_snd_layer[:,2],marker='*')
            #ax1.set_xlabel("Days since start of simulation")
            ax1.set_ylabel("Temperature (\N{DEGREE SIGN}C)")
            
            #rename figure and change legend to include experimental data
            # ax1.set_title("Mean temperature of sand store, Run " + run_number)
            ax1.legend()
            
            
            #Documentation on date locators
            #https://matplotlib.org/stable/api/dates_api.html
            locator = mdates.AutoDateLocator()
            ax1.xaxis.set_major_locator(locator)
            ax1.xaxis.set_minor_locator(mdates.DayLocator())
            
            loc = ticker.MultipleLocator(base=2.0)
            ax1.yaxis.set_major_locator(loc)
            ax1.yaxis.set_minor_locator(ticker.MultipleLocator(base=1.0))
            
            # #set the major locator on x-axis to every 6 days
            # locator.intervald[mdates.DAILY]=[6] #https://matplotlib.org/stable/api/dates_api.html#matplotlib.dates.AutoDateLocator
            
            #Formatting - make the x-axis labels more consices
            #https://matplotlib.org/stable/gallery/ticks_and_spines/date_concise_formatter.html
            formatter = mdates.ConciseDateFormatter(locator)
            ax1.xaxis.set_major_formatter(formatter)
            fig1.autofmt_xdate() #automatically makes the x-labels rotate
            
            #gridlines on
            ax1.grid(True)
            
            #Set max and min values for axes
            ax1.set_xlim(exp_time_array[0], exp_time_array[-1])
            if mode == 6:
                ax1.set_ylim(20,58)
            else:
                ax1.set_ylim(30,58)
            
            if Save_graphs == 1:
                #save graph to file with proper naming scheme
                plt.savefig(graph_dir.joinpath("r" + df.loc[run,'runs'] + "_entireSS.png"), dpi=150, bbox_inches='tight')
                
        if plot_multiple_on_same == 1: #if user wants each data set on the same figure
            #%%%% Plots of exp and sim averages of entire SS 
                        
            if run==0:  #if it's the first loop through the runs, plot the experimental data
                colors='black'
                labels = 'Exp data'
                # Plot experimental data 
                # ax5.plot(exp_time_array_rnd,avg_expmtl_SStemp4resid,color=colors, label=labels)  
                
            #Plot simulated data on same figure
            # colors='crimson'
            labels = 'Sim data'
            linestyle=['dashed','dashed','dashed']
            clrs = ['blue',
                    'g',
                    'r',
                    'dodgerblue',
                    'darkorchid',
                    'blue',]
            #less than r115 has rewinds, so we don't want to graph them...start at line 964 in TC sand temp data
            if int(run_number) < 115:
                ax5.plot(sndTemps_datetimes_rnd[964:],avg_sim_temps_snd[964:], label=labels5[run], linestyle=linestyle[run],color=clrs[run])
            else:
                ax5.plot(sndTemps_datetimes_rnd[2:],avg_sim_temps_snd[2:], label=labels5[run], linestyle=linestyle[run],color=clrs[run])
    
    
            #plt.plot(sndTemps_datetimes,avg_sim_temps_snd_layer[:,2],marker='*')
            #ax5.set_xlabel("Days since start of simulation")
            ax5.set_ylabel("Temperature (\N{DEGREE SIGN}C)")
            
            #rename figure and change legend to include experimental data
            # ax5.set_title("Comparing mean temperature of SSTES for different runs")
            ax5.legend()
            
            
            #Documentation on date locators
            #https://matplotlib.org/stable/api/dates_api.html
            locator = mdates.AutoDateLocator()
            ax5.xaxis.set_major_locator(locator)
            ax5.xaxis.set_minor_locator(mdates.DayLocator())
            
            loc = ticker.MultipleLocator(base=2.0)
            ax5.yaxis.set_major_locator(loc)
            ax5.yaxis.set_minor_locator(ticker.MultipleLocator(base=1.0))
            
            # #set the major locator on x-axis to every 6 days
            # locator.intervald[mdates.DAILY]=[6] #https://matplotlib.org/stable/api/dates_api.html#matplotlib.dates.AutoDateLocator
            
            #Formatting - make the x-axis labels more consices
            #https://matplotlib.org/stable/gallery/ticks_and_spines/date_concise_formatter.html
            formatter = mdates.ConciseDateFormatter(locator)
            ax5.xaxis.set_major_formatter(formatter)
            fig5.autofmt_xdate() #automatically makes the x-labels rotate
            
            #gridlines on
            ax5.grid(True)
            
            #Set max and min values for axes
            ax5.set_xlim(exp_time_array[0], exp_time_array[-1])
            if mode == 6:
                ax5.set_ylim(20,58)
            else:
                ax5.set_ylim(30,58)
            
            if (runs[run] == runs[-1] and Save_graphs == 1):
                #save graph to file with proper naming scheme
                plt.savefig(graph_dir.joinpath("r" + df.loc[run,'runs'] + "_entireSS_multiple.png"), dpi=150, bbox_inches='tight')
        #%%%% Plots of residuals for averages of 3 layers 
        
        # # Plot of residuals
        # sns.lineplot(x=residSimVsExpLayers['datetime'],y=residSimVsExpLayers['LayerC'])
        # sns.lineplot(x=residSimVsExpLayers['datetime'],y=residSimVsExpLayers['LayerB'])
        # sns.lineplot(x=residSimVsExpLayers['datetime'],y=residSimVsExpLayers['LayerA'])

        

        #%%%% plot 108 residuals over time
        # for c in range(0,3):
        #     fig,ax = plt.subplots()
        #     for b in range (0,6):
        #         for a in range (0,6):
        #             ax.plot(sndTemps_datetimes_rnd[2:-1],res108[:,a,b,c],label = str([a,b]))
        #             ax.legend()
        #             ax.set_title(col_list[c] + " residuals")
        #             ax.set_ylim([-3,3])
        #             ax.set_ylabel("Temp, C")
        
        #%%%%plot 108 RMSD on a heatmap
        # fig,axs = plt.subplots(3,1, figsize = (10,10))
        # for c in range(0,3):
        #     sns.heatmap(RMSD108[:,:,c], ax=axs[c], annot=True, cmap="Blues").invert_yaxis()
        #     axs[c].set_title(col_list[c] + ' TC RMSDs')  
            
        #%%% Ouput RMSD info          
        
        #Create RMSD array for the runs, or append to it
        if run==0:       #if the run is the first run processed
            RMSD_array = temp_row
        else:
            RMSD_array = np.vstack((RMSD_array,temp_row))
                        
        #if last run, convert to Pandas dataframe 
        if runs[run] == runs[-1]:
            RMSD_array_df = pd.DataFrame(RMSD_array,columns=["Run",
                                                             "Layer_C",
                                                             "Layer_B",
                                                             "Layer_A",
                                                             "EntireSS_RMSD",
                                                             "AvgRMSDofLayers",
                                                             ])
            #print dataframe to screen
            print(RMSD_array_df)
            
                    

#%% 11. KM3 calculation

    #does this section need to be done?
    if dE_ExpVSsim == 1:
        ExpTCimport = 0
                
        #%%% Check if temperature data has been imported
        #were sim TC temps already imported?
        if (SimExpPlot == 1 or crossrowsNcols == 1 or TdiffExpVSsim == 1):    
            pass
        else: 
            
            #Check if sim file exists
            try: 
                open(df.loc[run,'fileID'], 'r')
            except FileNotFoundError:   #if it doesn't exist, print error and move on to next run
                print('**************************')    
                print("File " + df.loc[run,'fileID'] + " does not exist!!!")
                print('**************************')    
                #skip the rest of that run
                continue
            
            #Import sim sand temp data
            sndTemps,sndTemps_time, start_date, end_date, run_number = readOutputSimuDatafile(df.loc[run,'fileID'])
          
            
            #Process Simulated Sand Temp data into Numpy arrays 
            avg_sim_temps_snd_layer = np.mean(np.mean(sndTemps,axis=1),axis=1)
            
            #take average of all sand temps per timestep
            avg_sim_temps_snd = np.mean(np.mean(np.mean(sndTemps,axis=1),axis=1),axis=1)

            
            # Make list of dates for x-axis of simulation plots
            sndTemps_datetimes = DateArrayFromSimDayHoursArray(sndTemps_time[:,0], start_date)
            sndTemps_datetimes[1] += timedelta(hours = sndTemps_time[1,1])  #the offset of 0.001h needs to be put back into the array -> it was lost with thw function DateArrayFromSimDayHoursArray
        
        #were exp temps already imported?
        if (SimExpPlot == 1 or TdiffExpVSsim == 1):    
            pass
        else: 
            #it's the first run or the experimental data needed isn't the same as the last run 
            if (run == 0) or (diff_runs.Period[int(runs[run-1])] != diff_runs.Period[int(runs[run])]): 
                ##IMPORT MULTIPLE DATES OF SERRANO DATA
                # import data
                Exp_sand_temps, exp_start_date, exp_end_date = readSerranoSandMultipleDates(filename1)
                
                # Process Experimental Sand Temp data into Numpy arrays
                Exp_sand_temps_4d,exp_time_array  = processExpTCDataToArrays(Exp_sand_temps,exp_start_date)
                
                #take layer averages of sand temps (lowest index is bottom-most layer of sand store)
                avg_expmtl_temps = np.mean(np.mean(Exp_sand_temps_4d,axis=1),axis=1)  
                
    #%%% Calc thermal properties of simulated SS 
    #(code copied from Master_v5_4.py on Jun 15, 2022, svn rev 1133)
    
        # %%%% 2. Geometry of Domain
        # %%%% High-level geometry
    
        #x-y-z origin lies at north-west corner, bottom of sand store
        sand_L = 6 #(m) length of sand store  x = 0 m to x = 6 m
        sand_W = 6 #(m) width of sand store  y = 0 m to y = 6 m
        sand_H = 3 #(m) height of sand store, from bottom z = 0 m to top z = 3 m
    
        XPS_th = 0.35 #(m) spec'd thickness of insulation around sand store (14")
        conc_th = 0.10 #(m) spec'd thickness of concrete
    
        soil_top_th = 1.5 #(m) thickness of soil layer on top of sand store to surface
        soil_nrthside_th = 3 #(m) thickness of soil layer on north side of sand store to soil near house 
        soil_th = 3 #(m) thickness of soil layer on S, E, W, and bottom sides of sand store to far-field temp  
    
        no_TC = 0.5            #(m) buffer of 0.5m between sand store wall and 1st TCs (according to Briana Kemery's Visio document:https://chernode.mae.carleton.ca/sbes_svn/CHEeR_instrumentation/Sensor_connections_to_DAQs/Sand_store_TC_and_moisture_meter_placements.vsdx
        node_space = (sand_L - 2*no_TC)/5 #Calculating the distance (m) between TCs in the sand store, assuming that there are 6 x 6 TCs equally spaced on each layer,
    
        no_pwr = 0.3			#(m) length around outside of sand store where the PEX tubes are not layed down (estimated from photos)
        pwr_bott = 0.4			#(m) bottom z-value of x-z plane where PEX tubing is laid down
        pwr_mid = 1.3			#(m) middle z-value of x-z plane where PEX tubing is laid down
        pwr_upr = 2.2			#(m) top z-value of x-z plane where PEX tubing is laid down
    
        domain_L = soil_th + soil_nrthside_th + conc_th*2 + XPS_th*2 + sand_L #(m) length of numerical domain, x-axis
        domain_W =  soil_th*2 + conc_th*2 + XPS_th*2 + sand_W#(m) width of numerical domain, y-axis
        domain_H =  soil_th + soil_top_th + conc_th + XPS_th*2 + sand_H#(m) height of numerical domain, z-axis
    
    
        #%%%% Node Placement (varying mesh size)
    
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
    
        #%%%% Horizontal Coordinate Vectors
        #%%%%% SOIL COORDS
    
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
    
    
        # %%%%% CONCRETE COORDS
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
    
    
        #%%%%% XPS COORDS
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
    
    
        #%%%%% SAND, NO-POWER ZONE COORDS (north and west sides)
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
    
    
        #%%%%% SAND, POWER ZONE COORDS
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
    
    
        #%%%%% SAND NO-POWER ZONE COORDS (south and east sides)
        #============================================================
        #1st no_pwr node on south and east sides
        Hnopwr_coords_s[0] = soil_nrthside_th + conc_th + XPS_th + sand_L - no_pwr + Q5/2
        Hnopwr_coords_e[0] = soil_th + conc_th + XPS_th + sand_W - no_pwr + Q5/2
    
        #remainder of nodes on south and east sides
        for i in range(1,nodes_no_pwr):
            Hnopwr_coords_s[i] = Hnopwr_coords_s[i-1] + Q5
            Hnopwr_coords_e[i] = Hnopwr_coords_e[i-1] + Q5
    
    
        #%%%%% PUTTING X- AND Y- MESH COORDINATES VECTORS TOGETHER
        #============================================================
        node_xcoords = np.concatenate((Hsoil_coords_n,Hconc_coords_n,HXPS_coords_n,Hnopwr_coords_n,Hsand_coords_x,Hnopwr_coords_s,HXPS_coords_s,Hconc_coords_s,Hsoil_coords_s))
    
        node_ycoords = np.concatenate((Hsoil_coords_w,Hconc_coords_w,HXPS_coords_w,Hnopwr_coords_w,Hsand_coords_y,Hnopwr_coords_e,HXPS_coords_e,Hconc_coords_e,Hsoil_coords_e))
    
    
        #%%%% Vertical Coordinate Vectors
        #%%%%% SOIL COORDS
    
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
    
    
        #%%%%% CONCRETE COORDS
        #============================================================
        Vconc_coords_b = np.zeros(shape=[nodes_conc])
    
        #dx for concrete = Q3:
        #remainder of concrete length needing to be accounted for = R3
    
        #1st conc node
        Vconc_coords_b[0] = soil_th + Q3/2 + R3
    
        #remainder of nodes
        for i in range(1,nodes_conc):
            Vconc_coords_b[i] = Vconc_coords_b[i-1] + Q3
    
    
        #%%%%% XPS COORDS
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
    
    
        #%%%%% SAND COORDS
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
    
    
    
        #%%%%% PUTTING Z- COORDINATE VECTORS TOGETHER
        #============================================================
        node_zcoords = np.concatenate((Vsoil_coords_b,Vconc_coords_b,VXPS_coords_b,Vsand1,Vsand2,Vsand3,Vsand4,VXPS_coords_t,Vsoil_coords_t))
        # figure
        # plot(ones(length(node_zcoords),1),node_zcoords,'o')
    
    
    
        #%%%%VERTICAL index boundaries (6 layers)
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
    
        #%%%%HORIZONTAL index boundaries [9 layers]
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
    
    
        #%%%%Indices for points within the "power slab" in the x-y plane
        x_pwr_low_indx = indx_bndrz_x[8] #inclusive of this index
        x_pwr_high_indx = indx_bndrz_x[9]#inclusive of this index
        y_pwr_low_indx = indx_bndrz_y[8] #inclusive of this index
        y_pwr_high_indx = indx_bndrz_y[9]#inclusive of this index
    
        #============================================================
        #%%%%Vectors with CARTESIAN COORDS OF LAYER BOUNDARIES
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
        #%%%%Vectors holding the x and y coordinates of the TCs in the sand store
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
              
    
        # %%%% 2. Material Thermal Properties
        '''
        This script defines the thermal properties of the materials used in the simulation and also builds the 3-D thermal property matrices for the simulation.
        '''
        #Properties of water
        water_Cp = 4.186*1e3  # (J/kgK) specific heat for water
        water_rho = 998.2  # (kg/m3) density for water at 293 K https://hypertextbook.com/facts/2007/AllenMa.shtml
    
        #Properties of sand
        drySand_Cp = 830 #(J/kgK) specific heat for dry sand https://www.engineeringtoolbox.com/specific-heat-capacity-d_391.html
        drySand_rho = 1850 #(kg/m3) density for dry sand (personal correspondence with Abdulghader Abdulrahman, Research Manager, Dr.Paul Simms' lab)
    
        #volume percents of water layers in sand store
        vol_percent = np.zeros(shape=[3])
        vol_percent[0] = diff_runs['vol% bott'][int(runs[run])] #bottom layer = 33 vol#
        vol_percent[1] = diff_runs['vol% mid'][int(runs[run])] #middle layer = 11 vol#
        vol_percent[2] = diff_runs['vol% top'][int(runs[run])] #top layer = 7 vol#
    
        sand_lambda = np.zeros(shape=[3])
        sand_lambda[0] = diff_runs['λ_bott'][int(runs[run])] #(W/mK) bottom layer - thermal conductivity for saturated sand F. Rad (2009)
        sand_lambda[1] = diff_runs['λ_mid'][int(runs[run])] #(W/mK) middle layer - moist sand https://www.engineeringtoolbox.com/thermal-conductivity-d_429.html
        sand_lambda[2] = diff_runs['λ_top'][int(runs[run])] #(W/mK) top layer - moist sand https://www.engineeringtoolbox.com/thermal-conductivity-d_429.html
    
        #Properties of concrete at 300 K (Source - Incropera, 6e, p.939)
        conc_Cp = 880 #(J/kgK) specific heat 
        conc_rho = 2300  # (kg/m3) density 
        conc_lambda = 1.4 #(W/mK) thermal conductivity
    
        #Properties of XPS at 297 K(Source - Owens Corning data sheets and papers - see ppt presentation MCG5157 final project for sources)
        XPS_Cp_orig = 1500 #(J/kgK) specific heat source - article by BASF staff
        XPS_rho_orig = 30  # (kg/m3) density 
    
        #Properties of XPS
        vol_percent_H2O_SETface_XPS = diff_runs['SET(B) XPS Vol % H2O'][int(runs[run])]
        XPS_lambda_SET = diff_runs['SET(B)'][int(runs[run])]  #(W/mK) thermal conductivity
        XPS_Cp_SET = ((water_Cp * (water_rho*vol_percent_H2O_SETface_XPS)) + (XPS_Cp_orig*XPS_rho_orig)) / (water_rho * vol_percent_H2O_SETface_XPS + XPS_rho_orig) 
                      # (Number of Joules to raise sand in m3 by 1K + Joules to raise water in 1m3 by 1K) / (number of kg in 1m3)           
        XPS_rho_SET = water_rho * vol_percent_H2O_SETface_XPS + XPS_rho_orig 
    
    
        #Properties of soil (soil_rhoCp and soil_lambda) around CHEeR house sand
        #store
        soil_Cp = 750 #(J/kgK) (A. Wills MASc,Table 6.1)  
        soil_rho = 2800 #(kg/m3) (A. Wills MASc,Table 6.1)
        soil_lambda = 2   #(W/mK) thermal conductivity (A. Wills MASc,Table 6.1) (TRY PARAMETRIZING FOR CLAY VALUES )
    
    
        #%%%% 2. Build matrices of thermal properties
        # (from outside layers inwards)
    
        U_snd = 31
        V_snd = 31
        W_snd = 23
        z_shift = indx_bndrz_z[6]
        x_shift = indx_bndrz_x[6]
        y_shift = indx_bndrz_y[6]
        #Initialize thermal property matrices
        Lambda = np.zeros(shape=[U_snd,V_snd,W_snd])
        Cp = np.zeros(shape=[U_snd,V_snd,W_snd])
        rho = np.zeros(shape=[U_snd,V_snd,W_snd])
    
        #-----------------------            
                    
        #1st Sand layer [0m to ~halfway between first two PEX layers]
        for k in range(indx_bndrz_z[6] - z_shift,moisture_layer_height[0] - z_shift +1):
            for i in range(indx_bndrz_x[6] - x_shift,indx_bndrz_x[11] - x_shift+1):
                for j in range(indx_bndrz_y[6] - y_shift,indx_bndrz_y[11]+1 - y_shift):
                    Lambda[i,j,k] = sand_lambda[0] 
                    Cp[i,j,k] = ((water_Cp * (water_rho*vol_percent[0])) + (drySand_Cp*drySand_rho)) / (water_rho * vol_percent[0] + drySand_rho) 
                                 # (Number of Joules to raise sand in m3 by 1K + Joules to raise water in 1m3 by 1K) / (number of kg in 1m3)           
                    rho[i,j,k] = water_rho * vol_percent[0] + drySand_rho 
                
            
    
    
        #2nd Sand layer [ of 1st layer to ~halfway between next two PEX layers]
        for k in range((moisture_layer_height[0]+1 - z_shift),moisture_layer_height[1]+1 - z_shift):
            for i in range(indx_bndrz_x[6] - x_shift,indx_bndrz_x[11]+ 1 - x_shift):
                for j in range(indx_bndrz_y[6] - y_shift,indx_bndrz_y[11]+1 - y_shift):
                    Lambda[i,j,k] = sand_lambda[1]
                    Cp[i,j,k] = ((water_Cp * (water_rho*vol_percent[1])) + (drySand_Cp*drySand_rho)) / (water_rho * vol_percent[1] + drySand_rho) 
                    rho[i,j,k] = water_rho * vol_percent[1] + drySand_rho         
                
            
    
    
        #3rd Sand layer [ of 2nd layer to top of sand store]
        for k in range((moisture_layer_height[1]+1 - z_shift),moisture_layer_height[2]+1 - z_shift):
            for i in range(indx_bndrz_x[6] - x_shift,indx_bndrz_x[11]+1 - x_shift):
                for j in range(indx_bndrz_y[6] - y_shift,indx_bndrz_y[11]+1 - y_shift):
                    Lambda[i,j,k] = sand_lambda[2]
                    Cp[i,j,k] = ((water_Cp * (water_rho*vol_percent[2])) + (drySand_Cp*drySand_rho)) / (water_rho * vol_percent[2] + drySand_rho) 
                    rho[i,j,k] = water_rho * vol_percent[2] + drySand_rho    
    
    
        #%%% Calc avg Cp, rho of SSTES 
        
        avg_SSCp = np.mean(Cp)
        avg_SSrho = np.mean(rho)
        
        #%%%Calc KM3 metric
        #%%%%Heat input to Sand Store
        #Read in Headers file
        dat_headers = pd.read_csv(data_dir.joinpath('Serrano_dat_headers.csv'))

        #Read in loops heat transfer data into DataFrame
        Exp_heat_transfer_data = func_serr.read_loops_heat_transfer(filename2, dat_headers, main_dir, data_dir)
        
        #Process heat input data 
        #Put heat source data into a new numpy array
        solar_to_Sand_W = Exp_heat_transfer_data[['time','sand_H2O']].copy().to_numpy()
        
        #Integrate over the rate of energy going into the sand store to find total energy added
        int_energy_added = 0
        for row in range(0,solar_to_Sand_W.shape[0]):
            int_energy_added += 5 * 60 * solar_to_Sand_W[row,1] # 5 minute time intervals multiplied by the Watts injected
        
        int_energy_added = int_energy_added/1e9 #(convert J to GJ)

        # %%%% Heat extracted from Sand Store
        # Process discharging data 

        #Put discharging data into a new numpy array
        Sand_to_SH_W = Exp_heat_transfer_data[['time','sand_2_SH']].copy().to_numpy()

        #Integrate over the rate of energy going out of the sand store to find total energy extracted
        int_energy_taken = 0
        for row in range(0,Sand_to_SH_W.shape[0]):
            int_energy_taken += 5 * 60 * Sand_to_SH_W[row,1] # 5 minute time intervals multiplied by the Watts injected
        
        int_energy_taken = int_energy_taken/1e9 #(convert J to GJ)
        
        
        
        
        #%%% Calc dE_tot 
        
        #let the reference temp equal to 15C. Calculate all dE in reference to it (see p5.53 in notes)
        ref_temp = 30
        #pre-calculate mCp, from dE = mCpdT
        mCp = (avg_SSrho * 108) * avg_SSCp    # mCp = rho*volume*Cp
            # 108 m^3 is volume of SS
        
        #starting temp for both sim and exp temperatures
        T1 = avg_sim_temps_snd[0]
        
        #last temperature in simulated temp array
        T2_sim = avg_sim_temps_snd[-1]
        
        #last temperature in exp temp array
        try: #check if Section 10 was run - this pared-down hourly temperature array will exist if it was
            T2_exp = avg_expmtl_SStemp4resid[-1]
        except NameError:   #if not, take the avg of the last timestep in the experimental temp array, which wasn't pared down to hourly data
            T2_exp = np.mean(avg_expmtl_temps[-1,:])
    
        #Calculate delta energies
        dE1 = mCp * (T1 - ref_temp) /1e9 #(convert J to GJ)
        dE2_sim = mCp * (T2_sim - ref_temp) /1e9 #(convert J to GJ)
        dE2_exp = mCp * (T2_exp - ref_temp) /1e9 #(convert J to GJ)
        
        dE_tot_sim = dE1 - dE2_sim
        dE_tot_exp = dE1 - dE2_exp
        
        
        #Calc dE_heatloss
        dE_HL = dE1 + int_energy_added - dE2_exp - int_energy_taken
        
        #Calc normalized quantities
        denom = abs(int_energy_added) + abs(int_energy_taken) + abs(dE_HL)
        
        dE_norm_exp = dE_tot_exp / denom
        dE_norm_sim = dE_tot_sim / denom

        KM3 =  (dE2_exp - dE2_sim)/denom *100
        
        
        #make a row of the results 
        temp_row1 = np.array([[int(runs[run]),
                              KM3,
                              dE1,
                              dE2_exp,
                              dE2_sim,
                              abs(int_energy_added),
                              abs(int_energy_taken),
                              abs(dE_HL)]]) 
        
        
        # #the simulated change in stored Energy as a percent of the experimental change in stored Energy
        # pct_energy_metric = dE_tot_sim/dE_tot_exp * 100
        
        # print("\nExperiment: %.2f GJ reduction in STES stored energy" % (dE_tot_exp/1e9))
        # print("Simulation: %.2f GJ reduction in STES stored energy \n" % (dE_tot_sim/1e9))
        
        # print("The STES started with %.2f GJ of stored energy, above a zero reference energy at 15\N{degree sign}C." % (dE1/1e9))

        #%%% Ouput KM3 metric 
        
        #Create RMSD array for the runs, or append to it
        if run==0:       #if the run is the first run processed
            KM3_array = temp_row1
        else:
            KM3_array = np.vstack((KM3_array,temp_row1))
                        
        #if last run, convert to Pandas dataframe 
        if runs[run] == runs[-1]:
            KM3_array_df = pd.DataFrame(KM3_array,columns=["Run",
                                                             "KM3",
                                                             "Stored_E_init(GJ)",
                                                             "Stored_E_fin_exp(GJ)",
                                                             "Stored_E_fin_sim(GJ)",
                                                             "E_ch (GJ)",
                                                             "E_disch (GJ)",
                                                             "E_hl (GJ)"])
            #print dataframe to screen
            print(KM3_array_df)
            
    
#%% 12. PLOTS FOR PUBLISHING 

#%%% PCP/initialization 

# #Plot first two data points of viz data, to show SS state after initialization
# fig1,ax1 = plt.subplots()

# #rewinds no longer being used at run 115 and above
# if int(run_number_ID2) >= 115:
#     for time in range(0,1):
#         print("Graphed point (day,hour): " + str(sndVizTemps_time[time]))
#         ax1.plot(x_coords2plt[:,0],sndVizTemps[time,:,bb,cc])  
#     #plot second temp distribution in array, steady state temp distribution after IC
#     ax1.plot(x_coords2plt[:,0],sndVizTemps[1,:,bb,cc],linestyle='--')   
#     print("Graphed point (day,hour): " + str(sndVizTemps_time[1]))
            
# ax1.legend(["IC",
#             "After PCP"], loc='upper right')         

# #plot vertical shading for XPS and conc
# ax1.axvspan(x_conc_coord1,x_conc_coord2,color='grey', alpha=0.5, lw=0)  
# ax1.axvspan(x_conc_coord3,x_conc_coord4,color='grey', alpha=0.5, lw=0) 
# ax1.axvspan(x_XPS_coord1,x_XPS_coord2,color='pink', alpha=0.5, lw=0) 
# ax1.axvspan(x_XPS_coord3,x_XPS_coord4,color='pink', alpha=0.5, lw=0) 

# # ax1.set_title("Run "+run_number_ID2+ ": Daily temp distribution for " + str(round(sndVizTemps_time[-1,0])) + "days across x-axis, at y=" + str(y_viz_indices[bb]) + " and z=" + str(z_viz_indices[cc]))
# ax1.set_ylim(0,60)
# ax1.set_xlabel('Length of num. domain (m)')
# ax1.set_ylabel("Temperature (\N{DEGREE SIGN}C)")
    
#%% 13. PLOT PERFORMANCE METRICS DATA
if PerfMetr == 1:
#%%% Import performance metrics data
    Perf_metrics, exp_start_date2, exp_end_date2 = readSerranoPerformanceMetricsData(filename2)
    
    
#%%% Plot measured ground temps near Sand
        
    #create empty list
    exp_time_array_days2 = [0]*Perf_metrics.shape[0]
    #put start date in top row of list
    exp_time_array_days2[0] = exp_start_date2
    for d in range(1,Perf_metrics.shape[0]):
        exp_time_array_days2[d] = exp_time_array_days2[d-1] + timedelta(days=int(Perf_metrics.iloc[d,0]-Perf_metrics.iloc[d-1,0]))
    
    fig,ax1 = plt.subplots()
    ax1.plot(exp_time_array_days2,Perf_metrics['S_g_2_0'],label='2.0m depth (meas.)') #2.0m temp depth
    ax1.plot(exp_time_array_days2,Perf_metrics['S_g_2_7'],label='2.7m depth (meas.)') #2.0m temp depth
    ax1.plot(exp_time_array_days2,Perf_metrics['S_g_3_3'],label='3.3m depth (meas.)') #2.0m temp depth
    ax1.set_ylabel('Temperature ($\degree$C)')
    ax1.set_title("Ground Temperatures ~1m west of sand store, measured")
    
    locator = mdates.AutoDateLocator()
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_minor_locator(mdates.DayLocator())
    
    # #Sets major and minor tick spacing
    # loc = ticker.MultipleLocator(base=7.0)
    # ax1.xaxis.set_major_locator(loc)
    # ax1.xaxis.set_minor_locator(ticker.MultipleLocator(base=1.0))
    
    #set the major locator on x-axis to every 6 days
    locator.intervald[mdates.DAILY]=[7] #https://matplotlib.org/stable/api/dates_api.html#matplotlib.dates.AutoDateLocator
    
    #Formatting - make the x-axis labels more consices
    #https://matplotlib.org/stable/gallery/ticks_and_spines/date_concise_formatter.html
    formatter = mdates.ConciseDateFormatter(locator)
    ax1.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate() #automatically makes the x-labels rotate
    ax1.set_ylim(0,20)
    
    
    ax1.legend()
    
#%%% Plot simulated ground temps near ground Sand TCs
if PerfMetrGrTCs == 1:    
# Estimate the x-indices closest to where the ground TC hole might be (~4.2m/14ft south of the house wall)
    ground_TC_xindices = np.zeros(shape=[2], dtype=int)
    
    #location on x-axis of approximately 14 ft (4.2m) south of the house's south wall. The concrete wall of 
    #the SS is appx 3.52m south of the south wall. The ground TC is therefore 0.25m south of the starting portion of the sand store
    ground_TC_xcoord = TCx_coords[0] - 0.25
    
    #loop to find two closest TCs
    diff = 0
    for i in range(0,len(x_coords2plt)):
        diff = ground_TC_xcoord - x_coords2plt[i,0]
       #this finds the closest nodes to the real locations of the TCs
        if diff <=0:
            #then we want those two indices, which are sandwiching the ground_TC_xcoord
            ground_TC_xindices[0] = i-1
            ground_TC_xindices[1] = i
            break

# Find the two y-indices nearest to (sandwiching) the ground TC location (1.0m west of Sand Store)
    ground_TC_yindices = np.zeros(shape=[2], dtype=int)
    
    #location on y-axis of west-most TC sand node, minus the widths of the no-power-zone, XPS insulation, and concrete
    SS_west_wall_y_coord = TCy_coords[0] - 0.5 - 0.35 - 0.1 
    
    #location of ground TCs are appx 1m west of SS wall
    ground_TC_ycoord = SS_west_wall_y_coord - 1 
    
    #loop to find two closest TCs
    diff = 0
    for j in range(0,len(y_coords2plt)):
        diff = ground_TC_ycoord - y_coords2plt[j,0]
       #this finds the closest nodes to the real locations of the TCs
        if diff <=0:
            #then we want those two indices, which are sandwiching the ground_TC_ycoord
            ground_TC_yindices[0] = j-1
            ground_TC_yindices[1] = j
            break
        
# Find the z-indices nearest to the ground TC depths (2.0, 2.7, 3.3 below grade)
    ground_TC_zindices = np.zeros(shape=[6], dtype=int)
    
    #location of ground TCs are appx 2.0, 2.7, 3.3m below grade 
    ground_TC_zcoords = [2.0, 2.7, 3.3]
    
    #loop to find two closest TCs
    diff = 0
    for TC in range(0,3):
        for k in range(0,len(z_coords2plt)):
            diff = (z_coords2plt[-1,0] - ground_TC_zcoords[TC]) - z_coords2plt[k,0]
           #this finds the closest nodes to the real locations of the TCs
            if diff <=0:
                #then we want those two indices, which are sandwiching the ground_TC_ycoord
                ground_TC_zindices[TC*2] = k-1
                ground_TC_zindices[(TC*2)+1] = k
                break
                                        
#%% Plot the simulated temps nearest to the ground TCs    

 
    Perf_metric_dates = DateArrayFromSimDayHoursArray(Perf_metrics['day'].astype('float'),
                                                      exp_time_array_days2[0])
    
    color_scheme=['blue','red','orange','black']
     #Plot simulated ground temp data with experimental data
     
    for k in range(0,3):
        #plot one figure for each z-depth ground TC
        fig1, ax1 = plt.subplots()
        c=0
        #plot temps from both indices around the ground temp y-coordinate
        for j in range(0,len(ground_TC_yindices)):  
            #plot temps from both indices around the ground temp x-coordinate
            for i in range(0,len(ground_TC_xindices)):
                 
                ax1.plot(Perf_metric_dates[:],sndVizTemps[77:-1,
                         ground_TC_xindices[i],
                         ground_TC_yindices[j],
                         ground_TC_zindices[(k*2)]],
                         '--',
                         color=color_scheme[c])
                ax1.plot(Perf_metric_dates[:],sndVizTemps[77:-1,
                         ground_TC_xindices[i],
                         ground_TC_yindices[j],
                         ground_TC_zindices[(k*2)+1]],
                         color=color_scheme[c])
                
                c=c+1 #next colour
                    
               #plt.plot(sndTemps_datetimes,avg_sim_temps_snd_layer[:,2],marker='*')
                  
        ax1.set_title("Ground temperatures at depth of " + str(ground_TC_zcoords[k]) + 'm')
        ax1.set_ylabel("Temperature (\N{degree sign}C)")
        ax1.set_ylim(20,40)
        ax1.legend(['x=3.5m, y=1.75m, z=6.2m','x=3.5m, y=1.75m, z=6.4m',
                    'x=3.8m, y=1.75m, z=6.2m','x=3.8m, y=1.75m, z=6.4m',
                    'x=3.5m, y=2.35m, z=6.2m','x=3.5m, y=2.35m, z=6.4m',
                    'x=3.8m, y=2.35m, z=6.2m','x=3.8m, y=2.35m, z=6.4m'],loc='best')
        locator = mdates.AutoDateLocator()
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_minor_locator(mdates.DayLocator())
        
        # #Sets major and minor tick spacing
        # loc = ticker.MultipleLocator(base=7.0)
        # ax1.xaxis.set_major_locator(loc)
        # ax1.xaxis.set_minor_locator(ticker.MultipleLocator(base=1.0))
        
        #set the major locator on x-axis to every 6 days
        locator.intervald[mdates.DAILY]=[7] #https://matplotlib.org/stable/api/dates_api.html#matplotlib.dates.AutoDateLocator
        
        #Formatting - make the x-axis labels more consices
        #https://matplotlib.org/stable/gallery/ticks_and_spines/date_concise_formatter.html
        formatter = mdates.ConciseDateFormatter(locator)
        ax1.xaxis.set_major_formatter(formatter)
        fig.autofmt_xdate() #automatically makes the x-labels rotate
        
    
 #    # Plot experimental data on same figure
    
 #    for col in range(0,np.size(avg_expmtl_temps,1)):
 #        ax1.plot(exp_time_array,avg_expmtl_temps[:,col],marker=',')                
    
 #    #rename figure and change legend to include experimental data
 #    ax1.set_title("Mean temperatures of sand store layers, Run " + run_number)
 #    ax1.legend(['Sim-bott','Sim-mid','Sim-top','Exp-bott','Exp-mid','Exp-top',],loc='best')
    
    
 #    #Documentation on date locators
 #    #https://matplotlib.org/stable/api/dates_api.html
 #    locator = mdates.AutoDateLocator()
 #    ax1.xaxis.set_major_locator(locator)
 #    ax1.xaxis.set_minor_locator(mdates.DayLocator())
    
 #    loc = ticker.MultipleLocator(base=2.0)
 #    ax1.yaxis.set_major_locator(loc)
 #    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(base=1.0))
    
 #    # #set the major locator on x-axis to every 6 days
 #    # locator.intervald[mdates.DAILY]=[6] #https://matplotlib.org/stable/api/dates_api.html#matplotlib.dates.AutoDateLocator
    
 #    #Formatting - make the x-axis labels more consices
 #    #https://matplotlib.org/stable/gallery/ticks_and_spines/date_concise_formatter.html
 #    formatter = mdates.ConciseDateFormatter(locator)
 #    ax1.xaxis.set_major_formatter(formatter)
 #    fig1.autofmt_xdate() #automatically makes the x-labels rotate
    
 #    #gridlines on
 #    ax1.grid(True)
    
 #    #Set max and min values for axes
 #    ax1.set_xlim(exp_time_array[0], exp_time_array[-1])
 #    #ax1.set_ylim(16,24)
 #    ax1.set_ylim(30,58)
    
    

 
    
#%%% Plot a sand TC temp with GHI and Ambient temp (Outside Air)
        
    # df_temp=func_serr.read_performance_metrics_data('Performance_metrics_data_2021-05-28-to-2021-07-06.dat',dat_headers, main_dir, data_dir)
    
    # #create empty list
    # exp_time_array_days2 = [0]*df_temp.shape[0]
    # #put start date in top row of list
    # exp_time_array_days2[0] = exp_start_date
    # for d in range(1,df_temp.shape[0]):
    #     exp_time_array_days2[d] = exp_time_array_days2[d-1] + timedelta(days=int(df_temp.iloc[d,0]-df_temp.iloc[d-1,0]))
    
    # fig,ax1 = plt.subplots()
    # ax2 = ax1.twinx() #secondary axis
    # ax1.plot(exp_time_array_days2,df_temp['amb_temp'],'+',label='Ambient temp', color='b') #Ambient temp
    # ax2.plot(exp_time_array_days2,df_temp['GHI'],marker='+',label='GHI', color='g') #GHI
    # ax1.plot(exp_time_array,Exp_sand_temps['TC(A-5-1)'],label='TC(A-5-1)', color='orange') #Top layer TC A-5-1
    # ax1.set_ylabel('Temperature ($\degree$C)')
    # ax2.set_ylabel('Global Horiz. Irradiance (MJ/m2)')
    # fig.legend()

    


#%% 14. PLOT ALL NODES IN Temps, WHEN Master_v5_2.py is interrupted during solver-> A=Temps[:,:,:,0]
    # ie. The following are to be used with F9, not as part of this current program

# #Plot all node temps in y-z cross sectional plane
# A=Temps[:,:,:,0]
# import matplotlib.pyplot as plt
# for x in range(0,A.shape[0]):
#     fig1,ax1 = plt.subplots()
#     ax1.plot(range(0,A.shape[1]),A[x,:,:])    
#     ax1.set_title("y-z cross-sections, node "+ str(x))
#     ax1.set_xlabel('Length of num. domain (m)')
#     ax1.set_ylabel("Temperature (\N{DEGREE SIGN}C)")
    
# #To graph cross-sectional temps of one x-z-plane at a time, going up through all z-points
# x=37
# for z in range(0,A.shape[2]):
#     fig1,ax1 = plt.subplots()
#     ax1.plot(range(0,A.shape[1]),A[x,:,z])    
#     ax1.set_title("y cross-section temps, x node "+ str(x) + ", z node " +str(z))
#     ax1.set_xlabel('Nodes across y domain')
#     ax1.set_ylabel("Temperature (\N{DEGREE SIGN}C)")
#     ax1.set_ylim(0,60)
#     ax1.minorticks_on()
#     ax1.grid(which='both',axis='x',color='0.9')
#     ax1.axvspan(11.5,13.5, color='lightgray',alpha=0.5) #shading for concrete nodes (12,13)
#     ax1.axvspan(48.5,50.5, color='lightgray',alpha=0.5) #shading for concrete nodes (49,50)
#     ax1.axvspan(13.5,15.5, color='pink',alpha=0.5) #shading for XPS nodes (14,15)
#     ax1.axvspan(46.5,48.5, color='pink',alpha=0.5) #shading for XPS nodes (47,48)
    
# #To graph cross-sectional temps of one x-z-plane at a time, going north to south through all x-points
# z=37
# for x in range(0,A.shape[0]):
#     fig1,ax1 = plt.subplots()
#     ax1.plot(range(0,A.shape[1]),A[x,:,z])    
#     ax1.set_title("y cross-section temps, x node "+ str(x) + ", z node " +str(z))
#     ax1.set_xlabel('Nodes across y domain')
#     ax1.set_ylabel("Temperature (\N{DEGREE SIGN}C)")
#     ax1.set_ylim(0,60)
#     ax1.minorticks_on()
#     ax1.grid(which='both',axis='x',color='0.9')
#     ax1.axvspan(11.5,13.5, color='lightgray',alpha=0.5) #shading for concrete nodes (12,13)
#     ax1.axvspan(48.5,50.5, color='lightgray',alpha=0.5) #shading for concrete nodes (49,50)
#     ax1.axvspan(13.5,15.5, color='pink',alpha=0.5) #shading for XPS nodes (14,15)
#     ax1.axvspan(46.5,48.5, color='pink',alpha=0.5) #shading for XPS nodes (47,48)

