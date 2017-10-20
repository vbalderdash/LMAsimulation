import pandas as pd
import numpy as np
import subprocess
import os

def hex2page(value):
    new_value = -111+int(value, 16)*0.488
    return new_value

def check_log(filename,form='wtreg'):
    """Checks to make sure the lines in the file have the correct length.

    Files with less than 26 errored lines will be copied to 
    [filename]_original and [filename] will be rewritten without the 
    errored lines.

    This function will not run on a Windows platform!
    """
    i=0
    j=0
    f=open(filename)
    if form=='wtreg':
        len_arrays = np.array([47, 83])
    if form=='old7':
        len_arrays = np.array([42, 78])
    if form=='newok':
        len_arrays = np.array([47, 88])
    for line_no, line in enumerate(f):
        if np.all(len(line.strip()) != len_arrays):
            print (line)
            i+=1
    f.close()
    if i>26:
        print ("%s may be in a different format!" %(filename))
        print ("Moving to '_original' and ignoring!") 
        subprocess.call(['mv', filename, filename+'_original'])
    if (i>0) & (i<26):
        if os.path.isfile(filename+'_original'):
            print ("Original copies already exists for %s!" %(filename))
        else:
            f = open(filename)
            print ("%s has some bad lines," %(filename))
            print ("original copy will be made and bad lines will be removed" )
            
            subprocess.call(['cp', filename, filename+'_original'])
            for line_no, line in enumerate(f):
                if np.all(len(line.strip()) != len_arrays):
                    subprocess.call(['sed', '-i.bak', "%s d" %(line_no+1-j), 
                                    filename])
                    j+=1
            subprocess.call(['rm', filename+'.bak'])
            f.close()

def parsing(filename, T_set='False',form='wtreg'):
    """filename must be in path/TxYYMMDD format. Returns Pandas dataframe 

    The log file will be run through a checker to make sure that there are 
    no bad lines.

    Thresholds will be converted from hex format to dBm
    
    If T_set is set to 'True' only the thresholds, latitudes, longitudes and 
    altitudes will be returned with the station identifier as a suffix, 
    otherwise the entire log file will be parsed.
    """
    check_log(filename,form)
    if os.path.isfile(filename):
        dateparse = lambda x: pd.datetime.strptime(x, '%m/%d/%y %H:%M:%S')
        namelist = ['ID','Datetime','Version','Threshold','?',
                                   'Triggers','GPS_Number','GPS_Mode','Temp',
                                   'Lat','Lon','Alt']
        if form=='wtreg':
            widths_list = [1,18,4,5,12,7,3,3,3,9,10,8]
            collist = [1,3,9,10,11]
        if form=='old7':
            widths_list = [1,18,4,5,7,7,3,3,3,9,10,8]
            collist = [1,3,9,10,11]
        if form=='newok':
            widths_list = [1,18,4,5,12,7,3,3,4,4,9,10,8]
            collist = [1,3,10,11,12]
            namelist = ['ID','Datetime','Version','Threshold','???',
                       'Triggers','GPS_Number','GPS_Mode','Temp','Batt',
                       'Lat','Lon','Alt']
        if T_set=='True':
            df = pd.read_fwf(filename, 
                            widths=widths_list,
                            names=namelist,
                            usecols=collist,
                            parse_dates = [0],
                            date_parser = dateparse,
                            na_values='\n')
            station=filename[-7]
            df['Threshold'] = df['Threshold'].apply(hex2page)
            df=df.rename(columns = {'Threshold':'Threshold_%s'%station,
                                    'Lat':'Lat_%s'%station,
                                    'Lon':'Lon_%s'%station,
                                    'Alt':'Alt_%s'%station})
        else:
            df = pd.read_fwf(filename, 
                            widths=widths_list,
                            names=namelist,
                            parse_dates = [1],
                            date_parser = dateparse,
                            na_values='\n')
            df['Threshold'] = df['Threshold'].apply(hex2page)
        df=df.set_index('Datetime')
        return df

def parsing_variable(filename, T_set='False'):
    """filename must be in path/TxYYMMDD format. Returns Pandas dataframe 

    The log file will NOT be run through a checker to make sure that there are 
    no bad lines as not all files will be the same widths for the quick check
    used in the check_log function.

    Thresholds will be converted from hex format to dBm
    
    If T_set is set to 'True' only the thresholds, latitudes, longitudes and 
    altitudes will be returned with the station identifier as a suffix, 
    otherwise the entire log file will be parsed.
    """
    if os.path.isfile(filename):
        dateparse = lambda x: pd.datetime.strptime(x, '%m/%d/%y %H:%M:%S')
        namelist = ['ID','Date','time','Version','Threshold','?','??'
                                   'Triggers','GPS_Number','GPS_Mode','Temp',
                                   'Lat','Lon','Alt']
        collist = [1,2,4,12,13,14]
        namelist = ['ID','Date','time','Version','Threshold','?','??',
                   'Triggers','GPS_Number','GPS_Mode','Temp','Batt',
                   'Lat','Lon','Alt']
        if T_set=='True':
            df = pd.read_fwf(filename, 
                            names=namelist,
                            usecols=collist,
                            parse_dates = [[0,1]],
                            date_parser = dateparse,
                            na_values='\n')
            station=filename[-7]
            df['Threshold'] = df['Threshold'].apply(hex2page)
            df=df.rename(columns = {'Threshold':'Threshold_%s'%station,
                                    'Lat':'Lat_%s'%station,
                                    'Lon':'Lon_%s'%station,
                                    'Alt':'Alt_%s'%station})
        else:
            df = pd.read_fwf(filename, 
                            names=namelist,
                            parse_dates = [[0,1]],
                            date_parser = dateparse,
                            na_values='\n')
            df['Threshold'] = df['Threshold'].apply(hex2page)
        df=df.set_index('Date_time')
        return df