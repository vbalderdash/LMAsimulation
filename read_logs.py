import pandas as pd
import numpy as np
import subprocess
import os

def hex2page(value):
	new_value = -111+int(value, 16)*0.488
	return new_value

def check_log(filename):
	"""Checks to make sure the lines in the file have the correct length.

	Files with less than 26 errored lines will be copied to 
	[filename]_original	and [filename] will be rewritten without the 
	errored lines.

	This function will not run on a Windows platform!
	"""
	i=0
	j=0
	f=open(filename)
	for line_no, line in enumerate(f):
	    if np.all(len(line.strip()) != np.array([47, 83])):
	    	print line
	        i+=1
	f.close()
	if i>26:
		print "%s may be in a different format!" %(filename)
		print "Moving to '_original' and ignoring!" 
		subprocess.call(['mv', filename, filename+'_original'])
	if (i>0) & (i<26):
		if os.path.isfile(filename+'_original'):
		    print "Original copies already exists for %s!" %(filename)
		else:
			f = open(filename)
			print "%s has some bad lines," %(filename)
			print "original copy will be made and bad lines will be removed" 
			
			subprocess.call(['cp', filename, filename+'_original'])
			for line_no, line in enumerate(f):
				if np.all(len(line.strip()) != np.array([47, 83])):
					subprocess.call(['sed', '-i.bak', "%s d" %(line_no+1-j), 
						            filename])
					j+=1
			subprocess.call(['rm', filename+'.bak'])
			f.close()

def parsing(filename, T_set='False'):
	"""filename must be in path/TxYYMMDD format. Returns Pandas dataframe 

	The log file will be run through a checker to make sure that there are 
	no bad lines.

	Thresholds will be converted from hex format to dBm
	
	If T_set is set to 'True' only the thresholds, latitudes, longitudes and 
	altitudes will be returned with the station identifier as a suffix, 
	otherwise the entire log file will be parsed.
	"""
	check_log(filename)
	if os.path.isfile(filename):
		dateparse = lambda x: pd.datetime.strptime(x, '%m/%d/%y %H:%M:%S')
		if T_set=='True':
			df = pd.read_fwf(filename, 
		                    widths=[1,18,4,5,12,7,
		                            3,3,3,9,10,8],
		                    names=['ID','Datetime','Version','Threshold','?',
		                           'Triggers','GPS_Number','GPS_Mode','Temp',
		                           'Lat','Lon','Alt'],
		                    usecols=[1,3,9,10,11],
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
		                    widths=[1,18,4,5,12,7,
		                            3,3,3,9,10,8],
		                    names=['ID','Datetime','Version','Threshold','?',
		                           'Triggers','GPS_Number','GPS_Mode','Temp',
		                           'Lat','Lon','Alt'],
		                    parse_dates = [1],
		                    date_parser = dateparse,
		                    na_values='\n')
			df['Threshold'] = df['Threshold'].apply(hex2page)
		df=df.set_index('Datetime')
		return df