
import h5py
import numpy as np
import os, fnmatch
from numpy import linalg as LA
import copy 



#read in hdf---------------------------------------------------------


#read hdf and save each configuration in config array
def get_configs_from_file(file):
    with h5py.File(file,'r') as f:
        alist = []
        list_keys = list(f.keys())
        #print('List of datasets: ', list_keys)

        data = f.get(list_keys[0])
        config_keys = list(data.keys())
        
        print(data)
        
        configs = []
    
    
        for nb_config in config_keys:
            configs.append(np.array(data.get(nb_config).get("fields")).view(complex))
            
        
    return np.array(configs)


#several configs form hdf folder
def get_configs_from_hdf_number(file_name, therm, load_max,ac_time, dimensions):

    with h5py.File(file_name,'r') as f:
        list_keys = list(f.keys())
        data = f.get(list_keys[0])
    
        config_keys = list(data.keys())

        to_get_configs = config_keys[therm:load_max:ac_time]
        field_configs = np.zeros( (len(to_get_configs), dimensions[0], dimensions[1]) )
        for k, nb_config in enumerate( to_get_configs ):
            field_configs[k] = np.array(data.get(nb_config).get("fields")).view(complex).real[:,:,0]
            
    return field_configs



##get time slices via hdf
def get_time_slices_from_hdf(file_name, field_axis):
    hdf_configs = get_configs_from_file(file_name) #with sigma,pi1,pi2,pi3
    field_configs =  hdf_configs.take(indices=field_axis, axis=5)

    axis = tuple([i+2 for i in range(len(field_configs.shape)-2)]) #do not average over configs and time
    TimeSlices_hdf = field_configs.mean(axis).real

    return TimeSlices_hdf


#read in time slices generated on the fly
def get_time_slices_from_timeslicefile(file_name, field_axis=0, return_all=True):

    with open(file_name,encoding = "ISO-8859-1") as logfile:       
        #Need to check if the pions are already included in the time_slice output
        for i, line in enumerate(logfile):
            if i == 4: break    
        
        if not line.strip():        #If 4th row is empty, then the pions are included

            try:
                TimeSlices_fly = np.loadtxt(file_name)        #test if the last row is complete, if not delete 4 last rows
            except ValueError:
                #TimeSlices_fly = np.loadtxt(open(file_name).readlines()[:-1], skiprows=0, dtype=None)   #skips last row
                TimeSlices_fly = np.genfromtxt(file_name, skip_header=0, skip_footer=4)                #can be used to skip multiple rows

            if return_all:
                return TimeSlices_fly
            else:
                return TimeSlices_fly[field_axis::4]


        else:        #only time slices for sigma
            if field_axis!=0:
                "Only Sigma Time slices were saved!"

            try:
                TimeSlices_fly = np.loadtxt(file_name)        #test if the last row is complete, if not delete last row
            except ValueError:
                #TimeSlices_fly = np.loadtxt(open(file_name).readlines()[:-1], skiprows=0, dtype=None)   #skips last row
                TimeSlices_fly = np.genfromtxt(file_name, skip_header=0, skip_footer=1)                #can be used to skip multiple rows

        return TimeSlices_fly





#read in slurm output file -----------------------------------------------


#read in the columns of observables
def get_observables(file_name, number_columns):

    observables_raw = []
    with open(file_name) as logfile:
        for linenumber,line in enumerate(logfile):
            splitted_line= line.split()
            if len(splitted_line)==number_columns and line[0]!="#" and line[0].isdigit():
                columns = [float(i) for i in splitted_line]
                observables_raw.append(columns)

    observables_raw = np.array(observables_raw)
    print(observables_raw.shape)
    return np.array(observables_raw)


#Before Stoch_quant_paper:
#read in the columns of observables
"""
def get_observables(file_name, number_columns):

    observables_raw = []
    with open(file_name,encoding='ISO-8859-1') as logfile:

        for linenumber,line in enumerate(logfile):
            splitted_line= line.split()
            if len(splitted_line)==number_columns:
                columns = [float(i) for i in splitted_line]
                observables_raw.append(columns)

    observables_raw = np.array(observables_raw)
    print(observables_raw.shape)
    return np.array(observables_raw)
"""




"""old stuff

#find first and last line of the output of observables
#and give number of columns
#sucht nach anfang und ende vom file
def get_output_lines(file_name, number_columns):
    linenumber_output_begin = np.nan
    linenumber_output_end = np.nan
    old_output_end = np.nan
    with open(file_name,encoding = "ISO-8859-1") as logfile:        #,encoding = "ISO-8859-1"
        
        for linenumber,line in enumerate(logfile):
            old_output_end = copy.copy(linenumber_output_end)
            if len(line.split())==number_columns and line:
                if np.isnan(linenumber_output_begin):
                    linenumber_output_begin = linenumber
                
                linenumber_output_end = linenumber
            if (old_output_end == linenumber_output_end) and not(np.isnan(linenumber_output_end)):
                break
             
            
        print("Linenumber where observable output starts: ", linenumber_output_begin)
        print("Linenumber where observable output ends: ", linenumber_output_end)
        
        
    return linenumber_output_begin, linenumber_output_end, number_columns



#read in the columns of observables
def get_observables(file_name, number_columns):
    linenumber_output_begin, linenumber_output_end, number_columns = get_output_lines(file_name, number_columns)
    
    #test before if it is an error output
    if np.isnan(linenumber_output_end):
        observables_raw = np.empty((1,number_columns))
        observables_raw[:] = np.nan
    
    else:
        observables_raw = np.zeros((linenumber_output_end-linenumber_output_begin+1,number_columns))
        line_idx = 0
        with open(file_name,encoding='ISO-8859-1') as logfile:

            for linenumber,line in enumerate(logfile):
                if linenumber>= linenumber_output_begin and linenumber<=linenumber_output_end:
                    splitted_line = line.split()

                    for n_column in range(number_columns):
                        observables_raw[line_idx][n_column] = splitted_line[n_column] 
                    line_idx+=1
                
    return observables_raw


#sucht nach allen Zeilen mit columns anzahl =28
def get_output_lines_new(file_name, number_columns):
    lines_to_include = []
    
    with open(file_name,encoding = "ISO-8859-1") as logfile:        #,encoding = "ISO-8859-1"
        
        for linenumber,line in enumerate(logfile):
            if len(line.split())==number_columns:
                lines_to_include.append(linenumber)
        
    return np.array(lines_to_include)


#read in the columns of observables
def get_observables_new(file_name, number_columns):
    lines_to_include= get_output_lines_new(file_name, number_columns)
      
    number_of_configs = len(lines_to_include)
          
    #test before if it is an error output
    if number_of_configs==0:
        observables_raw = np.empty((1,number_columns))
        observables_raw[:] = np.nan
    
    else:
        observables_raw = np.zeros((number_of_configs,number_columns))
        line_idx = 0
        with open(file_name,encoding='ISO-8859-1') as logfile:

            for linenumber,line in enumerate(logfile):
                if np.isin(linenumber, lines_to_include):
                    splitted_line = line.split()

                    for n_column in range(number_columns):
                        observables_raw[line_idx][n_column] = splitted_line[n_column] 
                    line_idx+=1
                
    return observables_raw


"""
