import os
import csv
import sys 
from pandas import read_csv

dir = sys.argv[1]
Nt = int(sys.argv[2])

home_dir = os.getcwd()

files_to_keep = ['traces.csv', 'data.csv', 'input.toml', 'slice.dat', 'tr.csv', 'corr.csv']

# eliminates last row of field file (in case the simulation ran out of time)
for f in os.listdir(dir):
	os.chdir(dir + "/" + f)
	if not "tr.csv" in os.listdir():
		input = open('traces.csv', 'r')
		output = open('tr.csv', 'w')
		writer = csv.writer(output)
		for row in csv.reader(input):
			if len(row) == 3:
				writer.writerow(row)
		input.close()
		output.close()
	#os.remove("traces.csv")
	os.chdir(home_dir)

# eliminates last correlator data (in case the simulation ran out of time)
'''for f in os.listdir(dir):
	os.chdir(dir + "/" + f)
	if not "corr.csv" in os.listdir():
		input = open('data.csv', 'r')
		row_count = sum(1 for row in input)  # fileObject is your csv.reader
		input.close()
		input = open('data.csv', 'r')
		n_correlators = int((row_count - 1) / Nt)
		output = open('corr.csv', 'w')
		writer = csv.writer(output)
		count = 0
		for row in csv.reader(input):
			if count <= n_correlators*Nt:
				writer.writerow(row)
			count += 1
		input.close()
		output.close()
	#os.remove("data.csv")
	os.chdir(home_dir)'''

# eliminates useless files and folder
'''for f in os.listdir:
	os.chdir(dir + "/" + f)
	for subf in os.listdir():
		if not subf in files_to_keep:
			os.remove(subf)
	os.chdir(home_dir)'''
