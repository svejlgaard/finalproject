import matplotlib.pyplot as plt
import numpy as np
import os, contextlib, sys

# Sets the directory to the current directory
os.chdir(sys.path[0])

data_dict = dict()

def loaddata(directory):
    for filename in os.listdir(directory):
        raw_data = np.genfromtxt(f'{directory}/{filename}', delimiter=',', names=True)
        data_matrix = np.genfromtxt(f'{directory}/{filename}', delimiter=',')
        date = filename.split(sep='_')[2]
        data_dict.update({f'{date}_data': data_matrix})
        data_dict.update({f'{date}_features': list(raw_data.dtype.names)})

loaddata('data')

print(data_dict)