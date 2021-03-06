import matplotlib.pyplot as plt
import numpy as np
import os, contextlib, sys

# Sets the directory to the current directory
os.chdir(sys.path[0])


data_dict = dict()

user_name = input('What is your name? [Simone/Jonathan/Marcus/Runi]')

#user_name = 'Simone'

if user_name == 'Simone':
    print('Welcome, master!')
elif user_name == 'Jonathan':
    directory = r'C:\Users\jonat\Documents\GitHub\finalproject'
    os.chdir(directory)
elif user_name == 'Runi':
    directory = r'D:\Sapientia\Dropbox\Fysik på KU\Big Data Analysis\Final Project'
    os.chdir(directory)
elif user_name == 'Marcus':
    directory = r'C:\Users\Marcus96\Documents\GitHub\finalprojectMarcus'
    os.chdir(directory)

def loaddata(directory):
        #data_dict = dict()
        for filename in os.listdir(directory):
            raw_data = np.genfromtxt(f'{directory}/{filename}', delimiter=',', names=True)
            data_matrix = np.genfromtxt(f'{directory}/{filename}', delimiter=',')
            date = filename.split(sep='_')[2]
            data_dict.update({f'{date}_data': data_matrix})
            data_dict.update({f'{date}_features': list(raw_data.dtype.names)})
    
loaddata('data')