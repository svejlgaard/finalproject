
# =============================================================================
# import matplotlib.pyplot as plt
# import numpy as np
# import os, contextlib, sys
# 
# # Sets the directory to the current directory
# os.chdir('C:\Users\'+os.getlogin()+'\Documents\GitHub\finalproject')
# 
# def loaddata(directory):
#     data_dict = dict()
#     for filename in os.listdir(directory):
#         raw_data = np.genfromtxt(f'{directory}/{filename}', delimiter=',', names=True)
#         data_matrix = np.genfromtxt(f'{directory}/{filename}', delimiter=',')
#         date = filename.split(sep='_')[2]
#         data_dict.update({f'{date}_data': data_matrix})
#         data_dict.update({f'{date}_features': list(raw_data.dtype.names)})
# 
# loaddata('data')
# =============================================================================

# =============================================================================
# import csv
# import numpy
# 
# with open('etalon_jitter_16Mar20_etalon.ccfSum-telemetry') as input_file:
#     raw_data = numpy.array([row for row in csv.reader(input_file)]).astype(numpy.float)
# 
# labels = raw_data[:, 0 ]
# data   = raw_data[:, 1:]
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# load dataset into Pandas DataFrame - Use your own filepath to the data
df = pd.read_csv('D:\Sapientia\Dropbox\Fysik på KU\Big Data Analysis\Final Project\etalon_jitter_16Mar20_etalon.ccfSum-telemetry.csv',names=['JD_UTC','RV','RV_ERR','FIES_ROOM_WEBSENSOR1_TEMPERATURE_NEAR_CEILING','FIES_ROOM_WEBSENSOR1_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_TEMPERATURE','FIES_FRONT_ROOM_WEBSENSOR2_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_ILLUMINATION','FIES_INSIDE_BLACK_BOX_AT_REAR','FIES_ROOM_NEAR_CEILING_AND_FAN_CONTROL','FIES_UNDER_GRATING_IN_TANK','FIES_FRONT_ROOM','FIES_BELOW_TABLE','FIES_ABOVE_HEATER_RADIATOR','FIES_INSIDE_BLACK_BOX_CENTRE','FIES_ABOVE_CEILING','FIES_GRATING_TANK_HOUSING','FIES_INSIDE_WHITE_BOX_REAR','FIES_INSIDE_WHITE_BOX_CENTRE','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_INCLUDE_GND','FIES_CEILING_NEAR_INSTRUMENT_AARHUS_1','FIES_INSIDE_BLACK_BOX_AARHUS_2','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3','FIES_DETECTOR_TEMPERATURE','FIES_DEWAR_TEMPERATURE','FIES_DETECTOR_PRESSURE','FIES_ATMOSPHERIC_PRESSURE','FIES_GRATING_TANK_PRESSURE'],skiprows=1)

# Separating out the features
features = ['JD_UTC','RV_ERR','FIES_ROOM_WEBSENSOR1_TEMPERATURE_NEAR_CEILING','FIES_ROOM_WEBSENSOR1_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_TEMPERATURE','FIES_FRONT_ROOM_WEBSENSOR2_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_ILLUMINATION','FIES_INSIDE_BLACK_BOX_AT_REAR','FIES_ROOM_NEAR_CEILING_AND_FAN_CONTROL','FIES_UNDER_GRATING_IN_TANK','FIES_FRONT_ROOM','FIES_BELOW_TABLE','FIES_ABOVE_HEATER_RADIATOR','FIES_INSIDE_BLACK_BOX_CENTRE','FIES_ABOVE_CEILING','FIES_GRATING_TANK_HOUSING','FIES_INSIDE_WHITE_BOX_REAR','FIES_INSIDE_WHITE_BOX_CENTRE','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_INCLUDE_GND','FIES_CEILING_NEAR_INSTRUMENT_AARHUS_1','FIES_INSIDE_BLACK_BOX_AARHUS_2','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3','FIES_DETECTOR_TEMPERATURE','FIES_DEWAR_TEMPERATURE','FIES_DETECTOR_PRESSURE','FIES_ATMOSPHERIC_PRESSURE','FIES_GRATING_TANK_PRESSURE']
X = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['RV']].values

# =============================================================================
X = StandardScaler().fit_transform(X)

# PCA Projection to 2D
pca = PCA(n_components=5)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 'principal component 5'])

# Concatenating DataFrame along axis = 1. finalDf is the final DataFrame before plotting the data.
finalDf = pd.concat([principalDf, df[['RV']]], axis = 1)

#Visualize 2D Projection
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

# 's' is the marker size, I reduced it significantly from the tutorial (s = 50), to better distinguish the data points on the plot.
targets = [0,1,2,3,4]
colors = ['r', 'g', 'y','orange','black']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['RV'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 5)
ax.legend(targets)
ax.grid()

# The explained variance tells you how much information (variance) can be attributed to each of the principal components.
print(pca.explained_variance_ratio_)
# 
print(pca.singular_values_)