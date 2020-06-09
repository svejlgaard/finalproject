# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 09:52:42 2020

@author: Sapientia
"""


import datetime

begin_time = datetime.datetime.now()

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

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# load dataset into Pandas DataFrame - Use your own filepath to the data
#df = pd.read_csv('D:\Sapientia\Dropbox\Fysik på KU\Big Data Analysis\Final Project\etalon_jitter_16Mar20_etalon.ccfSum-telemetry.csv',names=['JD_UTC','RV','RV_ERR','FIES_ROOM_WEBSENSOR1_TEMPERATURE_NEAR_CEILING','FIES_ROOM_WEBSENSOR1_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_TEMPERATURE','FIES_FRONT_ROOM_WEBSENSOR2_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_ILLUMINATION','FIES_INSIDE_BLACK_BOX_AT_REAR','FIES_ROOM_NEAR_CEILING_AND_FAN_CONTROL','FIES_UNDER_GRATING_IN_TANK','FIES_FRONT_ROOM','FIES_BELOW_TABLE','FIES_ABOVE_HEATER_RADIATOR','FIES_INSIDE_BLACK_BOX_CENTRE','FIES_ABOVE_CEILING','FIES_GRATING_TANK_HOUSING','FIES_INSIDE_WHITE_BOX_REAR','FIES_INSIDE_WHITE_BOX_CENTRE','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_INCLUDE_GND','FIES_CEILING_NEAR_INSTRUMENT_AARHUS_1','FIES_INSIDE_BLACK_BOX_AARHUS_2','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3','FIES_DETECTOR_TEMPERATURE','FIES_DEWAR_TEMPERATURE','FIES_DETECTOR_PRESSURE','FIES_ATMOSPHERIC_PRESSURE','FIES_GRATING_TANK_PRESSURE'],skiprows=1)
#df = pd.read_csv('D:\Sapientia\Dropbox\Fysik på KU\Big Data Analysis\Final Project\data\20200316.txt',names=['JD_UTC','RV','RV_ERR','FIES_ROOM_WEBSENSOR1_TEMPERATURE_NEAR_CEILING','FIES_ROOM_WEBSENSOR1_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_TEMPERATURE','FIES_FRONT_ROOM_WEBSENSOR2_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_ILLUMINATION','FIES_INSIDE_BLACK_BOX_AT_REAR','FIES_ROOM_NEAR_CEILING_AND_FAN_CONTROL','FIES_UNDER_GRATING_IN_TANK','FIES_FRONT_ROOM','FIES_BELOW_TABLE','FIES_ABOVE_HEATER_RADIATOR','FIES_INSIDE_BLACK_BOX_CENTRE','FIES_ABOVE_CEILING','FIES_GRATING_TANK_HOUSING','FIES_INSIDE_WHITE_BOX_REAR','FIES_INSIDE_WHITE_BOX_CENTRE','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_INCLUDE_GND','FIES_CEILING_NEAR_INSTRUMENT_AARHUS_1','FIES_INSIDE_BLACK_BOX_AARHUS_2','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3','FIES_DETECTOR_TEMPERATURE','FIES_DEWAR_TEMPERATURE','FIES_DETECTOR_PRESSURE','FIES_ATMOSPHERIC_PRESSURE','FIES_GRATING_TANK_PRESSURE'],delim_whitespace=True,skiprows=0)
df = pd.read_csv('D:\Sapientia\Dropbox\Fysik på KU\Big Data Analysis\Final Project\data\Data1345scaled.txt',names=['JD_UTC','RV','RV_ERR','FIES_ROOM_WEBSENSOR1_TEMPERATURE_NEAR_CEILING','FIES_ROOM_WEBSENSOR1_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_TEMPERATURE','FIES_FRONT_ROOM_WEBSENSOR2_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_ILLUMINATION','FIES_INSIDE_BLACK_BOX_AT_REAR','FIES_ROOM_NEAR_CEILING_AND_FAN_CONTROL','FIES_UNDER_GRATING_IN_TANK','FIES_FRONT_ROOM','FIES_BELOW_TABLE','FIES_ABOVE_HEATER_RADIATOR','FIES_INSIDE_BLACK_BOX_CENTRE','FIES_ABOVE_CEILING','FIES_GRATING_TANK_HOUSING','FIES_INSIDE_WHITE_BOX_REAR','FIES_INSIDE_WHITE_BOX_CENTRE','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_INCLUDE_GND','FIES_CEILING_NEAR_INSTRUMENT_AARHUS_1','FIES_INSIDE_BLACK_BOX_AARHUS_2','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3','FIES_DETECTOR_TEMPERATURE','FIES_DEWAR_TEMPERATURE','FIES_DETECTOR_PRESSURE','FIES_ATMOSPHERIC_PRESSURE','FIES_GRATING_TANK_PRESSURE'],delim_whitespace=True,skiprows=0)


# Separating out the features
features = ['FIES_ROOM_WEBSENSOR1_TEMPERATURE_NEAR_CEILING','FIES_ROOM_WEBSENSOR1_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_TEMPERATURE','FIES_FRONT_ROOM_WEBSENSOR2_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_ILLUMINATION','FIES_INSIDE_BLACK_BOX_AT_REAR','FIES_ROOM_NEAR_CEILING_AND_FAN_CONTROL','FIES_UNDER_GRATING_IN_TANK','FIES_FRONT_ROOM','FIES_BELOW_TABLE','FIES_ABOVE_HEATER_RADIATOR','FIES_INSIDE_BLACK_BOX_CENTRE','FIES_ABOVE_CEILING','FIES_GRATING_TANK_HOUSING','FIES_INSIDE_WHITE_BOX_REAR','FIES_INSIDE_WHITE_BOX_CENTRE','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_INCLUDE_GND','FIES_CEILING_NEAR_INSTRUMENT_AARHUS_1','FIES_INSIDE_BLACK_BOX_AARHUS_2','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3','FIES_DETECTOR_TEMPERATURE','FIES_DEWAR_TEMPERATURE','FIES_DETECTOR_PRESSURE','FIES_ATMOSPHERIC_PRESSURE','FIES_GRATING_TANK_PRESSURE']
#features = ['FIES_INSIDE_BLACK_BOX_CENTRE','FIES_GRATING_TANK_HOUSING','FIES_INSIDE_WHITE_BOX_CENTRE']
#X = df.loc[:, features].values
# Dette beholder features i forhold til SHAP!
X = df[features]
# Separating out the target
y = df.loc[:,['RV']].values
y=y[:,0]

# Ser ud til at denne måde beholder den som "dataframe", og så virker det bedre med SHAP og feature names.
# =============================================================================
# X = df[features]
# y = df['RV']
# =============================================================================


# =============================================================================
# =============================================================================
# X = StandardScaler().fit_transform(X)
# 
# # PCA Projection to 2D
# pca = PCA(n_components=5)
# principalComponents = pca.fit_transform(X)
# principalDf = pd.DataFrame(data = principalComponents
#              , columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 'principal component 5'])
# 
# # Concatenating DataFrame along axis = 1. finalDf is the final DataFrame before plotting the data.
# finalDf = pd.concat([principalDf, df[['RV']]], axis = 1)
# 
# #Visualize 2D Projection
# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1) 
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_title('2 component PCA', fontsize = 20)
# 
# # 's' is the marker size, I reduced it significantly from the tutorial (s = 50), to better distinguish the data points on the plot.
# targets = [0,1,2,3,4]
# colors = ['r', 'g', 'y','orange','black']
# for target, color in zip(targets,colors):
#     indicesToKeep = finalDf['RV'] == target
#     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
#                , finalDf.loc[indicesToKeep, 'principal component 2']
#                , c = color
#                , s = 5)
# ax.legend(targets)
# ax.grid()
# 
# # The explained variance tells you how much information (variance) can be attributed to each of the principal components.
# print(pca.explained_variance_ratio_)
# # 
# print(pca.singular_values_)
# =============================================================================

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)


# =============================================================================
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# 
# # Fit only to the training data
# scaler.fit(X_train)
# 
# 
# # Now apply the transformations to the data:
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
# =============================================================================

import shap
#import matplotlib.pyplot as plt



import numpy as np



from sklearn.linear_model import LinearRegression
from sklearn import linear_model

model = LinearRegression(fit_intercept=False, normalize=False, copy_X=True, n_jobs=None)
#model = linear_model.Lasso(alpha=0.1)
#model = linear_model.BayesianRidge()

model = model.fit(X_train,y_train)



y_pred = model.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix



from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score

print(explained_variance_score(y_test,y_pred))
print(r2_score(y_test,y_pred))


#y_test_size = y_test.size
#y_pred = np.reshape(y_pred,(y_test_size,1))
y_pred_test = np.subtract(y_test,y_pred)

plt.figure()
plt.plot(y_test,'-', color='C0', linewidth = 1 ,label="test")
plt.plot((y_pred),'-', color='C1', linewidth = 1 , label="predicted")
plt.legend()
plt.xlabel('Index (shuffled)')
plt.ylabel('RV')
plt.title("XGBoost simpel model - training sensor data on RV drift")
plt.show()


plt.figure()
plt.plot((y_pred_test), label="true - predicted")
#plt.plot(y_pred,label="predicted")
plt.legend()
plt.xlabel('Index (shuffled)')
plt.ylabel('RV')
plt.title("XGBoost simpel model - training sensor data on RV drift")
plt.show()

from numpy import mean, sqrt, square, arange
rms = sqrt(mean(square(y_pred_test)))



## For at plotte unshuffled, importeres det samme datasæt som der blev testet på i første omgang, og dette er så unshuffled da test_split funktionen ikke er brugt.

#csv
#df2 = pd.read_csv('D:\Sapientia\Dropbox\Fysik på KU\Big Data Analysis\Final Project\data\AllDatav3.txt',names=['JD_UTC','RV','RV_ERR','FIES_ROOM_WEBSENSOR1_TEMPERATURE_NEAR_CEILING','FIES_ROOM_WEBSENSOR1_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_TEMPERATURE','FIES_FRONT_ROOM_WEBSENSOR2_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_ILLUMINATION','FIES_INSIDE_BLACK_BOX_AT_REAR','FIES_ROOM_NEAR_CEILING_AND_FAN_CONTROL','FIES_UNDER_GRATING_IN_TANK','FIES_FRONT_ROOM','FIES_BELOW_TABLE','FIES_ABOVE_HEATER_RADIATOR','FIES_INSIDE_BLACK_BOX_CENTRE','FIES_ABOVE_CEILING','FIES_GRATING_TANK_HOUSING','FIES_INSIDE_WHITE_BOX_REAR','FIES_INSIDE_WHITE_BOX_CENTRE','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_INCLUDE_GND','FIES_CEILING_NEAR_INSTRUMENT_AARHUS_1','FIES_INSIDE_BLACK_BOX_AARHUS_2','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3','FIES_DETECTOR_TEMPERATURE','FIES_DEWAR_TEMPERATURE','FIES_DETECTOR_PRESSURE','FIES_ATMOSPHERIC_PRESSURE','FIES_GRATING_TANK_PRESSURE'],skiprows=1)

#txt
df2 = pd.read_csv('D:\Sapientia\Dropbox\Fysik på KU\Big Data Analysis\Final Project\data\Data2scaled.txt',names=['JD_UTC','RV','RV_ERR','FIES_ROOM_WEBSENSOR1_TEMPERATURE_NEAR_CEILING','FIES_ROOM_WEBSENSOR1_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_TEMPERATURE','FIES_FRONT_ROOM_WEBSENSOR2_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_ILLUMINATION','FIES_INSIDE_BLACK_BOX_AT_REAR','FIES_ROOM_NEAR_CEILING_AND_FAN_CONTROL','FIES_UNDER_GRATING_IN_TANK','FIES_FRONT_ROOM','FIES_BELOW_TABLE','FIES_ABOVE_HEATER_RADIATOR','FIES_INSIDE_BLACK_BOX_CENTRE','FIES_ABOVE_CEILING','FIES_GRATING_TANK_HOUSING','FIES_INSIDE_WHITE_BOX_REAR','FIES_INSIDE_WHITE_BOX_CENTRE','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_INCLUDE_GND','FIES_CEILING_NEAR_INSTRUMENT_AARHUS_1','FIES_INSIDE_BLACK_BOX_AARHUS_2','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3','FIES_DETECTOR_TEMPERATURE','FIES_DEWAR_TEMPERATURE','FIES_DETECTOR_PRESSURE','FIES_ATMOSPHERIC_PRESSURE','FIES_GRATING_TANK_PRESSURE'],delim_whitespace=True,skiprows=0)

# Separating out the features
features3 = ['FIES_ROOM_WEBSENSOR1_TEMPERATURE_NEAR_CEILING','FIES_ROOM_WEBSENSOR1_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_TEMPERATURE','FIES_FRONT_ROOM_WEBSENSOR2_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_ILLUMINATION','FIES_INSIDE_BLACK_BOX_AT_REAR','FIES_ROOM_NEAR_CEILING_AND_FAN_CONTROL','FIES_UNDER_GRATING_IN_TANK','FIES_FRONT_ROOM','FIES_BELOW_TABLE','FIES_ABOVE_HEATER_RADIATOR','FIES_INSIDE_BLACK_BOX_CENTRE','FIES_ABOVE_CEILING','FIES_GRATING_TANK_HOUSING','FIES_INSIDE_WHITE_BOX_REAR','FIES_INSIDE_WHITE_BOX_CENTRE','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_INCLUDE_GND','FIES_CEILING_NEAR_INSTRUMENT_AARHUS_1','FIES_INSIDE_BLACK_BOX_AARHUS_2','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3','FIES_DETECTOR_TEMPERATURE','FIES_DEWAR_TEMPERATURE','FIES_DETECTOR_PRESSURE','FIES_ATMOSPHERIC_PRESSURE','FIES_GRATING_TANK_PRESSURE']
#features3 = ['FIES_INSIDE_BLACK_BOX_CENTRE','FIES_GRATING_TANK_HOUSING','FIES_INSIDE_WHITE_BOX_CENTRE']
X3 = df2.loc[:, features3].values
X3 = df2[features3]
# Separating out the target
y3 = df2.loc[:,['RV']].values
y3=y3[:,0]
# =============================================================================
# X3 = df[features3]
# y3 = df['RV']
# =============================================================================

#X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.001, random_state=0, shuffle=True)
#X3_train = scaler.transform(X3_train)

#X3 = scaler.transform(X3)



y_pred3 = model.predict(X3)

from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score

print(explained_variance_score(y3,y_pred3))
print(r2_score(y3,y_pred3))

t_mean_diff_train_vali = np.mean(y_pred_test)

print("Mean difference train/validate:", np.mean(y_pred_test))

# =============================================================================
# y3_test_size = y3.size
# y_pred3 = np.reshape(y_pred3,(y3_test_size,1))
# =============================================================================
y_pred_test3 = np.subtract(y3,y_pred3)


plt.figure()
plt.plot(y3,'-', color='C0', linewidth = 1 ,label="test")
plt.plot((y_pred3),'-', color='C1', linewidth = 1 , label="predicted")
plt.legend()
plt.xlabel('Time')
plt.ylabel('RV')
plt.title("XGBoost simpel model - prediciton etalon_jitter_26Mar_whitebox_etalon (unshuffled)")
plt.show()

plt.figure()
plt.plot((y_pred_test3), label="true - predicted")
#plt.plot(y_pred,label="predicted")
plt.legend()
plt.xlabel('Index (shuffled)')
plt.ylabel('RV')
plt.title("XGBoost simpel model - training sensor data on RV drift")
plt.show()


from numpy import mean, sqrt, square, arange
rms3 = sqrt(mean(square(y_pred_test3)))

rms_sum = rms+rms3

u_mean_diff_train_test = np.mean(y_pred_test3)

print("Samlet RMS:", rms+rms3) 
print("Mean difference train/test:", np.mean(y_pred_test3))

# =============================================================================
# # Køre prediciton på et andet datasæt, i dette tilfælde det første, for at bekræfte at Shuffle ikke er afgørende i forhold til prediction.
# 
# #####
# df1 = pd.read_csv('D:\Sapientia\Dropbox\Fysik på KU\Big Data Analysis\Final Project\data\etalon_jitter_16Mar20_etalon.ccfSum-telemetryCLEANED.csv',names=['JD_UTC','RV','RV_ERR','FIES_ROOM_WEBSENSOR1_TEMPERATURE_NEAR_CEILING','FIES_ROOM_WEBSENSOR1_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_TEMPERATURE','FIES_FRONT_ROOM_WEBSENSOR2_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_ILLUMINATION','FIES_INSIDE_BLACK_BOX_AT_REAR','FIES_ROOM_NEAR_CEILING_AND_FAN_CONTROL','FIES_UNDER_GRATING_IN_TANK','FIES_FRONT_ROOM','FIES_BELOW_TABLE','FIES_ABOVE_HEATER_RADIATOR','FIES_INSIDE_BLACK_BOX_CENTRE','FIES_ABOVE_CEILING','FIES_GRATING_TANK_HOUSING','FIES_INSIDE_WHITE_BOX_REAR','FIES_INSIDE_WHITE_BOX_CENTRE','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_INCLUDE_GND','FIES_CEILING_NEAR_INSTRUMENT_AARHUS_1','FIES_INSIDE_BLACK_BOX_AARHUS_2','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3','FIES_DETECTOR_TEMPERATURE','FIES_DEWAR_TEMPERATURE','FIES_DETECTOR_PRESSURE','FIES_ATMOSPHERIC_PRESSURE','FIES_GRATING_TANK_PRESSURE'],skiprows=1)
# #df = pd.read_csv('D:\Sapientia\Dropbox\Fysik på KU\Big Data Analysis\Final Project\data\AllData.txt',names=['JD_UTC','RV','RV_ERR','FIES_ROOM_WEBSENSOR1_TEMPERATURE_NEAR_CEILING','FIES_ROOM_WEBSENSOR1_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_TEMPERATURE','FIES_FRONT_ROOM_WEBSENSOR2_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_ILLUMINATION','FIES_INSIDE_BLACK_BOX_AT_REAR','FIES_ROOM_NEAR_CEILING_AND_FAN_CONTROL','FIES_UNDER_GRATING_IN_TANK','FIES_FRONT_ROOM','FIES_BELOW_TABLE','FIES_ABOVE_HEATER_RADIATOR','FIES_INSIDE_BLACK_BOX_CENTRE','FIES_ABOVE_CEILING','FIES_GRATING_TANK_HOUSING','FIES_INSIDE_WHITE_BOX_REAR','FIES_INSIDE_WHITE_BOX_CENTRE','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_INCLUDE_GND','FIES_CEILING_NEAR_INSTRUMENT_AARHUS_1','FIES_INSIDE_BLACK_BOX_AARHUS_2','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3','FIES_DETECTOR_TEMPERATURE','FIES_DEWAR_TEMPERATURE','FIES_DETECTOR_PRESSURE','FIES_ATMOSPHERIC_PRESSURE','FIES_GRATING_TANK_PRESSURE'],delim_whitespace=True,skiprows=0)
# 
# # Separating out the features
# features2 = ['FIES_ROOM_WEBSENSOR1_TEMPERATURE_NEAR_CEILING','FIES_ROOM_WEBSENSOR1_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_TEMPERATURE','FIES_FRONT_ROOM_WEBSENSOR2_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_ILLUMINATION','FIES_INSIDE_BLACK_BOX_AT_REAR','FIES_ROOM_NEAR_CEILING_AND_FAN_CONTROL','FIES_UNDER_GRATING_IN_TANK','FIES_FRONT_ROOM','FIES_BELOW_TABLE','FIES_ABOVE_HEATER_RADIATOR','FIES_INSIDE_BLACK_BOX_CENTRE','FIES_ABOVE_CEILING','FIES_GRATING_TANK_HOUSING','FIES_INSIDE_WHITE_BOX_REAR','FIES_INSIDE_WHITE_BOX_CENTRE','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_INCLUDE_GND','FIES_CEILING_NEAR_INSTRUMENT_AARHUS_1','FIES_INSIDE_BLACK_BOX_AARHUS_2','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3','FIES_DETECTOR_TEMPERATURE','FIES_DEWAR_TEMPERATURE','FIES_DETECTOR_PRESSURE','FIES_ATMOSPHERIC_PRESSURE','FIES_GRATING_TANK_PRESSURE']
# #features = ['FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3','FIES_DETECTOR_TEMPERATURE','FIES_DEWAR_TEMPERATURE','FIES_DETECTOR_PRESSURE','FIES_ATMOSPHERIC_PRESSURE','FIES_GRATING_TANK_PRESSURE']
# X2 = df1.loc[:, features2].values
# # Separating out the target
# y2 = df1.loc[:,['RV']].values
# #y=y[:,0]
# 
# 
# X2 = scaler.transform(X2)
# 
# 
# D_test2 = xgb.DMatrix(X2, label=y2)
# 
# 
# y_pred2 = model.predict(D_test2)
# 
# from sklearn.metrics import explained_variance_score
# from sklearn.metrics import r2_score
# 
# print(explained_variance_score(y2,y_pred2))
# print(r2_score(y2,y_pred2))
# 
# y2_test_size = y2.size
# y_pred2 = np.reshape(y_pred2,(y2_test_size,1))
# y_pred_test2 = np.subtract(y2,y_pred2)
# 
# plt.figure()
# plt.plot(y2,'-', color='C0', linewidth = 1 ,label="test")
# plt.plot((y_pred2),'-', color='C1', linewidth = 1 , label="predicted")
# plt.legend()
# plt.xlabel('Time')
# plt.ylabel('RV')
# plt.title("XGBoost simpel model - training sensor data on RV drift")
# plt.show()
# 
# plt.figure()
# plt.plot((y_pred_test2), label="true - predicted")
# #plt.plot(y_pred,label="predicted")
# plt.legend()
# plt.xlabel('Time')
# plt.ylabel('RV')
# plt.title("XGBoost simpel model - training sensor data on RV drift")
# plt.show()
# 
# from numpy import mean, sqrt, square, arange
# rms2 = sqrt(mean(square(y_pred_test2)))
# =============================================================================

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
explainer = shap.LinearExplainer(model, X)
shap_values = explainer.shap_values(X)

# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
#shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:],matplotlib=True)


# visualize the training set predictions
#shap.force_plot(explainer.expected_value, shap_values, X,matplotlib=True)


# create a dependence plot to show the effect of a single feature across the whole dataset
#shap.dependence_plot("p_weta2", shap_values, X)

# summarize the effects of all the features

plt.figure()
shap.summary_plot(shap_values, X)
plt.show()

plt.figure()
shap.summary_plot(shap_values, X, plot_type="bar")
plt.show()


# Hvordan de forskellige features interacter
#plt.figure()
#shap.interaction = shap.LinearExplainer(model, X).shap_interaction_values(X)
#plt.show()

# Hvis man vil øge antal features displayet
#shap.summaryplot(shapvalues, Ximportance , classnames=classes ,max_display=40)


#shap.dependence_plot("FIES_ATMOSPHERIC_PRESSURE", shap_values, X)


#FIES_INSIDE_BLACK_BOX_CENTRE
#FIES_INSIDE_BLACK_BOX_AT_REAR
#FIES_ROOM_WEBSENSOR1_HUMIDITY


# =============================================================================
# #Histogram af prediction error
# plt.figure()
# plt.hist(y_pred_test, bins=100, histtype='bar',rwidth=0.92)
# plt.xlabel('RV prediciton error')
# plt.ylabel('Counts')
# plt.title("Histogram af prediction error")
# plt.show()
# =============================================================================

# =============================================================================
# Temp1 = df.loc[:,['FIES_ROOM_WEBSENSOR1_TEMPERATURE_NEAR_CEILING']].values
# Temp2 = df.loc[:,['FIES_FRONT_ROOM_WEBSENSOR2_TEMPERATURE']].values
# Temp3 = df.loc[:,['FIES_INSIDE_BLACK_BOX_AT_REAR']].values
# Temp5 = df.loc[:,['FIES_ABOVE_HEATER_RADIATOR']].values
# 
# Illum1 = df.loc[:,['FIES_FRONT_ROOM_WEBSENSOR2_ILLUMINATION']].values
# 
# 
# 
# plt.figure()
# plt.plot(y,'-', color='C0', linewidth = 1 ,label="RV")
# plt.plot((Temp1),'-', color='C1', linewidth = 1 , label="Temp1")
# plt.plot((Temp2),'-', color='C2', linewidth = 1 , label="Temp2")
# plt.plot((Temp3),'-', color='C3', linewidth = 1 , label="Temp3")
# plt.plot((Illum1),'-', color='C4', linewidth = 1 , label="Illum1")
# plt.plot((Temp5),'-', color='C5', linewidth = 1 , label="Temp5")
# plt.legend()
# plt.xlabel('Time')
# plt.ylabel('RV, Temp')
# plt.title("Plot af Temp og RV")
# plt.show()
# =============================================================================

print(datetime.datetime.now() - begin_time)