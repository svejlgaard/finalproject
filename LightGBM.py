
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# load dataset into Pandas DataFrame - Use your own filepath to the data
#df = pd.read_csv('D:\Sapientia\Dropbox\Fysik på KU\Big Data Analysis\Final Project\etalon_jitter_16Mar20_etalon.ccfSum-telemetry.csv',names=['JD_UTC','RV','RV_ERR','FIES_ROOM_WEBSENSOR1_TEMPERATURE_NEAR_CEILING','FIES_ROOM_WEBSENSOR1_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_TEMPERATURE','FIES_FRONT_ROOM_WEBSENSOR2_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_ILLUMINATION','FIES_INSIDE_BLACK_BOX_AT_REAR','FIES_ROOM_NEAR_CEILING_AND_FAN_CONTROL','FIES_UNDER_GRATING_IN_TANK','FIES_FRONT_ROOM','FIES_BELOW_TABLE','FIES_ABOVE_HEATER_RADIATOR','FIES_INSIDE_BLACK_BOX_CENTRE','FIES_ABOVE_CEILING','FIES_GRATING_TANK_HOUSING','FIES_INSIDE_WHITE_BOX_REAR','FIES_INSIDE_WHITE_BOX_CENTRE','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_INCLUDE_GND','FIES_CEILING_NEAR_INSTRUMENT_AARHUS_1','FIES_INSIDE_BLACK_BOX_AARHUS_2','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3','FIES_DETECTOR_TEMPERATURE','FIES_DEWAR_TEMPERATURE','FIES_DETECTOR_PRESSURE','FIES_ATMOSPHERIC_PRESSURE','FIES_GRATING_TANK_PRESSURE'],skiprows=1)
df = pd.read_csv('D:\Sapientia\Dropbox\Fysik på KU\Big Data Analysis\Final Project\data\AllData234.txt',names=['JD_UTC','RV','RV_ERR','FIES_ROOM_WEBSENSOR1_TEMPERATURE_NEAR_CEILING','FIES_ROOM_WEBSENSOR1_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_TEMPERATURE','FIES_FRONT_ROOM_WEBSENSOR2_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_ILLUMINATION','FIES_INSIDE_BLACK_BOX_AT_REAR','FIES_ROOM_NEAR_CEILING_AND_FAN_CONTROL','FIES_UNDER_GRATING_IN_TANK','FIES_FRONT_ROOM','FIES_BELOW_TABLE','FIES_ABOVE_HEATER_RADIATOR','FIES_INSIDE_BLACK_BOX_CENTRE','FIES_ABOVE_CEILING','FIES_GRATING_TANK_HOUSING','FIES_INSIDE_WHITE_BOX_REAR','FIES_INSIDE_WHITE_BOX_CENTRE','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_INCLUDE_GND','FIES_CEILING_NEAR_INSTRUMENT_AARHUS_1','FIES_INSIDE_BLACK_BOX_AARHUS_2','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3','FIES_DETECTOR_TEMPERATURE','FIES_DEWAR_TEMPERATURE','FIES_DETECTOR_PRESSURE','FIES_ATMOSPHERIC_PRESSURE','FIES_GRATING_TANK_PRESSURE'],delim_whitespace=True,skiprows=0)

# Separating out the features
features = ['JD_UTC','FIES_ROOM_WEBSENSOR1_TEMPERATURE_NEAR_CEILING','FIES_ROOM_WEBSENSOR1_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_TEMPERATURE','FIES_FRONT_ROOM_WEBSENSOR2_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_ILLUMINATION','FIES_INSIDE_BLACK_BOX_AT_REAR','FIES_ROOM_NEAR_CEILING_AND_FAN_CONTROL','FIES_UNDER_GRATING_IN_TANK','FIES_FRONT_ROOM','FIES_BELOW_TABLE','FIES_ABOVE_HEATER_RADIATOR','FIES_INSIDE_BLACK_BOX_CENTRE','FIES_ABOVE_CEILING','FIES_GRATING_TANK_HOUSING','FIES_INSIDE_WHITE_BOX_REAR','FIES_INSIDE_WHITE_BOX_CENTRE','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_INCLUDE_GND','FIES_CEILING_NEAR_INSTRUMENT_AARHUS_1','FIES_INSIDE_BLACK_BOX_AARHUS_2','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3','FIES_DETECTOR_TEMPERATURE','FIES_DEWAR_TEMPERATURE','FIES_DETECTOR_PRESSURE','FIES_ATMOSPHERIC_PRESSURE','FIES_GRATING_TANK_PRESSURE']
#features = ['FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3','FIES_DETECTOR_TEMPERATURE','FIES_DEWAR_TEMPERATURE','FIES_DETECTOR_PRESSURE','FIES_ATMOSPHERIC_PRESSURE','FIES_GRATING_TANK_PRESSURE']
X = df.loc[:, features].values
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


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Fit only to the training data
scaler.fit(X_train)


# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

import xgboost as xgb
from xgboost.sklearn import XGBRegressor
import shap
#import matplotlib.pyplot as plt

# load JS visualization code to notebook
#shap.initjs()

D_train = xgb.DMatrix(X_train, label=y_train)
D_test = xgb.DMatrix(X_test, label=y_test)


# =============================================================================
# # train XGBoost model
# model = xgb.train({"learning_rate": 0.01}, xgb.DMatrix(X_train, label=y_train), 100)
# =============================================================================

import numpy as np

# Cluster classifier:
# =============================================================================
# param = {
#     'eta': 0.3, 
#     'max_depth': 3,  
#     'objective': 'multi:softprob',  
#     'num_class': 3} 
# =============================================================================

import lightgbm as lgb
from lightgbm import LGBMRegressor
import shap

D_train = lgb.Dataset(X_train, label=y_train)
D_test = lgb.Dataset(X_test, label=y_test)


params = {
    "max_bin": 512,
    "learning_rate": 0.5,
    "boosting_type": "gbdt",
    "objective": "regression",
    "num_leaves": 15,
    "verbose": 0,
    "min_data": 50,
    "boost_from_average": True
}

LGBMRegressor()
model = lgb.train(params, D_train, 1000, valid_sets=[D_test], early_stopping_rounds=50, verbose_eval=100)

# =============================================================================
# params = {
#     "eta": 0.5,
#     "max_depth": 4,
#     "objective": "binary:logistic",
#     "silent": 1,
#     "base_score": np.mean(y_train),
#     "eval_metric": "logloss"
# }
# model = xgb.train(params, X_train, 300, [(X_train, "train"),(y_train, "valid")], early_stopping_rounds=5, verbose_eval=25)
# =============================================================================


# =============================================================================
# y_pred = model.predict(D_test)
# 
# from sklearn.metrics import explained_variance_score
# from sklearn.metrics import r2_score
# 
# print(explained_variance_score(y_test,y_pred))
# print(r2_score(y_test,y_pred))
# 
# 
# y_test_size = y_test.size
# y_pred = np.reshape(y_pred,(y_test_size,1))
# y_pred_test = np.subtract(y_test,y_pred)
# 
# plt.figure()
# plt.plot(y_test,'-', color='C0', linewidth = 1 ,label="test")
# plt.plot((y_pred),'-', color='C1', linewidth = 1 , label="predicted")
# plt.legend()
# plt.xlabel('Index (shuffled)')
# plt.ylabel('RV')
# plt.title("XGBoost simpel model - training sensor data on RV drift")
# plt.show()
# 
# 
# plt.figure()
# plt.plot((y_pred_test), label="true - predicted")
# #plt.plot(y_pred,label="predicted")
# plt.legend()
# plt.xlabel('Index (shuffled)')
# plt.ylabel('RV')
# plt.title("XGBoost simpel model - training sensor data on RV drift")
# plt.show()
# 
# from numpy import mean, sqrt, square, arange
# rms = sqrt(mean(square(y_pred_test)))
# =============================================================================



## For at plotte unshuffled, importeres det samme datasæt som der blev testet på i første omgang, og dette er så unshuffled da test_split funktionen ikke er brugt.

df2 = pd.read_csv('D:\Sapientia\Dropbox\Fysik på KU\Big Data Analysis\Final Project\data\etalon_jitter_16Mar20_etalon.ccfSum-telemetryCLEANED.csv',names=['JD_UTC','RV','RV_ERR','FIES_ROOM_WEBSENSOR1_TEMPERATURE_NEAR_CEILING','FIES_ROOM_WEBSENSOR1_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_TEMPERATURE','FIES_FRONT_ROOM_WEBSENSOR2_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_ILLUMINATION','FIES_INSIDE_BLACK_BOX_AT_REAR','FIES_ROOM_NEAR_CEILING_AND_FAN_CONTROL','FIES_UNDER_GRATING_IN_TANK','FIES_FRONT_ROOM','FIES_BELOW_TABLE','FIES_ABOVE_HEATER_RADIATOR','FIES_INSIDE_BLACK_BOX_CENTRE','FIES_ABOVE_CEILING','FIES_GRATING_TANK_HOUSING','FIES_INSIDE_WHITE_BOX_REAR','FIES_INSIDE_WHITE_BOX_CENTRE','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_INCLUDE_GND','FIES_CEILING_NEAR_INSTRUMENT_AARHUS_1','FIES_INSIDE_BLACK_BOX_AARHUS_2','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3','FIES_DETECTOR_TEMPERATURE','FIES_DEWAR_TEMPERATURE','FIES_DETECTOR_PRESSURE','FIES_ATMOSPHERIC_PRESSURE','FIES_GRATING_TANK_PRESSURE'],skiprows=1)
#df = pd.read_csv('D:\Sapientia\Dropbox\Fysik på KU\Big Data Analysis\Final Project\data\AllData.txt',names=['JD_UTC','RV','RV_ERR','FIES_ROOM_WEBSENSOR1_TEMPERATURE_NEAR_CEILING','FIES_ROOM_WEBSENSOR1_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_TEMPERATURE','FIES_FRONT_ROOM_WEBSENSOR2_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_ILLUMINATION','FIES_INSIDE_BLACK_BOX_AT_REAR','FIES_ROOM_NEAR_CEILING_AND_FAN_CONTROL','FIES_UNDER_GRATING_IN_TANK','FIES_FRONT_ROOM','FIES_BELOW_TABLE','FIES_ABOVE_HEATER_RADIATOR','FIES_INSIDE_BLACK_BOX_CENTRE','FIES_ABOVE_CEILING','FIES_GRATING_TANK_HOUSING','FIES_INSIDE_WHITE_BOX_REAR','FIES_INSIDE_WHITE_BOX_CENTRE','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_INCLUDE_GND','FIES_CEILING_NEAR_INSTRUMENT_AARHUS_1','FIES_INSIDE_BLACK_BOX_AARHUS_2','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3','FIES_DETECTOR_TEMPERATURE','FIES_DEWAR_TEMPERATURE','FIES_DETECTOR_PRESSURE','FIES_ATMOSPHERIC_PRESSURE','FIES_GRATING_TANK_PRESSURE'],delim_whitespace=True,skiprows=0)

# Separating out the features
features3 = ['JD_UTC','FIES_ROOM_WEBSENSOR1_TEMPERATURE_NEAR_CEILING','FIES_ROOM_WEBSENSOR1_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_TEMPERATURE','FIES_FRONT_ROOM_WEBSENSOR2_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_ILLUMINATION','FIES_INSIDE_BLACK_BOX_AT_REAR','FIES_ROOM_NEAR_CEILING_AND_FAN_CONTROL','FIES_UNDER_GRATING_IN_TANK','FIES_FRONT_ROOM','FIES_BELOW_TABLE','FIES_ABOVE_HEATER_RADIATOR','FIES_INSIDE_BLACK_BOX_CENTRE','FIES_ABOVE_CEILING','FIES_GRATING_TANK_HOUSING','FIES_INSIDE_WHITE_BOX_REAR','FIES_INSIDE_WHITE_BOX_CENTRE','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_INCLUDE_GND','FIES_CEILING_NEAR_INSTRUMENT_AARHUS_1','FIES_INSIDE_BLACK_BOX_AARHUS_2','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3','FIES_DETECTOR_TEMPERATURE','FIES_DEWAR_TEMPERATURE','FIES_DETECTOR_PRESSURE','FIES_ATMOSPHERIC_PRESSURE','FIES_GRATING_TANK_PRESSURE']
#features3 = ['FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3','FIES_DETECTOR_TEMPERATURE','FIES_DEWAR_TEMPERATURE','FIES_DETECTOR_PRESSURE','FIES_ATMOSPHERIC_PRESSURE','FIES_GRATING_TANK_PRESSURE']
X3 = df2.loc[:, features3].values
# Separating out the target
y3 = df2.loc[:,['RV']].values
y3=y3[:,0]
# =============================================================================
# X3 = df[features3]
# y3 = df['RV']
# =============================================================================


X3 = scaler.transform(X3)


D_test3 = xgb.DMatrix(X3, label=y3)


y_pred3 = model.predict(D_test3)

from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score

print(explained_variance_score(y3,y_pred3))
print(r2_score(y3,y_pred3))

y3_test_size = y3.size
y_pred3 = np.reshape(y_pred3,(y3_test_size,1))
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
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X3)

# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
#shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:],matplotlib=True)


# visualize the training set predictions
#shap.force_plot(explainer.expected_value, shap_values, X,matplotlib=True)


# create a dependence plot to show the effect of a single feature across the whole dataset
#shap.dependence_plot("p_weta2", shap_values, X)

# summarize the effects of all the features

plt.figure()
shap.summary_plot(shap_values, X3)
plt.show()

plt.figure()
shap.summary_plot(shap_values, X3, plot_type="bar")
plt.show()

# Hvordan de forskellige features interacter
#shap.interaction = shap.TreeExplainer(model).shap_interaction_values(X)

# Hvis man vil øge antal features displayet
#shap.summaryplot(shapvalues, Ximportance , classnames=classes ,max_display=40)

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