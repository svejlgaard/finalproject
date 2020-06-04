
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
df = pd.read_csv('D:\Sapientia\Dropbox\Fysik på KU\Big Data Analysis\Final Project\data\Alldata1234scaled.txt',names=['JD_UTC','RV','RV_ERR','FIES_ROOM_WEBSENSOR1_TEMPERATURE_NEAR_CEILING','FIES_ROOM_WEBSENSOR1_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_TEMPERATURE','FIES_FRONT_ROOM_WEBSENSOR2_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_ILLUMINATION','FIES_INSIDE_BLACK_BOX_AT_REAR','FIES_ROOM_NEAR_CEILING_AND_FAN_CONTROL','FIES_UNDER_GRATING_IN_TANK','FIES_FRONT_ROOM','FIES_BELOW_TABLE','FIES_ABOVE_HEATER_RADIATOR','FIES_INSIDE_BLACK_BOX_CENTRE','FIES_ABOVE_CEILING','FIES_GRATING_TANK_HOUSING','FIES_INSIDE_WHITE_BOX_REAR','FIES_INSIDE_WHITE_BOX_CENTRE','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_INCLUDE_GND','FIES_CEILING_NEAR_INSTRUMENT_AARHUS_1','FIES_INSIDE_BLACK_BOX_AARHUS_2','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3','FIES_DETECTOR_TEMPERATURE','FIES_DEWAR_TEMPERATURE','FIES_DETECTOR_PRESSURE','FIES_ATMOSPHERIC_PRESSURE','FIES_GRATING_TANK_PRESSURE'],delim_whitespace=True,skiprows=0)

# Separating out the features
features = ['FIES_ROOM_WEBSENSOR1_TEMPERATURE_NEAR_CEILING','FIES_ROOM_WEBSENSOR1_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_TEMPERATURE','FIES_FRONT_ROOM_WEBSENSOR2_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_ILLUMINATION','FIES_INSIDE_BLACK_BOX_AT_REAR','FIES_ROOM_NEAR_CEILING_AND_FAN_CONTROL','FIES_UNDER_GRATING_IN_TANK','FIES_FRONT_ROOM','FIES_BELOW_TABLE','FIES_ABOVE_HEATER_RADIATOR','FIES_INSIDE_BLACK_BOX_CENTRE','FIES_ABOVE_CEILING','FIES_GRATING_TANK_HOUSING','FIES_INSIDE_WHITE_BOX_REAR','FIES_INSIDE_WHITE_BOX_CENTRE','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_INCLUDE_GND','FIES_CEILING_NEAR_INSTRUMENT_AARHUS_1','FIES_INSIDE_BLACK_BOX_AARHUS_2','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3','FIES_DETECTOR_TEMPERATURE','FIES_DEWAR_TEMPERATURE','FIES_DETECTOR_PRESSURE','FIES_ATMOSPHERIC_PRESSURE','FIES_GRATING_TANK_PRESSURE']
#features = ['FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3','FIES_DETECTOR_TEMPERATURE','FIES_DEWAR_TEMPERATURE','FIES_DETECTOR_PRESSURE','FIES_ATMOSPHERIC_PRESSURE','FIES_GRATING_TANK_PRESSURE']
X = df.loc[:, features].values

# Loading the scaled data
#X = np.loadtxt('D:\Sapientia\Dropbox\Fysik på KU\Big Data Analysis\Final Project\data\X1234scaled.txt')

#
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
# =============================================================================
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import minmax_scale
# from sklearn.preprocessing import MaxAbsScaler
# from sklearn.preprocessing import RobustScaler
# from sklearn.preprocessing import Normalizer
# from sklearn.preprocessing import QuantileTransformer
# from sklearn.preprocessing import PowerTransformer
# =============================================================================

scaler = StandardScaler()
#scaler = MinMaxScaler()
#scaler = MaxAbsScaler()
#scaler = MaxAbsScaler()


# =============================================================================
# # Denne boks er lavet til at skalerer hvert enkelt datasæt til sig selv
# scaler.fit(df)
# 
# df = scaler.transform(df)
# 
# np.savetxt(r'D:\Sapientia\Dropbox\Fysik på KU\Big Data Analysis\Final Project\data\Data5scaled.txt', df)
# #df.to_csv(r'D:\Sapientia\Dropbox\Fysik på KU\Big Data Analysis\Final Project\data\Data1scaled.txt', header=None, index=None, sep=' ', mode='a')
# #np.savetxt(r'D:\Sapientia\Dropbox\Fysik på KU\Big Data Analysis\Final Project\data\Data1scaled.txt', df.values, fmt='%d')
# =============================================================================



# Fit only to the training data
#scaler.fit(X_train)





# Now apply the transformations to the data:
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)

import shap
#import matplotlib.pyplot as plt

# load JS visualization code to notebook
#shap.initjs()


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


from sklearn.neural_network import MLPRegressor


# =============================================================================
# MLPRegressor()
# =============================================================================
# =============================================================================
# model = MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
#               beta_2=0.999, early_stopping=False, epsilon=1e-08,
#               hidden_layer_sizes=(25,10), learning_rate='adaptive',
#               learning_rate_init=0.001, max_fun=15000, max_iter=10000,
#               momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
#               power_t=0.5, random_state=0, shuffle=True, solver='sgd',
#               tol=0.0001, validation_fraction=0.1, verbose=True,
#               warm_start=False)
# =============================================================================
model = MLPRegressor(activation='identity', alpha=0.0001, batch_size='auto', beta_1=0.9,
             beta_2=0.999, early_stopping=True, epsilon=1e-08,
             hidden_layer_sizes=(3), learning_rate='adaptive',
             learning_rate_init=0.1, max_fun=15000, max_iter=2000,
             momentum=0.9, n_iter_no_change=100, nesterovs_momentum=True,
             power_t=0.5, random_state=0, shuffle=True, solver='adam',
             tol=0.0001, validation_fraction=0.1, verbose=True,
             warm_start=False)


print(model.fit(X_train,y_train))



# =============================================================================
# import scipy.stats as st
# from sklearn.model_selection import RandomizedSearchCV
# 
# clf_NN = MLPRegressor(random_state=0, learning_rate = 'adaptive', solver='adam', early_stopping=True, max_iter=2000)
# 
# class hidden_layer:
#     def __init__(self, min_range, max_range, max_layers=2):
#         self.max_layers = max_layers
#         self.min_range = min_range
#         self.max_range = max_range
# 
#     def rvs(self,random_state):
#         self.layers = np.random.randint(1,self.max_layers+1)
#         sizes = []
#         for i in range(self.layers):
#             if i==0:
#                 sizes.append( np.random.randint(self.min_range, self.max_range) )
#             elif sizes[i-1]==1:
#                 sizes.append( 1 )
#             else:
#                 sizes.append( np.random.randint(self.min_range, sizes[i-1]) )
#         return tuple(sizes)
# 
# # Parameters to search
# parameters_RandomSearch = {
#     "activation": ["relu"],
#     "hidden_layer_sizes": hidden_layer(1, 200, max_layers=200),
#     "learning_rate_init": st.uniform()
# }
# 
# # Number of search rounds
# n_iter_search = 50
# 
# # Initialize
# RandomSearch = RandomizedSearchCV(clf_NN, 
#                                   param_distributions=parameters_RandomSearch, 
#                                   n_iter=n_iter_search, 
#                                   cv=5,
#                                   return_train_score=True,
#                                   random_state=0,
#                                   scoring = 'r2',
#                                   verbose = 1, 
#                                   n_jobs = -1)
# 
# # fit the random search instance
# RandomSearch.fit(X_train, y_train);
# 
# # Printing best parameters
# print("Random Search: \tBest parameters: ", RandomSearch.best_params_, f", Best scores: {RandomSearch.best_score_:.4f}")
# =============================================================================

# =============================================================================
# import scipy.stats as st
# from sklearn.model_selection import RandomizedSearchCV
# 
# mlpsgd = MLPRegressor(max_iter = 1000, solver='sgd')
# 
# alpha = np.arange(0.01, 0.1, 0.01)
# hidden_layer_sizes = [(int(x),int(y),int(z)) for x in np.logspace(start = 0, stop = 2.2, num = 8) for y in np.logspace(start = 0, stop = 2.2, num = 8) for z in np.logspace(start = 0, stop = 2.2, num = 8)]
# hidden_layer_sizes.extend((int(x),int(y)) for x in np.logspace(start = 0, stop = 2, num = 25) for y in np.logspace(start = 0, stop = 2, num = 25))
# hidden_layer_sizes.extend((int(x),) for x in np.logspace(start = 1, stop = 2, num = 1000))    
# activation = ['logistic', 'tanh', 'relu']
# learning_rate = ['constant', 'invscaling','adaptive']
# learning_rate_init = np.arange(0.01, 0.1, 0.01)
# 
# random_grid3 = {'learning_rate': learning_rate,'activation': activation,'learning_rate_init': learning_rate_init,  'hidden_layer_sizes': hidden_layer_sizes, 'alpha': alpha}
# mlp_random3 = RandomizedSearchCV(estimator = mlpsgd, param_distributions = random_grid3, n_iter = 350, n_jobs=-1)
# 
# mlp_random3.fit(X_train, y_train)
# 
# print(mlp_random3.best_estimator_)
# 
# =============================================================================


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

#df2 = pd.read_csv('D:\Sapientia\Dropbox\Fysik på KU\Big Data Analysis\Final Project\data\etalon_jitter_16Mar20_etalon.ccfSum-telemetryCLEANED.csv',names=['JD_UTC','RV','RV_ERR','FIES_ROOM_WEBSENSOR1_TEMPERATURE_NEAR_CEILING','FIES_ROOM_WEBSENSOR1_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_TEMPERATURE','FIES_FRONT_ROOM_WEBSENSOR2_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_ILLUMINATION','FIES_INSIDE_BLACK_BOX_AT_REAR','FIES_ROOM_NEAR_CEILING_AND_FAN_CONTROL','FIES_UNDER_GRATING_IN_TANK','FIES_FRONT_ROOM','FIES_BELOW_TABLE','FIES_ABOVE_HEATER_RADIATOR','FIES_INSIDE_BLACK_BOX_CENTRE','FIES_ABOVE_CEILING','FIES_GRATING_TANK_HOUSING','FIES_INSIDE_WHITE_BOX_REAR','FIES_INSIDE_WHITE_BOX_CENTRE','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_INCLUDE_GND','FIES_CEILING_NEAR_INSTRUMENT_AARHUS_1','FIES_INSIDE_BLACK_BOX_AARHUS_2','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3','FIES_DETECTOR_TEMPERATURE','FIES_DEWAR_TEMPERATURE','FIES_DETECTOR_PRESSURE','FIES_ATMOSPHERIC_PRESSURE','FIES_GRATING_TANK_PRESSURE'],skiprows=1)
df2 = pd.read_csv('D:\Sapientia\Dropbox\Fysik på KU\Big Data Analysis\Final Project\data\Data5scaled.txt',names=['JD_UTC','RV','RV_ERR','FIES_ROOM_WEBSENSOR1_TEMPERATURE_NEAR_CEILING','FIES_ROOM_WEBSENSOR1_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_TEMPERATURE','FIES_FRONT_ROOM_WEBSENSOR2_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_ILLUMINATION','FIES_INSIDE_BLACK_BOX_AT_REAR','FIES_ROOM_NEAR_CEILING_AND_FAN_CONTROL','FIES_UNDER_GRATING_IN_TANK','FIES_FRONT_ROOM','FIES_BELOW_TABLE','FIES_ABOVE_HEATER_RADIATOR','FIES_INSIDE_BLACK_BOX_CENTRE','FIES_ABOVE_CEILING','FIES_GRATING_TANK_HOUSING','FIES_INSIDE_WHITE_BOX_REAR','FIES_INSIDE_WHITE_BOX_CENTRE','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_INCLUDE_GND','FIES_CEILING_NEAR_INSTRUMENT_AARHUS_1','FIES_INSIDE_BLACK_BOX_AARHUS_2','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3','FIES_DETECTOR_TEMPERATURE','FIES_DEWAR_TEMPERATURE','FIES_DETECTOR_PRESSURE','FIES_ATMOSPHERIC_PRESSURE','FIES_GRATING_TANK_PRESSURE'],delim_whitespace=True,skiprows=0)

# Separating out the features
features3 = ['FIES_ROOM_WEBSENSOR1_TEMPERATURE_NEAR_CEILING','FIES_ROOM_WEBSENSOR1_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_TEMPERATURE','FIES_FRONT_ROOM_WEBSENSOR2_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_ILLUMINATION','FIES_INSIDE_BLACK_BOX_AT_REAR','FIES_ROOM_NEAR_CEILING_AND_FAN_CONTROL','FIES_UNDER_GRATING_IN_TANK','FIES_FRONT_ROOM','FIES_BELOW_TABLE','FIES_ABOVE_HEATER_RADIATOR','FIES_INSIDE_BLACK_BOX_CENTRE','FIES_ABOVE_CEILING','FIES_GRATING_TANK_HOUSING','FIES_INSIDE_WHITE_BOX_REAR','FIES_INSIDE_WHITE_BOX_CENTRE','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_INCLUDE_GND','FIES_CEILING_NEAR_INSTRUMENT_AARHUS_1','FIES_INSIDE_BLACK_BOX_AARHUS_2','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3','FIES_DETECTOR_TEMPERATURE','FIES_DEWAR_TEMPERATURE','FIES_DETECTOR_PRESSURE','FIES_ATMOSPHERIC_PRESSURE','FIES_GRATING_TANK_PRESSURE']
#features3 = ['FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3','FIES_DETECTOR_TEMPERATURE','FIES_DEWAR_TEMPERATURE','FIES_DETECTOR_PRESSURE','FIES_ATMOSPHERIC_PRESSURE','FIES_GRATING_TANK_PRESSURE']
X3 = df2.loc[:, features3].values
# Loading the scaled data
#X3 = np.loadtxt('D:\Sapientia\Dropbox\Fysik på KU\Big Data Analysis\Final Project\data\X5scaled.txt')
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
# =============================================================================
# 
# # explain the model's predictions using SHAP
# # (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X3)
# 
# # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
# #shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:],matplotlib=True)
# 
# 
# # visualize the training set predictions
# #shap.force_plot(explainer.expected_value, shap_values, X,matplotlib=True)
# 
# 
# # create a dependence plot to show the effect of a single feature across the whole dataset
# #shap.dependence_plot("p_weta2", shap_values, X)
# 
# # summarize the effects of all the features
# 
# plt.figure()
# shap.summary_plot(shap_values, X3)
# plt.show()
# 
# plt.figure()
# shap.summary_plot(shap_values, X3, plot_type="bar")
# plt.show()
# 
# # Hvordan de forskellige features interacter
# #shap.interaction = shap.TreeExplainer(model).shap_interaction_values(X)
# 
# # Hvis man vil øge antal features displayet
# #shap.summaryplot(shapvalues, Ximportance , classnames=classes ,max_display=40)
# 
# #FIES_INSIDE_BLACK_BOX_CENTRE
# #FIES_INSIDE_BLACK_BOX_AT_REAR
# #FIES_ROOM_WEBSENSOR1_HUMIDITY
# =============================================================================
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