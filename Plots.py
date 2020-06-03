import datetime

begin_time = datetime.datetime.now()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load dataset into Pandas DataFrame - Use your own filepath to the data
#df = pd.read_csv('D:\Sapientia\Dropbox\Fysik på KU\Big Data Analysis\Final Project\etalon_jitter_16Mar20_etalon.ccfSum-telemetry.csv',names=['JD_UTC','RV','RV_ERR','FIES_ROOM_WEBSENSOR1_TEMPERATURE_NEAR_CEILING','FIES_ROOM_WEBSENSOR1_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_TEMPERATURE','FIES_FRONT_ROOM_WEBSENSOR2_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_ILLUMINATION','FIES_INSIDE_BLACK_BOX_AT_REAR','FIES_ROOM_NEAR_CEILING_AND_FAN_CONTROL','FIES_UNDER_GRATING_IN_TANK','FIES_FRONT_ROOM','FIES_BELOW_TABLE','FIES_ABOVE_HEATER_RADIATOR','FIES_INSIDE_BLACK_BOX_CENTRE','FIES_ABOVE_CEILING','FIES_GRATING_TANK_HOUSING','FIES_INSIDE_WHITE_BOX_REAR','FIES_INSIDE_WHITE_BOX_CENTRE','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_INCLUDE_GND','FIES_CEILING_NEAR_INSTRUMENT_AARHUS_1','FIES_INSIDE_BLACK_BOX_AARHUS_2','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3','FIES_DETECTOR_TEMPERATURE','FIES_DEWAR_TEMPERATURE','FIES_DETECTOR_PRESSURE','FIES_ATMOSPHERIC_PRESSURE','FIES_GRATING_TANK_PRESSURE'],skiprows=1)
df = pd.read_csv('D:\Sapientia\Dropbox\Fysik på KU\Big Data Analysis\Final Project\data\Data2.txt',names=['JD_UTC','RV','RV_ERR','FIES_ROOM_WEBSENSOR1_TEMPERATURE_NEAR_CEILING','FIES_ROOM_WEBSENSOR1_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_TEMPERATURE','FIES_FRONT_ROOM_WEBSENSOR2_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_ILLUMINATION','FIES_INSIDE_BLACK_BOX_AT_REAR','FIES_ROOM_NEAR_CEILING_AND_FAN_CONTROL','FIES_UNDER_GRATING_IN_TANK','FIES_FRONT_ROOM','FIES_BELOW_TABLE','FIES_ABOVE_HEATER_RADIATOR','FIES_INSIDE_BLACK_BOX_CENTRE','FIES_ABOVE_CEILING','FIES_GRATING_TANK_HOUSING','FIES_INSIDE_WHITE_BOX_REAR','FIES_INSIDE_WHITE_BOX_CENTRE','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_INCLUDE_GND','FIES_CEILING_NEAR_INSTRUMENT_AARHUS_1','FIES_INSIDE_BLACK_BOX_AARHUS_2','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3','FIES_DETECTOR_TEMPERATURE','FIES_DEWAR_TEMPERATURE','FIES_DETECTOR_PRESSURE','FIES_ATMOSPHERIC_PRESSURE','FIES_GRATING_TANK_PRESSURE'],delim_whitespace=True,skiprows=0)
#df = pd.read_csv('D:\Sapientia\Dropbox\Fysik på KU\Big Data Analysis\Final Project\data\AllDatav3.txt',names=['JD_UTC','FIES_ROOM_WEBSENSOR1_TEMPERATURE_NEAR_CEILING','FIES_ROOM_WEBSENSOR1_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_TEMPERATURE','FIES_FRONT_ROOM_WEBSENSOR2_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_ILLUMINATION','FIES_INSIDE_BLACK_BOX_AT_REAR','FIES_ROOM_NEAR_CEILING_AND_FAN_CONTROL','FIES_UNDER_GRATING_IN_TANK','FIES_FRONT_ROOM','FIES_BELOW_TABLE','FIES_ABOVE_HEATER_RADIATOR','FIES_INSIDE_BLACK_BOX_CENTRE','FIES_ABOVE_CEILING','FIES_GRATING_TANK_HOUSING','FIES_INSIDE_WHITE_BOX_REAR','FIES_INSIDE_WHITE_BOX_CENTRE','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_INCLUDE_GND','FIES_CEILING_NEAR_INSTRUMENT_AARHUS_1','FIES_INSIDE_BLACK_BOX_AARHUS_2','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3','FIES_DETECTOR_TEMPERATURE','FIES_DEWAR_TEMPERATURE','FIES_DETECTOR_PRESSURE','FIES_ATMOSPHERIC_PRESSURE','FIES_GRATING_TANK_PRESSURE'],delim_whitespace=True,skiprows=0)

# Separating out the features
features = ['FIES_ROOM_WEBSENSOR1_TEMPERATURE_NEAR_CEILING','FIES_ROOM_WEBSENSOR1_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_TEMPERATURE','FIES_FRONT_ROOM_WEBSENSOR2_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_ILLUMINATION','FIES_INSIDE_BLACK_BOX_AT_REAR','FIES_ROOM_NEAR_CEILING_AND_FAN_CONTROL','FIES_UNDER_GRATING_IN_TANK','FIES_FRONT_ROOM','FIES_BELOW_TABLE','FIES_ABOVE_HEATER_RADIATOR','FIES_INSIDE_BLACK_BOX_CENTRE','FIES_ABOVE_CEILING','FIES_GRATING_TANK_HOUSING','FIES_INSIDE_WHITE_BOX_REAR','FIES_INSIDE_WHITE_BOX_CENTRE','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_INCLUDE_GND','FIES_CEILING_NEAR_INSTRUMENT_AARHUS_1','FIES_INSIDE_BLACK_BOX_AARHUS_2','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3','FIES_DETECTOR_TEMPERATURE','FIES_DEWAR_TEMPERATURE','FIES_DETECTOR_PRESSURE','FIES_ATMOSPHERIC_PRESSURE','FIES_GRATING_TANK_PRESSURE']
#features = ['FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3','FIES_DETECTOR_TEMPERATURE','FIES_DEWAR_TEMPERATURE','FIES_DETECTOR_PRESSURE','FIES_ATMOSPHERIC_PRESSURE','FIES_GRATING_TANK_PRESSURE']
X = df.loc[:, features].values
# Separating out the target
# =============================================================================
# y = df.loc[:,['RV']].values
# y=y[:,0]
# =============================================================================

# gca stands for 'get current axis'
ax = plt.gca()
ax.tick_params(axis='y', labelcolor='red')
ax2 = ax.twinx()
ax2.tick_params(axis='y', labelcolor='blue')
ax3 = ax.twinx()
ax3.tick_params(axis='y', labelcolor='gray')
ax4 = ax.twinx()
ax4.tick_params(axis='y', labelcolor='gold')
ax5 = ax.twinx()
ax5.tick_params(axis='y', labelcolor='green')
ax6 = ax.twinx()
ax6.tick_params(axis='y', labelcolor='white')

# =============================================================================

#,y=df.columns[4]
df.plot(kind='line', x = 'JD_UTC',y='RV', color='black',ax=ax6, legend=True)

#df.plot(kind='line', x = 'JD_UTC',y='FIES_ROOM_WEBSENSOR1_TEMPERATURE_NEAR_CEILING', color='red',ax=ax, legend=True)

#df.plot(kind='line', x = 'JD_UTC',y='FIES_ROOM_WEBSENSOR1_HUMIDITY', color='deepskyblue', ax=ax2, legend=True)

#df.plot(kind='line', x = 'JD_UTC',y='FIES_FRONT_ROOM_WEBSENSOR2_TEMPERATURE', color='darkred', ax=ax, legend=True)

#df.plot(kind='line', x = 'JD_UTC',y='FIES_FRONT_ROOM_WEBSENSOR2_HUMIDITY', color='dodgerblue', ax=ax2, legend=True)

df.plot(kind='line', x = 'JD_UTC',y='FIES_FRONT_ROOM_WEBSENSOR2_ILLUMINATION', color='gold', ax=ax4, legend=True)

df.plot(kind='line', x = 'JD_UTC',y='FIES_INSIDE_BLACK_BOX_AT_REAR', color='maroon', ax=ax, legend=True)
df.plot(kind='line', x = 'JD_UTC',y='FIES_ROOM_NEAR_CEILING_AND_FAN_CONTROL', color='firebrick', ax=ax, legend=True)
df.plot(kind='line', x = 'JD_UTC',y='FIES_UNDER_GRATING_IN_TANK', color='indianred', ax=ax, legend=True)
df.plot(kind='line', x = 'JD_UTC',y='FIES_FRONT_ROOM', color='tomato', ax=ax, legend=True)
df.plot(kind='line', x = 'JD_UTC',y='FIES_BELOW_TABLE', color='orangered', ax=ax, legend=True)
df.plot(kind='line', x = 'JD_UTC',y='FIES_ABOVE_HEATER_RADIATOR', color='Blue', ax=ax, legend=True)
df.plot(kind='line', x = 'JD_UTC',y='FIES_INSIDE_BLACK_BOX_CENTRE', color='chocolate', ax=ax, legend=True)
df.plot(kind='line', x = 'JD_UTC',y='FIES_ABOVE_CEILING', color='sienna', ax=ax, legend=True)
df.plot(kind='line', x = 'JD_UTC',y='FIES_GRATING_TANK_HOUSING', color='crimson', ax=ax, legend=True)
df.plot(kind='line', x = 'JD_UTC',y='FIES_INSIDE_WHITE_BOX_REAR', color='hotpink', ax=ax, legend=True)
df.plot(kind='line', x = 'JD_UTC',y='FIES_INSIDE_WHITE_BOX_CENTRE', color='purple', ax=ax, legend=True)
df.plot(kind='line', x = 'JD_UTC',y='FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_INCLUDE_GND', color='indigo', ax=ax, legend=True)
#df.plot(kind='line', x = 'JD_UTC',y='FIES_CEILING_NEAR_INSTRUMENT_AARHUS_1', color='magenta', ax=ax, legend=True)
#df.plot(kind='line', x = 'JD_UTC',y='FIES_INSIDE_BLACK_BOX_AARHUS_2', color='orange', ax=ax, legend=True)
#df.plot(kind='line', x = 'JD_UTC',y='FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3', color='darkorange', ax=ax, legend=True)

df.plot(kind='line', x = 'JD_UTC',y='FIES_DETECTOR_TEMPERATURE', color='darkgreen', ax=ax5, legend=True)
df.plot(kind='line', x = 'JD_UTC',y='FIES_DEWAR_TEMPERATURE', color='lime', ax=ax5, legend=True)

df.plot(kind='line', x = 'JD_UTC',y='FIES_DETECTOR_PRESSURE', color='gray', ax=ax3, legend=True)
df.plot(kind='line', x = 'JD_UTC',y='FIES_ATMOSPHERIC_PRESSURE', color='darkgray', ax=ax3, legend=True)
df.plot(kind='line', x = 'JD_UTC',y='FIES_GRATING_TANK_PRESSURE', color='lightgray', ax=ax3, legend=True)

plt.show()

df.plot(x='JD_UTC',y='RV')
plt.show()

# =============================================================================
# # Dette laver hele dataframen på et plot, dog ikke brugbart da vi har forskellige skalaer
# plt.figure();
# 
# df.plot(x='JD_UTC');
# =============================================================================
