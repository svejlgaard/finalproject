import datetime

begin_time = datetime.datetime.now()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load dataset into Pandas DataFrame - Use your own filepath to the data
#df = pd.read_csv('D:\Sapientia\Dropbox\Fysik på KU\Big Data Analysis\Final Project\etalon_jitter_16Mar20_etalon.ccfSum-telemetry.csv',names=['JD_UTC','RV','RV_ERR','FIES_ROOM_WEBSENSOR1_TEMPERATURE_NEAR_CEILING','FIES_ROOM_WEBSENSOR1_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_TEMPERATURE','FIES_FRONT_ROOM_WEBSENSOR2_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_ILLUMINATION','FIES_INSIDE_BLACK_BOX_AT_REAR','FIES_ROOM_NEAR_CEILING_AND_FAN_CONTROL','FIES_UNDER_GRATING_IN_TANK','FIES_FRONT_ROOM','FIES_BELOW_TABLE','FIES_ABOVE_HEATER_RADIATOR','FIES_INSIDE_BLACK_BOX_CENTRE','FIES_ABOVE_CEILING','FIES_GRATING_TANK_HOUSING','FIES_INSIDE_WHITE_BOX_REAR','FIES_INSIDE_WHITE_BOX_CENTRE','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_INCLUDE_GND','FIES_CEILING_NEAR_INSTRUMENT_AARHUS_1','FIES_INSIDE_BLACK_BOX_AARHUS_2','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3','FIES_DETECTOR_TEMPERATURE','FIES_DEWAR_TEMPERATURE','FIES_DETECTOR_PRESSURE','FIES_ATMOSPHERIC_PRESSURE','FIES_GRATING_TANK_PRESSURE'],skiprows=1)

df = pd.read_csv('D:\Sapientia\Dropbox\Fysik på KU\Big Data Analysis\Final Project\data\Data2scaled.txt',names=['JD_UTC','RV','RV_ERR','FIES_ROOM_WEBSENSOR1_TEMPERATURE_NEAR_CEILING','FIES_ROOM_WEBSENSOR1_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_TEMPERATURE','FIES_FRONT_ROOM_WEBSENSOR2_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_ILLUMINATION','FIES_INSIDE_BLACK_BOX_AT_REAR','FIES_ROOM_NEAR_CEILING_AND_FAN_CONTROL','FIES_UNDER_GRATING_IN_TANK','FIES_FRONT_ROOM','FIES_BELOW_TABLE','FIES_ABOVE_HEATER_RADIATOR','FIES_INSIDE_BLACK_BOX_CENTRE','FIES_ABOVE_CEILING','FIES_GRATING_TANK_HOUSING','FIES_INSIDE_WHITE_BOX_REAR','FIES_INSIDE_WHITE_BOX_CENTRE','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_INCLUDE_GND','FIES_CEILING_NEAR_INSTRUMENT_AARHUS_1','FIES_INSIDE_BLACK_BOX_AARHUS_2','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3','FIES_DETECTOR_TEMPERATURE','FIES_DEWAR_TEMPERATURE','FIES_DETECTOR_PRESSURE','FIES_ATMOSPHERIC_PRESSURE','FIES_GRATING_TANK_PRESSURE'],delim_whitespace=True,skiprows=0)


# Separating out the features
features = ['FIES_ROOM_WEBSENSOR1_TEMPERATURE_NEAR_CEILING','FIES_ROOM_WEBSENSOR1_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_TEMPERATURE','FIES_FRONT_ROOM_WEBSENSOR2_HUMIDITY','FIES_FRONT_ROOM_WEBSENSOR2_ILLUMINATION','FIES_INSIDE_BLACK_BOX_AT_REAR','FIES_ROOM_NEAR_CEILING_AND_FAN_CONTROL','FIES_UNDER_GRATING_IN_TANK','FIES_FRONT_ROOM','FIES_BELOW_TABLE','FIES_ABOVE_HEATER_RADIATOR','FIES_INSIDE_BLACK_BOX_CENTRE','FIES_ABOVE_CEILING','FIES_GRATING_TANK_HOUSING','FIES_INSIDE_WHITE_BOX_REAR','FIES_INSIDE_WHITE_BOX_CENTRE','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_INCLUDE_GND','FIES_CEILING_NEAR_INSTRUMENT_AARHUS_1','FIES_INSIDE_BLACK_BOX_AARHUS_2','FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3','FIES_DETECTOR_TEMPERATURE','FIES_DEWAR_TEMPERATURE','FIES_DETECTOR_PRESSURE','FIES_ATMOSPHERIC_PRESSURE','FIES_GRATING_TANK_PRESSURE']


#features = ['FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3','FIES_DETECTOR_TEMPERATURE','FIES_DEWAR_TEMPERATURE','FIES_DETECTOR_PRESSURE','FIES_ATMOSPHERIC_PRESSURE','FIES_GRATING_TANK_PRESSURE']
X = df.loc[:, features].values
# Separating out the target
# =============================================================================
# y = df.loc[:,['RV']].values
# y=y[:,0]
# =============================================================================

plt.figure()
# gca stands for 'get current axis'
ax = plt.gca()
ax.tick_params(axis='y', labelcolor='red')
# Data 3 event

# =============================================================================
# plt.vlines(-1.205, ymin=-9.434258331744381, ymax=3.405960272063083, color="cyan", linestyle="--")
# plt.vlines(-0.497, ymin=-9.434258331744381, ymax=3.405960272063083, color="cyan", linestyle="--")
# =============================================================================
# =============================================================================
# # Data 4 event
# plt.vlines(-0.5625, ymin=-2.7096985469055133, ymax=1.7101372156051178, color="orange", linestyle="--")
# plt.vlines(0.266, ymin=-2.7096985469055133, ymax=1.7101372156051178, color="orange", linestyle="--")
# =============================================================================

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
plt.style.use('ggplot'); plt.style.use('seaborn-ticks')
#,y=df.columns[4]


# 0.01
df.plot(kind='line', x = 'JD_UTC',y='RV', color='black',ax=ax6, legend=True)

# 0.1
#df.plot(kind='line', x = 'JD_UTC',y='FIES_ROOM_WEBSENSOR1_TEMPERATURE_NEAR_CEILING', color='firebrick',ax=ax, legend=True)

# 0.1
#df.plot(kind='line', x = 'JD_UTC',y='FIES_ROOM_WEBSENSOR1_HUMIDITY', color='deepskyblue', ax=ax2, legend=True)

# 0.1
#df.plot(kind='line', x = 'JD_UTC',y='FIES_FRONT_ROOM_WEBSENSOR2_TEMPERATURE', color='darkred', ax=ax, legend=True)

# 0.1
#df.plot(kind='line', x = 'JD_UTC',y='FIES_FRONT_ROOM_WEBSENSOR2_HUMIDITY', color='dodgerblue', ax=ax2, legend=True)

# 0.1
#df.plot(kind='line', x = 'JD_UTC',y='FIES_FRONT_ROOM_WEBSENSOR2_ILLUMINATION', color='gold', ax=ax4, legend=True)

# 0.001
df.plot(kind='line', x = 'JD_UTC',y='FIES_INSIDE_BLACK_BOX_AT_REAR' , color='maroon', ax=ax, legend=True)

# 0.001
#df.plot(kind='line', x = 'JD_UTC',y='FIES_ROOM_NEAR_CEILING_AND_FAN_CONTROL', color='firebrick', ax=ax, legend=True)

# 0.001
df.plot(kind='line', x = 'JD_UTC',y='FIES_UNDER_GRATING_IN_TANK', color='indianred', ax=ax, legend=True)

# 0.001
#df.plot(kind='line', x = 'JD_UTC',y='FIES_FRONT_ROOM', color='tomato', ax=ax, legend=True)

# 0.001
#df.plot(kind='line', x = 'JD_UTC',y='FIES_BELOW_TABLE', color='orangered', ax=ax, legend=True)

# 0.001
#df.plot(kind='line', x  = 'JD_UTC', y='FIES_ABOVE_HEATER_RADIATOR', color='red', ax=ax, legend=True)

# 0.001
df.plot(kind='line',  x = 'JD_UTC',y='FIES_INSIDE_BLACK_BOX_CENTRE', color='chocolate', ax=ax, legend=True)

# 0.001
#df.plot(kind='line', x = 'JD_UTC',y='FIES_ABOVE_CEILING', color='sienna', ax=ax, legend=True)

# 0.001
df.plot(kind='line', x = 'JD_UTC',y='FIES_GRATING_TANK_HOUSING', color='crimson', ax=ax, legend=True)

# 0.001
#df.plot(kind='line', x = 'JD_UTC',y='FIES_INSIDE_WHITE_BOX_REAR', color='hotpink', ax=ax, legend=True)

# 0.001
df.plot(kind='line', x = 'JD_UTC',y='FIES_INSIDE_WHITE_BOX_CENTRE', color='purple', ax=ax, legend=True)

# 0.001
#df.plot(kind='line', x = 'JD_UTC',y='FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_INCLUDE_GND', color='indigo', ax=ax, legend=True)

# 0.01
#df.plot(kind='line', x = 'JD_UTC',y='FIES_CEILING_NEAR_INSTRUMENT_AARHUS_1', color='magenta', ax=ax, legend=True)

# 0.01
#df.plot(kind='line', x = 'JD_UTC',y='FIES_INSIDE_BLACK_BOX_AARHUS_2', color='orange', ax=ax, legend=True)

# 0.01
#df.plot(kind='line', x = 'JD_UTC',y='FIES_INSIDE_WHITE_BOX_UNDER_DEWAR_AARHUS_3', color='darkorange', ax=ax, legend=True)

# 0.01
#df.plot(kind='line', x = 'JD_UTC',y='FIES_DETECTOR_TEMPERATURE', color='darkgreen', ax=ax5, legend=True)

# 0.01
#df.plot(kind='line', x = 'JD_UTC',y='FIES_DEWAR_TEMPERATURE', color='lime', ax=ax5, legend=True)

# Ukendt
#df.plot(kind='line', x = 'JD_UTC',y='FIES_DETECTOR_PRESSURE', color='gray', ax=ax3, legend=True)

# 0.001
df.plot(kind='line', x = 'JD_UTC',y='FIES_ATMOSPHERIC_PRESSURE', color='darkgray', ax=ax3, legend=True)

# 0.001
#df.plot(kind='line', x = 'JD_UTC',y='FIES_GRATING_TANK_PRESSURE', color='lightgray', ax=ax3, legend=True)
#patches, labels = ax.get_legend_handles_labels()
patches, labels = ax.get_legend_handles_labels()
ax.legend(patches, labels, loc='lower center')
#ax.legend(patches, labels, loc='lower center', prop={'size': 15})
patches, labels = ax3.get_legend_handles_labels()
ax3.legend(patches, labels, loc='upper center')
patches, labels = ax6.get_legend_handles_labels()
ax6.legend(patches, labels, loc='lower left')


# =============================================================================
# df.plot(x='JD_UTC',y='RV')
# plt.show()
# =============================================================================

# =============================================================================
# # Dette laver hele dataframen på et plot, dog ikke brugbart da vi har forskellige skalaer
# plt.figure();
# 
# df.plot(x='JD_UTC');
# =============================================================================
# =============================================================================
# 
# #extended data
# df = pd.read_csv('D:\Sapientia\Dropbox\Fysik på KU\Big Data Analysis\Final Project\data\Data34extended.txt',names=['JD_UTC','FIES room WebSensor1 Temperature near ceiling','FIES room WebSensor1 Humidity','FIES room WebSensor1 Illumination','FIES front room WebSensor2 Temperature','FIES front room WebSensor2 Humidity','FIES front room WebSensor2 Illumination','FIES inside black box at rear','FIES room near ceiling and fan control','FIES under grating in tank','FIES front room','FIES below table girder','FIES above heater radiator','FIES inside black box centre','FIES above ceiling','FIES grating tank housing','FIES inside white box rear','FIES inside white box centre','FIES centre of table [inc. gnd]','FIES ceiling near instrument (Aarhus 1)','FIES inside black box (Aarhus 2)','FIES Inside white box under dewar (Aarhus 3)','FIES Detector temperature','FIES Dewar temperature','FIES Detector pressure','FIES Atmospheric Pressure','FIES grating tank pressure'],delim_whitespace=True,skiprows=0)
# 
# # For extended data
# 
# # når vi ser på extended data:
# features = ['JD_UTC','FIES room WebSensor1 Temperature near ceiling','FIES room WebSensor1 Humidity','FIES room WebSensor1 Illumination','FIES front room WebSensor2 Temperature','FIES front room WebSensor2 Humidity','FIES front room WebSensor2 Illumination','FIES inside black box at rear','FIES room near ceiling and fan control','FIES under grating in tank','FIES front room','FIES below table girder','FIES above heater radiator','FIES inside black box centre','FIES above ceiling','FIES grating tank housing','FIES inside white box rear','FIES inside white box centre','FIES centre of table [inc. gnd]','FIES ceiling near instrument (Aarhus 1)','FIES inside black box (Aarhus 2)','FIES Inside white box under dewar (Aarhus 3)','FIES Detector temperature','FIES Dewar temperature','FIES Detector pressure','FIES Atmospheric Pressure','FIES grating tank pressure']
# X = df.loc[:, features].values
# 
# #plt.style.use('ggplot'); plt.style.use('seaborn-ticks')
# 
# fig = plt.figure(figsize=(19.20,10.80))
# 
# ax = plt.gca()
# ax.tick_params(axis='y', labelcolor='red')
# ax2 = ax.twinx()
# ax2.tick_params(axis='y', labelcolor='blue')
# ax3 = ax.twinx()
# ax3.tick_params(axis='y', labelcolor='gray')
# ax4 = ax.twinx()
# ax4.tick_params(axis='y', labelcolor='gold')
# ax5 = ax.twinx()
# ax5.tick_params(axis='y', labelcolor='green')
# ax6 = ax.twinx()
# ax6.tick_params(axis='y', labelcolor='white')
# 
# #,y=df.columns[4]
# 
# # 0.01
# #df.plot(kind='line',y='RV', color='black',ax=ax6, legend=True)
# 
# # 0.1
# df.plot(kind='line', x = 'JD_UTC',y='FIES room WebSensor1 Temperature near ceiling', color='firebrick',ax=ax, legend=True)
# 
# # 0.1
# df.plot(kind='line', x = 'JD_UTC',y='FIES room WebSensor1 Humidity', color='deepskyblue', ax=ax2, legend=True)
# 
# # 0.1
# df.plot(kind='line', x = 'JD_UTC',y='FIES room WebSensor1 Illumination', color='gold', ax=ax4, legend=True)
# 
# # 0.1
# df.plot(kind='line', x = 'JD_UTC',y='FIES front room WebSensor2 Temperature', color='darkred', ax=ax, legend=True)
# 
# # 0.1
# df.plot(kind='line', x = 'JD_UTC',y='FIES front room WebSensor2 Humidity', color='dodgerblue', ax=ax2, legend=True)
# 
# # 0.1
# df.plot(kind='line', x = 'JD_UTC',y='FIES front room WebSensor2 Illumination', color='gold', ax=ax4, legend=True)
# 
# # 0.001
# df.plot(kind='line', x = 'JD_UTC',y='FIES inside black box at rear' , color='maroon', ax=ax, legend=True)
# 
# # 0.001
# df.plot(kind='line', x = 'JD_UTC',y='FIES room near ceiling and fan control', color='firebrick', ax=ax, legend=True)
# 
# # 0.001
# df.plot(kind='line', x = 'JD_UTC',y='FIES under grating in tank', color='indianred', ax=ax, legend=True)
# 
# # 0.001
# df.plot(kind='line', x = 'JD_UTC',y='FIES front room', color='tomato', ax=ax, legend=True)
# 
# # 0.001
# df.plot(kind='line', x = 'JD_UTC',y='FIES below table girder', color='orangered', ax=ax, legend=True)
# 
# # 0.001
# df.plot(kind='line', x = 'JD_UTC',y='FIES above heater radiator', color='red', ax=ax, legend=True)
# 
# # 0.001
# df.plot(kind='line', x = 'JD_UTC',y='FIES inside black box centre', color='chocolate', ax=ax, legend=True)
# 
# # 0.001
# df.plot(kind='line', x = 'JD_UTC',y='FIES above ceiling', color='sienna', ax=ax, legend=True)
# 
# # 0.001
# df.plot(kind='line', x = 'JD_UTC',y='FIES grating tank housing', color='crimson', ax=ax, legend=True)
# 
# # 0.001
# df.plot(kind='line', x = 'JD_UTC',y='FIES inside white box rear', color='hotpink', ax=ax, legend=True)
# 
# # 0.001
# df.plot(kind='line', x = 'JD_UTC',y='FIES inside white box centre', color='purple', ax=ax, legend=True)
# 
# # 0.001
# df.plot(kind='line', x = 'JD_UTC',y='FIES centre of table [inc. gnd]', color='indigo', ax=ax, legend=True)
# 
# # 0.01
# df.plot(kind='line', x = 'JD_UTC',y='FIES ceiling near instrument (Aarhus 1)', color='magenta', ax=ax, legend=True)
# 
# # 0.01
# df.plot(kind='line', x = 'JD_UTC',y='FIES inside black box (Aarhus 2)', color='orange', ax=ax, legend=True)
# 
# # 0.01
# df.plot(kind='line', x = 'JD_UTC',y='FIES Inside white box under dewar (Aarhus 3)', color='darkorange', ax=ax, legend=True)
# 
# # 0.01
# df.plot(kind='line', x = 'JD_UTC',y='FIES Detector temperature', color='darkgreen', ax=ax5, legend=True)
# 
# # 0.01
# df.plot(kind='line', x = 'JD_UTC',y='FIES Dewar temperature', color='lime', ax=ax5, legend=True)
# 
# # Ukendt
# df.plot(kind='line', x = 'JD_UTC',y='FIES Detector pressure', color='gray', ax=ax3, legend=True)
# 
# # 0.001
# df.plot(kind='line', x = 'JD_UTC',y='FIES Atmospheric Pressure', color='darkgray', ax=ax3, legend=True)
# 
# # 0.001
# df.plot(kind='line', x = 'JD_UTC',y='FIES grating tank pressure', color='lightgray', ax=ax3, legend=True)
# 
# #ax.axvline(x="2.46893e+06", ax= ax2, color="red", linestyle="--")
# 
# # Data 1 grænser
# # =============================================================================
# # plt.vlines(2.458925023748000152e+06, ymin=0, ymax=1, color="black", linestyle="--")
# # plt.vlines(2.458925169067999814e+06, ymin=0, ymax=1, color="black", linestyle="--")
# # =============================================================================
# # =============================================================================
# # # Data 2 grænser
# # plt.vlines(2.458927168012999929e+06, ymin=0, ymax=1, color="black", linestyle="--")
# # plt.vlines(2.458927216284999624e+06, ymin=0, ymax=1, color="black", linestyle="--")
# # =============================================================================
# # =============================================================================
# # # Data 3 grænser
# # 
# # plt.vlines(2.458934948158999905e+06, ymin=0, ymax=1, color="black", linestyle="--")
# # plt.vlines(2.458934999836999923e+06, ymin=0, ymax=1, color="black", linestyle="--")
# # 
# # plt.vlines(2458934.9569444451, ymin=0, ymax=1, color="cyan", linestyle="--")
# # plt.vlines(2458934.9673611117, ymin=0, ymax=1, color="cyan", linestyle="--")
# # =============================================================================
# # =============================================================================
# # # Data 4 grænser
# # 
# # plt.vlines(2.458935444020000286e+06, ymin=0, ymax=1, color="indigo", linestyle="--")
# # plt.vlines(2.458935530391999986e+06, ymin=0, ymax=1, color="indigo", linestyle="--")
# # 
# # plt.vlines(2458935.4729166673, ymin=0, ymax=1, color="orange", linestyle="--")
# # plt.vlines(2458935.4937500004, ymin=0, ymax=1, color="orange", linestyle="--")
# # =============================================================================
# # =============================================================================
# # # Data 3 og 4 grænser
# # plt.vlines(2.458934948158999905e+06, ymin=0, ymax=1, color="black", linestyle="--")
# # plt.vlines(2.458934999836999923e+06, ymin=0, ymax=1, color="black", linestyle="--")
# # 
# # plt.vlines(2458934.9569444451, ymin=0, ymax=1, color="cyan", linestyle="--")
# # plt.vlines(2458934.9673611117, ymin=0, ymax=1, color="cyan", linestyle="--")
# # 
# # 
# # plt.vlines(2.458935444020000286e+06, ymin=0, ymax=1, color="indigo", linestyle="--")
# # plt.vlines(2.458935530391999986e+06, ymin=0, ymax=1, color="indigo", linestyle="--")
# # 
# # plt.vlines(2458935.4729166673, ymin=0, ymax=1, color="orange", linestyle="--")
# # plt.vlines(2458935.4937500004, ymin=0, ymax=1, color="orange", linestyle="--")
# # =============================================================================
# # =============================================================================
# # 
# # 200326 cryo tests:
# # 
# # UT
# # 
# # cryo off 10:58 ( 2458934.9569444451)
# # 
# # on again 11:13 (2458934.9673611117)
# #
# #
# # 200326 white box tests:
# # 
# # At UT= 22:39  started  ./tesscalibs-fast.script
# # 
# # At UT= 23:21 fans were disconnected  ( 2458935.4729166673)
# # 
# # At UT= 23:53 fans were connected again (2458935.4937500004)
# # 
# # At UT ~ 00:45 the script was stopped
# # 
# #  
# =============================================================================

# =============================================================================

#fig.patch.set_facecolor("#f2f2f2")
#plt.plot(y_pred,label="predicted")
#plt.xlabel('TEST')
#plt.ylabel('Temperatur')
#plt.title("TEST")
                        
plt.show()

save = False

if save:
    fig.savefig("plots/Data1extended.pdf",  facecolor=fig.get_facecolor())

# pdftoppm -png -r 300 filename.pdf filename