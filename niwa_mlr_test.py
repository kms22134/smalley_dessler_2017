from xarray import open_dataset
from smalley2017_regress import * 
from metpy.calc import height_to_pressure_std
from metpy.units import units
from scipy.signal import detrend
from scipy.stats import pearsonr
import numpy as np
from matplotlib import pyplot as plt
from sys import exit

fIn1 = 'data/CCMI_monthly_NIWA-UKCA_refC2_allVars.nc'
fIn2 = 'data/CCMI_monthly_NIWA-UKCA_refC2_allVars_22.nc'

dataIn1 = open_dataset(fIn1)
dataIn2 = open_dataset(fIn2)

data_500 = dataIn1.sel(plev = 50000,time = slice('2000','2097'),lat = slice(-30,30))
data_80 = dataIn2.sel(hybridP = 79,time = slice('2000','2097'),lat = slice(-30,30))

weights = np.cos(np.deg2rad(data_80.lat))
weights.name = "weights"

ta_500   = data_500.ta
tntsw_80 = data_80.tntsw
tntlw_80 = data_80.tntlw
h2o_80   = data_80.hus * 10**6

ta_500   = ta_500.groupby('time.year').mean('time')
tntlw_80 = tntlw_80.groupby('time.year').mean('time')
tntsw_80 = tntsw_80.groupby('time.year').mean('time')
h2o_80   = h2o_80.groupby('time.year').mean('time')

ta_500   = ta_500.weighted(weights)
tntlw_80 = tntlw_80.weighted(weights)
tntsw_80 = tntsw_80.weighted(weights)
h2o_80   = h2o_80.weighted(weights)

ta_500   = ta_500.mean(dim = ['lon','lat'])
tntlw_80 = tntlw_80.mean(dim = ['lon','lat'])
tntsw_80 = tntsw_80.mean(dim = ['lon','lat'])
h2o_80   = h2o_80.mean(dim = ['lon','lat'])

ta_500_anom = ta_500 - ta_500.mean()
tntlw_80_anom = tntlw_80 - tntlw_80.mean()
tntsw_80_anom = tntsw_80 - tntsw_80.mean()
h2o_80_anom   = h2o_80 - h2o_80.mean()
netR = (tntlw_80_anom.values + tntsw_80_anom.values) * 86400.*((1000./80.)**(2./7.))

ta_fft_detrend_anoms   = detrendAnom(10,ta_500_anom.values)
netR_fft_detrend_anoms = detrendAnom(10,netR)
h2o_fft_detrend_anoms = detrendAnom(10,h2o_80_anom.values)

ta_linear_detrend_anoms   = detrend(ta_500_anom.values)
netR_linear_detrend_anoms = detrend(netR)
h2o_linear_detrend_anoms  = detrend(h2o_80_anom.values)

xSeg      = np.zeros((h2o_linear_detrend_anoms.size,3))##2D array used for trended century regressions
xSeg[:,0] = np.ones(h2o_linear_detrend_anoms.size)##first column of array is ones for regression calculations
xSeg[:,1] = ta_linear_detrend_anoms##trended temperature anomolies
xSeg[:,2] = netR_linear_detrend_anoms##trended net radiative heating anomolies

regressSEG   = multiregress(##set a multi-linear regression python object for the trended regressions
			    h2o_linear_detrend_anoms,##set a multi-linear regression python object inputs: response variable-water vapor,
			    xSeg,##predictor variables-xSeg
			    h2o_linear_detrend_anoms.size,##length of the regression-dSet
			    3,##index.values.size,number of variables-varNums
			    True
			   )
print(regressSEG.coefficents)
print()
print(regressSEG.adjR2)
print()

print(pearsonr(ta_500_anom.values,h2o_80_anom))
print(pearsonr(ta_fft_detrend_anoms,h2o_fft_detrend_anoms))
print(pearsonr(ta_linear_detrend_anoms,h2o_linear_detrend_anoms))

fig = plt.figure(0)
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

ax1.plot(ta_500.year.values,ta_500_anom.values)
ax2.plot(ta_500.year.values,ta_fft_detrend_anoms)
ax2.plot(ta_500.year.values,ta_linear_detrend_anoms)

fig = plt.figure(1)
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

ax1.plot(ta_500.year.values,netR)
ax2.plot(ta_500.year.values,netR_fft_detrend_anoms)
ax2.plot(ta_500.year.values,netR_linear_detrend_anoms)

fig = plt.figure(2)
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)

ax1.scatter(ta_500_anom.values,h2o_80_anom)
ax2.scatter(ta_fft_detrend_anoms[1:-1],h2o_fft_detrend_anoms[1:-1])
ax3.scatter(ta_linear_detrend_anoms,h2o_fft_detrend_anoms)

plt.show()
