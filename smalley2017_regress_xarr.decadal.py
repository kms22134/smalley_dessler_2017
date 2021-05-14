from datetime import datetime
from numpy import cos,deg2rad,zeros,average,fft,square,abs
from xarray import open_dataset,open_mfdataset,Dataset as xrDataset
from metpy.interpolate import log_interpolate_1d
from metpy.units import units
from pandas import Series,DataFrame
from sklearn import linear_model
import statsmodels.api as sm
from scipy import stats
from matplotlib import pyplot as plt
from sys import exit

def multi_regress_R(total_years,residuals,numVars,y,yHat):
    '''
    '''

    autoEST = residuals.autocorr(lag=1)##autocorrelation estimate
    adjx = autoEST
    adjx = (1 - adjx) / (1 + adjx)##adjustment factor for the degrees of freedom

    YST = y - y.mean()
    YSR = yHat - y.mean()

    SR2 = square(YSR)
    ST2 = square(YST)

    SSR = SR2.sum()
    SST = ST2.sum()
    PVE = SSR / SST

    effectiveDF = total_years * adjx##effective degrees of freedom

    R2 = PVE

    return R2,1-(((float(effectiveDF - 1))/(float(effectiveDF - numVars)))*(1-R2))
#    return 1-(((float(effectiveDF - 1))/(float(effectiveDF - numVars)))*(1-R2))

def multi_regres_conf_inter(total_years,residuals,standard_err,coeffs):
    #******************************************************************************************
    #******************************************************************************************

    autoEST = residuals.autocorr(lag=1)##autocorrelation estimate
    adjx = autoEST
    adjx = (1 - adjx) / (1 + adjx)##adjustment factor for the degrees of freedom

    #print(autoEST,adjx,total_years * adjx)
    t_std = stats.t((total_years * adjx) - coeffs.size - 1).isf(0.025) * standard_err

    return coeffs - t_std, coeffs + t_std

    #lowerConfidenceBound = self.coefficents - (stats.t((self.totYears*self.adjx) - self.numVars-1).isf(0.025)*self.ste)
    #upperConfidenceBound = self.coefficents + (stats.t((self.totYears*self.adjx) - self.numVars-1).isf(0.025)*self.ste)
    #self.Confidence = upperConfidenceBound - self.coefficents

def detrendAnom(cut,anom):
    #****************************************************************************
    #Purpose: detrend data using a boxcar filter
    #
    #Inputs:
    #  cut: cutoff frequency
    #  anom: anomoly values
    #
    #Output:
    #  varfft: detrended data
    #
    #****************************************************************************

    cuttoff    = cut
    varFFT     = fft.fft(anom)
    period     = 1/fft.fftfreq(varFFT.size,d = 1.0)
    varFFT    *= (abs(period) < cut)
    inverseFFT = fft.ifft(varFFT)
    varfft     = inverseFFT
    varfft     = (varfft).real

    return varfft

#def hybridPressure(corr1,corr2,surfP,dS,fullVar,VertVar,latVar,initialP = 1,check = False):
def hybridPressure(data,a_coeff = 'a',b_coeff = 'b',initial_press = 'p0',surf_press = 'ps',check = False):
    #**************************************************************************************
    #Purpose: approximate pressure levels given hybrid pressure surfaces
    #
    #Inputs:
    #  a_coeff: a variable from netcdf file
    #  b_coeff: b variable from netcdf file
    #  surfP: surface pressure given in file
    #  dS: netcdf4 dataset
    #  fullVar: multi-dimensional array from dataset
    #  VertVar: levels variable
    #  latVar: latitude variable
    #  initialP: initial pressure:
    #  check: boolean check on initialP
    #
    #Outputs:
    #  bPa: approximate isobaric surfaces
    #***************************************************************************************

    #from numpy import cos,pi,zeros,average

    bPa = zeros(data.ta.shape)

    cosLat = cos(deg2rad(data.lat))


    a   = data[a_coeff]
    b   = data[b_coeff]
    if(not check): p0  = data[initial_press]
    ps  = data[surf_press]
    lev = data.lev 

    h2o    = data.vmrh2o.metpy.quantify()
    ta     = data.ta.metpy.quantify()
    tntlw  = data.tntlw.metpy.quantify()
    tntsw  = data.tntsw.metpy.quantify()
    ua     = data.ua.metpy.quantify()
    #print(data.ta)
    #exit()

    if(check): aP = a
    else:      aP = a*p0

    for x in range(lev.shape[0]):
        bPa[:,x,:,:] = aP[x] + (b[x]*ps[:,:,:])
    bPa = units.Quantity(bPa,'Pa')

    plevs = [500.,80.,50.] * units.hPa
    h2o,ta,tntlw,tntsw,ua = log_interpolate_1d(plevs, bPa, h2o, ta, tntlw, tntsw, ua, axis=1)

    xarr_dataset = xrDataset(
                              data_vars = {
                                            'vmrh2o'    : (['time','plev','lat','lon'],h2o.m), 
                                            'ta'    : (['time','plev','lat','lon'],ta.m), 
                                            'tntlw' : (['time','plev','lat','lon'],tntlw.m),
                                            'tntsw' : (['time','plev','lat','lon'],tntsw.m),
                                            'ua'    : (['time','plev','lat','lon'],ua.m), 
                                          },
                              coords = {
                                         'time' : data.time.values,
                                         'plev' : plevs.m,
                                         'lat'  : data.lat.values,
                                         'lon'  : data.lon.values,
                                       }
                            )

    for k in ['vmrh2o','ta','tntlw','tntsw','ua']:
        for kk,v in data[k].attrs.items():
            xarr_dataset.ta.attrs[kk] = v

    return xarr_dataset
    #bPa = bPa.mean(axis = 3)##zonal average
    #bPa = bPa.mean(axis = 0)##temporal average
    #bPa = average(bPa,axis = 1,weights = cosLat)##temporal average
    
    #for t in range(bPa.shape[0]):
    #    for x in range(bPa.shape[2]):
    #        for y in range(bPa.shape[3]):
    #            print(bPa[t,:,x,y])


    #a = dS.variables[corr1][:]
    #b = dS.variables[corr2][:]
    #lat = dS.variables[latVar][:]
    #lev = dS.variables[VertVar][:]
    #if(check):
    #    p0 = dS.variables[initialP][:]
    #else:
    #    p0 = initialP

    #ps = dS.variables[surfP][:,:,:]

    #    bPa[:,x,:,:] = aP[x] + (b[x]*ps[:,:,:])

    #bPa = average(bPa,axis = 3)##zonal average
    #bPa = average(bPa,axis = 0)##temporal average
    #bPa = average(bPa,axis = 1,weights = cosLat)##meridional average
    #bPa = bPa/100.##convert from Pa to hPa

    #return bPa

def constrain_data_range(data,y1,y2):
    '''
    '''

    return data.sel(time = slice(f'{y1:d}',f'{y2:d}'))
    
def constrain_lat_range(data,lat1,lat2):
    '''
    '''

    if(data.lat[0].values - data.lat[1].values < 0):
        return data.sel(lat = slice(lat1,lat2))
    else:
        return data.sel(lat = slice(lat2,lat1))
    
def weighted_lat_mean(data):
    '''
    '''

    weights = cos(deg2rad(data.lat))
    weights.name = 'weights'

    data_weighted = data.weighted(weights)
    data_weighted_mean = data_weighted.mean(('lon','lat'))

    return data_weighted_mean

def yearly_anomolies(data,):
    '''
    '''

    return data.groupby('time.year').mean('time') - data.mean()

def monthly_anomolies(data,):
    '''
    '''

    return data.groupby('time.month') - data.groupby('time.month').mean('time')

def main():
    #***********************************************************************************************
    #Creator: Kevin Smalley
    #Date: 07/10/17
    #Office: 1002E O&M, TAMU
    #
    #Phone Number: (281)-748-7349
    #Email: ksmalley@tamu.edu
    #
    #Purpose: This script conducts the multivariate linear regression used on multiple
    #         CCMVal-2/CCMI-1 model water vapor using tropospheric temperature, brewer-dobson, and
    #         quasi-biennial oscillation anomolies as predictors in Smalley et al. (2017), and
    #         produces the same figures found within the paper.
    #
    #Inputs/Outputs: Documentation of classes, functions, and variables are found within the code
    #                comments
    #
    #If discrepencies are found PLEASE contact me using the contacts above
    #
    #***********************************************************************************************

    ##modules
    from collections import OrderedDict
    from netCDF4 import Dataset
    from numpy import arange,cos,pi,ma,median,zeros,ones,size,product
    from scipy.signal import detrend
    ##models to loop over
    model = [
             'MRI',
             'WACCM',
             'CMAM',
             'CCSRNIES',
             'LMDZrepro',
             'GEOSCCM',
             'CCSRNIES-MIROC3.2',
             'CNRM-CM5-3',
             'NIWA-UKCA',
             'CMAM_CCMI',
             'GEOSCCM_CCMI',
             'MRI-ESM1r1'
            ]

    model = sorted(model)##sort the models being evaluated alphebetically

    ###boolian directors
    teraD = True##true if we want decadal analysis
    teraA = True##true if we want century analysis

    seg = 10##length of a decade in years

    num_years = 98##number of years
    year0     = 2000##beginning year
    yearF     = 2097##ending year
    #yearF     = 2100##ending year

    lR = (-30,30)##latitude bounds defines the "tropics"

    segmentOfInterest = 10##number of segments + 1 to run (decadal data)
    varNums = 4##number of predictors for the multiple linear regression analysis

    ##Observational-Based Regression adjusted r2 values calculated by Dessler et al. (2014)
    ERAIadjR2  = 0.75
    MERRAadjR2 = 0.72

    ##Observational-Based Regression Coefficients (ERA and MERRA reanalysis) calculated by Dessler et al. (2014)
    eraITemp,merraTemp          = 0.34,0.30##Temperature coefficients
    eraITempConf,merraTempConf  = 0.17,0.20##Temperature coefficient 95% confidence

    eraIBD,merraBD         = -2.51,-3.48##Brewer-Dobson coefficients
    eraIBDConf,merraBDConf = 0.83,1.62##Brewer-Dobson coefficient 95% confidence

    eraIQBO,merraQBO         = 0.11,0.12##Quasi-Biennial Oscilation coefficients
    eraIQBOConf,merraQBOConf = 0.04,0.05##Quasi-Biennial Oscilation coefficient 95% confidence

    ##Observational-Based Standardized Regression Coefficients (ERA and MERRA reanalysis) calculated by Dessler et al. (2014)
    eraITempD,merraTempD         = 0.36,0.34##Temperature coefficients
    eraITempConfD,merraTempConfD = 0.17,0.21##Temperature coefficient 95% confidence

    eraIQBOD,merraQBOD         = 0.46,0.49##Brewer-Dobson coefficients
    eraIQBOConfD,merraQBOConfD = 0.18,0.23##Brewer-Dobson coefficient 95% confidence

    eraIBDD,merraBDD         = -0.56,-0.49##Quasi-Biennial Oscilation coefficients
    eraIBDConfD,merraBDConfD = 0.19,0.22##Quasi-Biennial Oscilation coefficient 95% confidence

    ##initialize lists to use later
    #adjR111,adjR222 = [],[]##full 21st century adjusted r2 values for the trended anomolies for each model
                           ##full 21st century adjusted r2 values for the detrended anomolies for each model

    temp_regressor_dict = {}
    qdot_regressor_dict = {}
    qbo_regressor_dict = {}
    monthly_regressor_dict          = {'model' : [],'temp' : [],'qdot' : [],'qbo' : []}
    standard_monthly_regressor_dict = {'model' : [],'temp' : [],'qdot' : [],'qbo' : []}
    for cv in model:
        temp_regressor_dict[cv] = []
        qdot_regressor_dict[cv] = []
        qbo_regressor_dict[cv]  = []
        ##***********************************keep in main function*************************************
        ##determine if simulation is in CCMVal-2 or CCMI-1 experiments
        print(cv)
        if(
           cv == 'CCSRNIES-MIROC3.2'
        or cv == 'CNRM-CM5-3'
        or cv == 'NIWA-UKCA'
        or cv == 'CMAM_CCMI'
        or cv == 'GEOSCCM_CCMI'
        or cv == 'MRI-ESM1r1'
          ):
            experiment = 'CCMI-1'
        else:
            experiment = 'CCMVal-2'

        ##ensemble member used
        if(cv == 'LMDZrepro'):
            num = '3'
        else:
            num = '1'
        ##**********************************rest add to "tasks" function******************************
        if(
           cv != 'GEOSCCM'
       and experiment == 'CCMVal-2'
          ):
            finUa = '/Users/ksmalley/multiple_regression/data/CCMVal2_REF-B2_' + cv +'_' + num + '_T3M_ua.nc'##zonal wind file
            finT  = '/Users/ksmalley/multiple_regression/data/CCMVal2_REF-B2_' + cv +'_' + num + '_T3M_ta.nc'##temperature file
            finW  = '/Users/ksmalley/multiple_regression/data/CCMVal2_REF-B2_' + cv +'_' + num + '_T3M_H2O.nc'##water vapor file
            finL  = '/Users/ksmalley/multiple_regression/data/CCMVal2_REF-B2_' + cv +'_' + num + '_T3M_tntlw.nc'##longwave heat flux
            finS  = '/Users/ksmalley/multiple_regression/data/CCMVal2_REF-B2_' + cv +'_' + num + '_T3M_tntsw.nc'##shortwave heat flux
            
        elif(
             cv == 'GEOSCCM'
         and experiment == 'CCMVal-2'
            ):
            #finA  = '/co1/ksmalley/CCMVal/' + cv + '/R2/CCMVal2_REF-B2_corr_' + cv + '_'+num+'_vars.nc'##all varaibles from GEOSCCM
            finA  = '/Users/ksmalley/multiple_regression/data/' + 'CCMVal2_REF-B2_corr_' + cv  + '_' + num + '_vars.nc'##all varaibles from GEOSCCM
            #print(finA)
        elif(
             experiment == 'CCMI-1'
         and cv != 'NIWA-UKCA'
            ):
            finA   = '/Users/ksmalley/multiple_regression/data/' + 'CCMI_monthly_' + cv + '_refC2_allVars.nc'##all variables for ALL CCMI-1 models EXCEPT NIWA-UKCA
            #print(finA)
        elif(cv == 'NIWA-UKCA'):
            finAA  = '/Users/ksmalley/multiple_regression/data/' + 'CCMI_monthly_' + cv + '_refC2_allVars_22.nc'##NIWA-UKCA
            finA   = '/Users/ksmalley/multiple_regression/data/' + 'CCMI_monthly_' + cv + '_refC2_allVars.nc'

        if(cv != 'GEOSCCM' and experiment == 'CCMVal-2'):
            data_vars = open_mfdataset([finUa,finT,finW,finL,finS,])
            #print(data_vars.plev.attrs['units'])

            data_vars = constrain_data_range(data_vars,year0,yearF)
            data_vars_equator  = data_vars.sel(lat = 0.,method = 'nearest')
            data_vars_weighted = constrain_lat_range(data_vars,-30,30)
            data_vars_weighted = weighted_lat_mean(data_vars_weighted)

            h2o   = data_vars_weighted.H2O.sel(plev = 80.) * 1e6
            temp  = data_vars_weighted.ta.sel(plev = 500.)
            tntlw = data_vars_weighted.tntlw.sel(plev = 80.) * 86400. * ((1000./80.)**(2./7.))
            tntsw = data_vars_weighted.tntsw.sel(plev = 80.) * 86400. * ((1000./80.)**(2./7.))
            ua    = data_vars_equator.ua.sel(plev = 50.).mean(dim = 'lon').compute()
            ua_seasonal_anoms = monthly_anomolies(ua)

            qdot = tntlw + tntsw

        elif(cv == 'GEOSCCM' and experiment == 'CCMVal-2'):
            data_vars = open_dataset(finA)
            #print(data_vars.plev.values)

            data_vars = constrain_data_range(data_vars,year0,yearF)

            data_vars_equator  = data_vars.sel(lat = 0.,method = 'nearest')
            data_vars_weighted = constrain_lat_range(data_vars,-30,30)
            data_vars_weighted = weighted_lat_mean(data_vars_weighted)

            h2o   = data_vars_weighted.H2O.sel(plev = 80.) * 1e6
            temp  = data_vars_weighted.ta.sel(plev = 500.)
            tntlw = data_vars_weighted.tntlw.sel(plev = 80.) * 86400. * ((1000./80.)**(2./7.))
            tntsw = data_vars_weighted.tntsw.sel(plev = 80.) * 86400. * ((1000./80.)**(2./7.))
            ua    = data_vars_equator.ua.sel(lev = 50.).mean(dim = 'lon').compute()

            h2o    = h2o.where(h2o < 1e4)
            temp   = temp.where(temp < 1e4)
            tntsw  = tntsw.where(tntsw < 1e4)
            tntlw  = tntlw.where(tntlw < 1e4)
            ua     = ua.where(ua < 1e4)
            ua_seasonal_anoms = monthly_anomolies(ua)

            qdot = tntlw + tntsw

        elif(cv == 'CCSRNIES-MIROC3.2' or cv == 'CMAM_CCMI'):
            data_vars = open_dataset(finA)

            data_vars = constrain_data_range(data_vars,year0,yearF)
            data_vars = hybridPressure(data_vars,a_coeff = 'ap',check = True)
            #print(data_vars.plev.values)

            data_vars_equator  = data_vars.sel(lat = 0.,method = 'nearest')
            data_vars_weighted = constrain_lat_range(data_vars,-30,30)
            data_vars_weighted = weighted_lat_mean(data_vars_weighted)

            h2o   = data_vars_weighted.vmrh2o.sel(plev = 80.) * 1e6
            temp  = data_vars_weighted.ta.sel(plev = 500.)
            tntlw = data_vars_weighted.tntlw.sel(plev = 80.) * 86400. * ((1000./80.)**(2./7.))
            tntsw = data_vars_weighted.tntsw.sel(plev = 80.) * 86400. * ((1000./80.)**(2./7.))
            ua    = data_vars_equator.ua.sel(plev = 50.).mean(dim = 'lon').compute()
            ua_seasonal_anoms = monthly_anomolies(ua)

            qdot = tntlw + tntsw

        elif(cv == 'GEOSCCM_CCMI' or cv == 'MRI-ESM1r1'):
            data_vars = open_dataset(finA)

            data_vars = constrain_data_range(data_vars,year0,yearF)

            data_vars = hybridPressure(data_vars)
            #print(data_vars.plev.values)

            data_vars_equator  = data_vars.sel(lat = 0.,method = 'nearest')
            data_vars_weighted = constrain_lat_range(data_vars,-30,30)
            data_vars_weighted = weighted_lat_mean(data_vars_weighted)

            h2o   = data_vars_weighted.vmrh2o.sel(plev = 80.) * 1e6
            temp  = data_vars_weighted.ta.sel(plev = 500.)
            tntlw = data_vars_weighted.tntlw.sel(plev = 80.) * 86400. * ((1000./80.)**(2./7.))
            tntsw = data_vars_weighted.tntsw.sel(plev = 80.) * 86400. * ((1000./80.)**(2./7.))
            ua    = data_vars_equator.ua.sel(plev = 50.).mean(dim = 'lon').compute()
            ua_seasonal_anoms = monthly_anomolies(ua)

            qdot = tntlw + tntsw

        elif(cv == 'CNRM-CM5-3'):
            data_vars = open_dataset(finA)
            #print(data_vars.plev.attrs['units'])
            data_vars['plev'] = data_vars.plev / 100.
            data_vars.plev.attrs['units'] = 'hPa'

            data_vars = constrain_data_range(data_vars,year0,yearF)

            data_vars_equator  = data_vars.sel(lat = 0.,method = 'nearest')
            data_vars_weighted = constrain_lat_range(data_vars,-30,30)

            h2o   = weighted_lat_mean(data_vars_weighted.vmrh2o)
            temp  = weighted_lat_mean(data_vars_weighted.ta)
            tntlw = weighted_lat_mean(data_vars_weighted.tntlw)
            tntsw = weighted_lat_mean(data_vars_weighted.tntsw)
            ua    = data_vars_equator.ua

            h2o   = h2o.sel(plev = 80.) * 1e6
            temp  = temp.sel(plev = 500.)
            tntlw = tntlw.sel(plev = 80.) * 86400. * ((1000./80.)**(2./7.))
            tntsw = tntsw.sel(plev = 80.) * 86400. * ((1000./80.)**(2./7.))
            ua    = ua.sel(plev = 50.).mean(dim = 'lon').compute()
            ua_seasonal_anoms = monthly_anomolies(ua)

            qdot = tntlw + tntsw

        elif(cv == 'NIWA-UKCA'):
            data_vars    = open_dataset(finA)
            #print(data_vars.plev.attrs['units'])
            data_vars_hp = open_dataset(finAA)
            data_vars['plev'] = data_vars.plev / 100.
            data_vars         = data_vars.assign_coords(lev = data_vars_hp.hybridP.values)

            data_vars = constrain_data_range(data_vars,year0,yearF)

            data_vars_equator  = data_vars.sel(lat = 0.,method = 'nearest')
            data_vars_weighted = constrain_lat_range(data_vars,-30,30)

            h2o   = weighted_lat_mean(data_vars_weighted.hus)
            temp  = weighted_lat_mean(data_vars_weighted.ta)
            tntlw = weighted_lat_mean(data_vars_weighted.tntlw)
            tntsw = weighted_lat_mean(data_vars_weighted.tntsw)
            ua    = data_vars_equator.ua

            #h2o   = h2o.sel(lev = 79) * (28. / (18. + 28.)) * 1e6
            h2o   = h2o.sel(lev = 79) * 1e6
            temp  = temp.sel(plev = 500.)
            tntlw = tntlw.sel(lev = 79) * 86400. * ((1000./79.)**(2./7.))
            tntsw = tntsw.sel(lev = 79) * 86400. * ((1000./79.)**(2./7.))
            ua    = ua.sel(plev = 50.).mean(dim = 'lon').compute()
            ua_seasonal_anoms = monthly_anomolies(ua)

            qdot = tntlw + tntsw

        else:
            print('more models')
            break

        for year in range(2000,2100,10):
            #print(year,year + 10)
            decadal_h2o  = constrain_data_range(h2o,year,year + 10).compute()
            decadal_temp = constrain_data_range(temp,year,year + 10).compute()
            decadal_qdot = constrain_data_range(qdot,year,year + 10).compute()
            decadal_ua   = constrain_data_range(ua_seasonal_anoms,year,year + 10).compute()

            h2o_monthly_anoms  = monthly_anomolies(decadal_h2o)
            temp_monthly_anoms = monthly_anomolies(decadal_temp)
            qdot_monthly_anoms = monthly_anomolies(decadal_qdot)
            ua_monthly_anoms   = monthly_anomolies(decadal_ua)

            h2o_monthly_anoms.name  = 'h2o'
            temp_monthly_anoms.name = 'temp'
            qdot_monthly_anoms.name = 'qdot'
            ua_monthly_anoms.name   = 'qbo'

            df_monthly_dict = {
                                'h2o'  : h2o_monthly_anoms.to_series(),
                                'temp' : temp_monthly_anoms.to_series(),
                                'qdot' : qdot_monthly_anoms.to_series(),
                                'qbo'  : ua_monthly_anoms.to_series(),
                              }
            df_monthly = DataFrame(df_monthly_dict)

            df_monthly['temp'] = df_monthly['temp'].shift(3)
            df_monthly['qdot'] = df_monthly['qdot'].shift(1)
            df_monthly['qbo']  = df_monthly['qbo'].shift(3)
            df_monthly         = df_monthly.dropna()

            decadal_h2o_predictors = df_monthly[['temp','qdot','qbo']]
            decadal_h2o_values     = df_monthly['h2o']

            decadal_h2o_predictors = sm.add_constant(decadal_h2o_predictors) # adding a constant

            decadal_regression = sm.OLS(decadal_h2o_values, decadal_h2o_predictors).fit()

            temp_regressor_dict[cv].append(decadal_regression.params['temp'])
            qdot_regressor_dict[cv].append(decadal_regression.params['qdot'])
            qbo_regressor_dict[cv].append(decadal_regression.params['qbo'])

            monthly_regressor_dict['model'].append(cv)
            monthly_regressor_dict['temp'].append(decadal_regression.params['temp'])
            monthly_regressor_dict['qdot'].append(decadal_regression.params['qdot'])
            monthly_regressor_dict['qbo'].append(decadal_regression.params['qbo'])

            standard_monthly_regressor_dict['model'].append(cv)
            standard_monthly_regressor_dict['temp'].append(abs(decadal_regression.params['temp'] * df_monthly['temp'].std()))
            standard_monthly_regressor_dict['qdot'].append(abs(decadal_regression.params['qdot'] * df_monthly['qdot'].std()))
            standard_monthly_regressor_dict['qbo'].append(abs(decadal_regression.params['qbo'] * df_monthly['qbo'].std()))

    monthly_regressor_df          = DataFrame(monthly_regressor_dict)
    standard_monthly_regressor_df = DataFrame(standard_monthly_regressor_dict)

    monthly_regressor_df.to_csv('regression_output_data/decadal_regression_params.csv')
    standard_monthly_regressor_df.to_csv('regression_output_data/decadal_regression_standard_params.csv')

    print(monthly_regressor_df)
    print()
    print(standard_monthly_regressor_df)

if __name__ == '__main__':
    main()
