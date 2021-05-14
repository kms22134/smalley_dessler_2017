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

    model_lst        = []

    trended_r2_lst    = []
    trended_adjr2_lst = []

    detrended_r2_lst    = []
    detrended_adjr2_lst = []

    standard_trended_r2_lst    = []
    standard_trended_adjr2_lst = []

    standard_detrended_r2_lst    = []
    standard_detrended_adjr2_lst = []

    trended_temp_lst = []
    trended_bd_lst   = []
    trended_qbo_lst  = []

    trended_temp_lower_ci_lst = []
    trended_bd_lower_ci_lst   = []
    trended_qbo_lower_ci_lst  = []

    detrended_temp_lower_ci_lst = []
    detrended_bd_lower_ci_lst   = []
    detrended_qbo_lower_ci_lst  = []

    standard_trended_temp_lower_ci_lst = []
    standard_trended_bd_lower_ci_lst   = []
    standard_trended_qbo_lower_ci_lst  = []

    standard_detrended_temp_lower_ci_lst = []
    standard_detrended_bd_lower_ci_lst   = []
    standard_detrended_qbo_lower_ci_lst  = []

    trended_temp_upper_ci_lst = []
    trended_bd_upper_ci_lst   = []
    trended_qbo_upper_ci_lst  = []

    detrended_temp_upper_ci_lst = []
    detrended_bd_upper_ci_lst   = []
    detrended_qbo_upper_ci_lst  = []

    standard_trended_temp_upper_ci_lst = []
    standard_trended_bd_upper_ci_lst   = []
    standard_trended_qbo_upper_ci_lst  = []

    standard_detrended_temp_upper_ci_lst = []
    standard_detrended_bd_upper_ci_lst   = []
    standard_detrended_qbo_upper_ci_lst  = []

    detrended_temp_lst = []
    detrended_bd_lst   = []
    detrended_qbo_lst  = []

    standard_trended_temp_lst = []
    standard_trended_bd_lst   = []
    standard_trended_qbo_lst  = []

    standard_detrended_temp_lst = []
    standard_detrended_bd_lst   = []
    standard_detrended_qbo_lst  = []
    #adjR111 = OrderedDict([(cv,0) for cv in model])##full 21st century adjusted r2 values for the trended anomolies for each model
    #adjR222 = OrderedDict([(cv,0) for cv in model])##full 21st century adjusted r2 values for the detrended anomolies for each model

    #tempCoeffA  = OrderedDict([(cv,0) for cv in model])##full 21st century temperature coefficients for the trended anomolies for each model
    #tempSCoeffA = OrderedDict([(cv,0) for cv in model])##full 21st century temperature coefficients for the detrended anomolies for each model

    #tempConfA  = OrderedDict([(cv,0) for cv in model])##full 21st century temperature coefficient 95% confidence of the trended anomolies
    #tempSConfA = OrderedDict([(cv,0) for cv in model])##full 21st century temperature coefficient 95% confidence of the detrended anomolies

    #bdCoeffA  = OrderedDict([(cv,0) for cv in model])##full 21st century brewer-dobson coefficients for the trended anomolies for each model
    #bdSCoeffA = OrderedDict([(cv,0) for cv in model])##full 21st century brewer-dobson coefficients for the detrended anomolies for each model

    #bdConfA  = OrderedDict([(cv,0) for cv in model])##full 21st century brewer-dobson coefficient 95% confidence of the trended anomolies
    #bdSConfA = OrderedDict([(cv,0) for cv in model])##full 21st century brewer-dobson coefficient 95% confidence of the detrended anomolies

    #qboCoeffA  = OrderedDict([(cv,0) for cv in model])##full 21st century quasi-biennial oscillation coefficients for the trended anomolies for each model
    #qboSCoeffA = OrderedDict([(cv,0) for cv in model])##full 21st century quasi-biennial oscillation coefficients for the detrended anomolies for each model

    #qboConfA  = OrderedDict([(cv,0) for cv in model])##full 21st century quasi-biennial oscillationcoefficient 95% confidence of the trended anomolies
    #qboSConfA = OrderedDict([(cv,0) for cv in model])##full 21st century quasi-biennial oscillation coefficient 95% confidence of the detrended anomolies

    #modelTrendSlope = OrderedDict([(cv,0) for cv in model])##simulated water vapor trends for each model
    #modelTrendConf  = OrderedDict([(cv,0) for cv in model])##simulated water vapor trends for each model 95% confidence

    #tempTrendSlope = OrderedDict([(cv,0) for cv in model])##simulated water vapor trends resulting from tropospheric temperature for each model
    #tempTrendConf  = OrderedDict([(cv,0) for cv in model])##simulated water vapor trends resulting from troposheric temperature for each model 95% confidence

    #bdTrendSlope = OrderedDict([(cv,0) for cv in model])##simulated water vapor trends resulting from bd for each model
    #bdTrendConf  = OrderedDict([(cv,0) for cv in model])##simulated water vapor trends resulting from bd for each model 95% confidence

    #qboTrendSlope = OrderedDict([(cv,0) for cv in model])##simulated water vapor trends resulting from qbo for each model
    #qboTrendConf  = OrderedDict([(cv,0) for cv in model])##simulated water vapor trends resulting from qbo for each model 95% confidence

    #tempCoeffD = OrderedDict([(cv,[]) for cv in model])##decadal temperature coefficients for the monthly anomolies for each model
    #bdCoeffD   = OrderedDict([(cv,[]) for cv in model])##decadal bd coefficients for the monthly anomolies for each model
    #qboCoeffD  = OrderedDict([(cv,[]) for cv in model])##decadal qbo coefficients for the monthly anomolies for each model
    #adjR2D     = OrderedDict([(cv,[]) for cv in model])##decadal adjusted r2 for the monthly anomolies for each model

    for cv in model:
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

            qdot = tntlw + tntsw

            h2o_year_anoms   = yearly_anomolies(h2o).compute()
            temp_year_anoms  = yearly_anomolies(temp).compute()
            tntlw_year_anoms = yearly_anomolies(tntlw).compute()
            tntsw_year_anoms = yearly_anomolies(tntsw).compute()
            qdot_year_anoms  = yearly_anomolies(qdot).compute()

            ua_seasonal_anoms = monthly_anomolies(ua)
            ua_year_anoms     = yearly_anomolies(ua_seasonal_anoms)

            #qdot = tntlw_year_anoms + tntsw_year_anoms

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

            qdot = tntlw + tntsw

            h2o_year_anoms   = yearly_anomolies(h2o).compute()
            temp_year_anoms  = yearly_anomolies(temp).compute()
            tntlw_year_anoms = yearly_anomolies(tntlw).compute()
            tntsw_year_anoms = yearly_anomolies(tntsw).compute()
            qdot_year_anoms  = yearly_anomolies(qdot).compute()

            ua_seasonal_anoms = monthly_anomolies(ua)
            ua_year_anoms     = yearly_anomolies(ua_seasonal_anoms)
            #ua_year_anoms     = yearly_anomolies(ua / ua.std())

            #qdot = tntlw_year_anoms + tntsw_year_anoms

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

            qdot = tntlw + tntsw

            h2o_year_anoms   = yearly_anomolies(h2o).compute()
            temp_year_anoms  = yearly_anomolies(temp).compute()
            tntlw_year_anoms = yearly_anomolies(tntlw).compute()
            tntsw_year_anoms = yearly_anomolies(tntsw).compute()
            qdot_year_anoms  = yearly_anomolies(qdot).compute()

            ua_seasonal_anoms = monthly_anomolies(ua)
            ua_year_anoms     = yearly_anomolies(ua_seasonal_anoms)

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

            qdot = tntlw + tntsw

            h2o_year_anoms   = yearly_anomolies(h2o).compute()
            temp_year_anoms  = yearly_anomolies(temp).compute()
            tntlw_year_anoms = yearly_anomolies(tntlw).compute()
            tntsw_year_anoms = yearly_anomolies(tntsw).compute()
            qdot_year_anoms  = yearly_anomolies(qdot).compute()
            #print(tntsw_year_anoms)
            #break

            ua_seasonal_anoms = monthly_anomolies(ua)
            ua_year_anoms     = yearly_anomolies(ua_seasonal_anoms)
            #ua_year_anoms     = yearly_anomolies(ua / ua.std())

            #qdot = tntlw_year_anoms + tntsw_year_anoms

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

            qdot = tntlw + tntsw

            h2o_year_anoms   = yearly_anomolies(h2o).compute()
            temp_year_anoms  = yearly_anomolies(temp).compute()
            tntlw_year_anoms = yearly_anomolies(tntlw).compute()
            tntsw_year_anoms = yearly_anomolies(tntsw).compute()
            qdot_year_anoms  = yearly_anomolies(qdot).compute()
            #print(tntsw_year_anoms)
            #break

            ua_seasonal_anoms = monthly_anomolies(ua)
            ua_year_anoms     = yearly_anomolies(ua_seasonal_anoms)
            #ua_year_anoms     = yearly_anomolies(ua / ua.std())

            #qdot = tntlw_year_anoms + tntsw_year_anoms

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

            qdot = tntlw + tntsw

            h2o_year_anoms   = yearly_anomolies(h2o).compute()
            temp_year_anoms  = yearly_anomolies(temp).compute()
            #tntlw_year_anoms = yearly_anomolies(tntlw).compute()
            #tntsw_year_anoms = yearly_anomolies(tntsw).compute()
            qdot_year_anoms  = yearly_anomolies(qdot).compute()
            #print(tntsw_year_anoms)
            #break

            ua_seasonal_anoms = monthly_anomolies(ua)
            ua_year_anoms     = yearly_anomolies(ua_seasonal_anoms)
            #ua_year_anoms     = yearly_anomolies(ua / ua.std())

            #qdot = tntlw_year_anoms + tntsw_year_anoms 

        else:
            print('more models')
            break

        h2o_detrended  = detrendAnom(10,h2o_year_anoms.values) 
        temp_detrended = detrendAnom(10,temp_year_anoms.values) 
        qdot_detrended = detrendAnom(10,qdot_year_anoms.values) 
        qbo_detrended  = detrendAnom(10,ua_year_anoms.values)

        df_trended_dict = {
                            'h2o'  : Series(h2o_year_anoms.values), 
                            'temp' : Series(temp_year_anoms.values), 
                            'qdot' : Series(qdot_year_anoms.values), 
                            'qbo'  : Series(ua_year_anoms.values),
                          }

        df_detrended_dict = {
                            'h2o'  : Series(h2o_detrended),
                            'temp' : Series(temp_detrended),
                            'qdot' : Series(qdot_detrended),
                            'qbo'  : Series(qbo_detrended),
                          }

        df_trended_standardized_dict = {
                                         'h2o'  : Series(h2o_year_anoms.values / h2o_year_anoms.values.std()), 
                                         'temp' : Series(temp_year_anoms.values / temp_year_anoms.values.std()), 
                                         'qdot' : Series(qdot_year_anoms.values / qdot_year_anoms.values.std()), 
                                         'qbo'  : Series(ua_year_anoms.values / ua_year_anoms.values.std()),
                                       }


        df_detrended_standardized_dict = {
                            'h2o'  : Series(h2o_detrended / h2o_detrended.std()),
                            'temp' : Series(temp_detrended / temp_detrended.std()),
                            'qdot' : Series(qdot_detrended / qdot_detrended.std()),
                            'qbo'  : Series(qbo_detrended / qbo_detrended.std()),
                          }

        df_trended                = DataFrame(df_trended_dict)
        df_detrended              = DataFrame(df_detrended_dict)
        df_standardized_trended   = DataFrame(df_trended_standardized_dict)
        df_standardized_detrended = DataFrame(df_detrended_standardized_dict)

        ################################Regressors#############################
        X_trended                = df_trended[['temp','qdot','qbo']]
        X_detrended              = df_detrended[['temp','qdot','qbo']]
        X_standardized_trended   = df_standardized_trended[['temp','qdot','qbo']]
        X_standardized_detrended = df_standardized_detrended[['temp','qdot','qbo']]
        #######################################################################

        ################################Regression Constant####################
        X_trended                = sm.add_constant(X_trended) # adding a constant
        X_detrended              = sm.add_constant(X_detrended) # adding a constant
        X_standardized_trended   = sm.add_constant(X_standardized_trended) # adding a constant
        X_standardized_detrended = sm.add_constant(X_standardized_detrended) # adding a constant
        #######################################################################

        ################################water vapor############################
        Y_trended                = df_trended['h2o']
        Y_detrended              = df_detrended['h2o']
        Y_standardized_trended   = df_standardized_trended['h2o']
        Y_standardized_detrended = df_standardized_detrended['h2o']
        #######################################################################

        #regr_trended                = linear_model.LinearRegression()
        #regr_detrended              = linear_model.LinearRegression()
        #regr_standardized_detrended = linear_model.LinearRegression()

        #regr_trended.fit(X_trended, Y_trended)
        #regr_detrended.fit(X_detrended, Y_detrended)
        #regr_standardized_detrended.fit(X_standardized_detrended, Y_standardized_detrended)

        #########################Regression Analysis############################
        model_trended                = sm.OLS(Y_trended, X_trended).fit()
        model_detrended              = sm.OLS(Y_detrended, X_detrended).fit()
        model_standardized_trended   = sm.OLS(Y_standardized_trended, X_standardized_trended).fit()
        model_standardized_detrended = sm.OLS(Y_standardized_detrended, X_standardized_detrended).fit()
        ########################################################################

        #########################Predicted Water Vapor##########################
        predictions_trended                = model_trended.predict(X_trended)
        predictions_detrended              = model_detrended.predict(X_detrended)
        predictions_standardized_trended   = model_standardized_trended.predict(X_standardized_trended)
        predictions_standardized_detrended = model_standardized_detrended.predict(X_standardized_detrended)
        ########################################################################

        #######################Regression Coefficients##########################
        trended_coeffs                = model_trended.params
        detrended_coeffs              = model_detrended.params
        standardized_trended_coeffs   = model_standardized_trended.params
        standardized_detrended_coeffs = model_standardized_detrended.params
        ########################################################################

        #model_trended_summary = model_trended.summary()
        model_detrended_summary = model_detrended.summary()
        model_standardized_detrended_summary = model_standardized_detrended.summary()
        #print(model_standardized_detrended_summary)
        trended_r2,trended_adjr2 = multi_regress_R(num_years,model_trended.resid,model_trended.params.size,Y_trended,predictions_trended)
        detrended_r2,detrended_adjr2 = multi_regress_R(num_years,model_detrended.resid,model_detrended.params.size,Y_detrended,predictions_detrended)
        #standardized_trended_r2,standardized_trended_adjr2 = multi_regress_R(num_years,model_standardized_trended.resid,model_standardized_trended.params.size,Y_standardized_trended,predictions_standardized_trended)
        #standardized_detrended_r2,standardized_detrended_adjr2 = multi_regress_R(num_years,model_standardized_detrended.resid,model_standardized_detrended.params.size,Y_standardized_detrended,predictions_standardized_detrended)

        trended_lower_cf,trended_upper_cf = multi_regres_conf_inter(num_years,model_trended.resid,model_trended.bse,model_trended.params)
        detrended_lower_cf,detrended_upper_cf = multi_regres_conf_inter(num_years,model_detrended.resid,model_detrended.bse,model_detrended.params)
        #standard_trended_lower_cf,standard_trended_upper_cf = multi_regres_conf_inter(num_years,model_standardized_trended.resid,model_standardized_trended.bse,model_standardized_trended.params)
        #standard_detrended_lower_cf,standard_detrended_upper_cf = multi_regres_conf_inter(num_years,model_standardized_detrended.resid,model_standardized_detrended.bse,model_standardized_detrended.params)

        #print(f"     Trended: Temp Coeff: {trended_coeffs['temp']:.2f};   bd Coeff: {trended_coeffs['qdot']:.2e};   qbo Coeff: {trended_coeffs['qbo']:.2e}")
        #print(f"     Detrended: Temp Coeff: {detrended_coeffs['temp']:.2f};   bd Coeff: {detrended_coeffs['qdot']:.2e};   qbo Coeff: {detrended_coeffs['qbo']:.2e}")
        #print(f"     Standardized Detrended: Temp Coeff: {standardized_detrended_coeffs['temp']:.2f};   bd Coeff: {standardized_detrended_coeffs['qdot']:.2e};   qbo Coeff: {standardized_detrended_coeffs['qbo']:.2e}")
        ####################################################################################
        model_lst.append(cv)

        trended_r2_lst.append(trended_r2)
        trended_adjr2_lst.append(trended_adjr2)

        detrended_r2_lst.append(detrended_r2)
        detrended_adjr2_lst.append(detrended_adjr2)

        #standard_trended_r2_lst.append(standardized_trended_r2)
        #standard_trended_adjr2_lst.append(standardized_trended_adjr2)

        #standard_detrended_r2_lst.append(standardized_detrended_r2)
        #standard_detrended_adjr2_lst.append(standardized_detrended_adjr2)

        trended_temp_lower_ci_lst.append(trended_lower_cf['temp'])
        trended_bd_lower_ci_lst.append(trended_lower_cf['qdot'])
        trended_qbo_lower_ci_lst.append(trended_lower_cf['qbo'])

        detrended_temp_lower_ci_lst.append(detrended_lower_cf['temp'])
        detrended_bd_lower_ci_lst.append(detrended_lower_cf['qdot'])
        detrended_qbo_lower_ci_lst.append(detrended_lower_cf['qbo'])

        standard_trended_temp_lower_ci_lst.append(abs(trended_lower_cf['temp']) * df_trended['temp'].std())
        standard_trended_bd_lower_ci_lst.append(abs(trended_lower_cf['qdot']) * df_trended['qdot'].std())
        standard_trended_qbo_lower_ci_lst.append(abs(trended_lower_cf['qbo']) * df_trended['qbo'].std())

        standard_detrended_temp_lower_ci_lst.append(abs(detrended_lower_cf['temp']) * df_detrended['temp'].std())
        standard_detrended_bd_lower_ci_lst.append(abs(detrended_lower_cf['qdot']) * df_detrended['qdot'].std())
        standard_detrended_qbo_lower_ci_lst.append(abs(detrended_lower_cf['qbo']) * df_detrended['qbo'].std())

        trended_temp_upper_ci_lst.append(trended_upper_cf['temp'])
        trended_bd_upper_ci_lst.append(trended_upper_cf['qdot'])
        trended_qbo_upper_ci_lst.append(trended_upper_cf['qbo'])

        detrended_temp_upper_ci_lst.append(detrended_upper_cf['temp'])
        detrended_bd_upper_ci_lst.append(detrended_upper_cf['qdot'])
        detrended_qbo_upper_ci_lst.append(detrended_upper_cf['qbo'])

        standard_trended_temp_upper_ci_lst.append(abs(trended_upper_cf['temp']) * df_trended['temp'].std())
        standard_trended_bd_upper_ci_lst.append(abs(trended_upper_cf['qdot']) * df_trended['qdot'].std())
        standard_trended_qbo_upper_ci_lst.append(abs(trended_upper_cf['qbo']) * df_trended['qbo'].std())

        standard_detrended_temp_upper_ci_lst.append(abs(detrended_upper_cf['temp']) * df_detrended['temp'].std())
        standard_detrended_bd_upper_ci_lst.append(abs(detrended_upper_cf['qdot']) * df_detrended['qdot'].std())
        standard_detrended_qbo_upper_ci_lst.append(abs(detrended_upper_cf['qbo']) * df_detrended['qbo'].std())

        detrended_temp_lst.append(detrended_coeffs['temp'])
        detrended_bd_lst.append(detrended_coeffs['qdot'])
        detrended_qbo_lst.append(detrended_coeffs['qbo'])

        trended_temp_lst.append(trended_coeffs['temp'])
        trended_bd_lst.append(trended_coeffs['qdot'])
        trended_qbo_lst.append(trended_coeffs['qbo'])

        standard_trended_temp_lst.append(abs(trended_coeffs['temp']) * df_trended['temp'].std())
        standard_trended_bd_lst.append(abs(trended_coeffs['qdot']) * df_trended['qdot'].std())
        standard_trended_qbo_lst.append(abs(trended_coeffs['qbo']) * df_trended['qbo'].std())

        standard_detrended_temp_lst.append(abs(detrended_coeffs['temp']) * df_detrended['temp'].std())
        standard_detrended_bd_lst.append(abs(detrended_coeffs['qdot']) * df_detrended['qdot'].std())
        standard_detrended_qbo_lst.append(abs(detrended_coeffs['qbo']) * df_detrended['qbo'].std())

    trended_df_dict = {
                        'model'           : Series(model_lst), 
                        'temp'            : Series(trended_temp_lst), 
                        'temp lower 95th' : Series(trended_temp_lower_ci_lst), 
                        'temp upper 95th' : Series(trended_temp_upper_ci_lst), 
                        'bd'              : Series(trended_bd_lst), 
                        'bd lower 95th'   : Series(trended_bd_lower_ci_lst), 
                        'bd upper 95th'   : Series(trended_bd_upper_ci_lst), 
                        'qbo'             : Series(trended_qbo_lst),
                        'qbo lower 95th'  : Series(trended_qbo_lower_ci_lst), 
                        'qbo upper 95th'  : Series(trended_qbo_upper_ci_lst), 
                        'r2'              : Series(trended_r2_lst),
                        'adjusted r2'     : Series(trended_adjr2_lst),
                     }
    detrended_df_dict = {
                          'model'           : Series(model_lst), 
                          'temp'            : Series(detrended_temp_lst), 
                          'temp lower 95th' : Series(detrended_temp_lower_ci_lst), 
                          'temp upper 95th' : Series(detrended_temp_upper_ci_lst), 
                          'bd'              : Series(detrended_bd_lst), 
                          'bd lower 95th'   : Series(detrended_bd_lower_ci_lst), 
                          'bd upper 95th'   : Series(detrended_bd_upper_ci_lst), 
                          'qbo'             : Series(detrended_qbo_lst),
                          'qbo lower 95th'  : Series(detrended_qbo_lower_ci_lst), 
                          'qbo upper 95th'  : Series(detrended_qbo_upper_ci_lst), 
                          'r2'              : Series(detrended_r2_lst),
                          'adjusted r2'     : Series(detrended_adjr2_lst),
                        }
    standard_trended_df_dict = {
                                   'model'           : Series(model_lst), 
                                   'temp'            : Series(standard_trended_temp_lst), 
                                   'temp lower 95th' : Series(standard_trended_temp_lower_ci_lst), 
                                   'temp upper 95th' : Series(standard_trended_temp_upper_ci_lst), 
                                   'bd'              : Series(standard_trended_bd_lst), 
                                   'bd lower 95th'   : Series(standard_trended_bd_lower_ci_lst), 
                                   'bd upper 95th'   : Series(standard_trended_bd_upper_ci_lst), 
                                   'qbo'             : Series(standard_trended_qbo_lst),
                                   'qbo lower 95th'  : Series(standard_trended_qbo_lower_ci_lst), 
                                   'qbo upper 95th'  : Series(standard_trended_qbo_upper_ci_lst), 
                                   #'r2'              : Series(standard_trended_r2_lst),
                                   #'adjusted r2'     : Series(standard_trended_adjr2_lst),
                                 }
    standard_detrended_df_dict = {
                                   'model'           : Series(model_lst), 
                                   'temp'            : Series(standard_detrended_temp_lst), 
                                   'temp lower 95th' : Series(standard_detrended_temp_lower_ci_lst), 
                                   'temp upper 95th' : Series(standard_detrended_temp_upper_ci_lst), 
                                   'bd'              : Series(standard_detrended_bd_lst), 
                                   'bd lower 95th'   : Series(standard_detrended_bd_lower_ci_lst), 
                                   'bd upper 95th'   : Series(standard_detrended_bd_upper_ci_lst), 
                                   'qbo'             : Series(standard_detrended_qbo_lst),
                                   'qbo lower 95th'  : Series(standard_detrended_qbo_lower_ci_lst), 
                                   'qbo upper 95th'  : Series(standard_detrended_qbo_upper_ci_lst), 
                                   #'r2'              : Series(standard_detrended_r2_lst),
                                   #'adjusted r2'     : Series(standard_detrended_adjr2_lst),
                                 }

    trended_df            = DataFrame(trended_df_dict)
    detrended_df          = DataFrame(detrended_df_dict)
    standard_trended_df   = DataFrame(standard_trended_df_dict)
    standard_detrended_df = DataFrame(standard_detrended_df_dict)

    trended_df            = trended_df.set_index('model')
    detrended_df          = detrended_df.set_index('model')
    standard_trended_df   = standard_trended_df.set_index('model')
    standard_detrended_df = standard_detrended_df.set_index('model')

    trended_df.to_csv('regression_output_data/trended_data.csv')
    detrended_df.to_csv('regression_output_data/detrended_data.csv')
    standard_trended_df.to_csv('regression_output_data/standard_trended_data.csv')
    standard_detrended_df.to_csv('regression_output_data/standard_detrended_data.csv')

    print(trended_df)
    print()
    print(detrended_df)
    print()
    print(standard_trended_df)
    print()
    print(standard_detrended_df)

if __name__ == '__main__':
    main()
