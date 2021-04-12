from xarray import open_dataset
from sklearn import linear_model
import statsmodels.api as sm

class multiregress:
    #********************************************************************************
    #Purpose: perform a multivariate linear regression on a set of variables, and
    #         calculate various statistics
    #********************************************************************************

    def __init__(self,dependent,independent,totYears,totVars,dBool):
        #************************************************************************
        #Purpose: initiate multiregress pyObject
        #
        #Inputs:
        #       self: connect variables to itself
        #       dependent: explainitory variables
        #       independent: predictor variables
        #       totYears: total number of years
        #       totVars: total number of vars
        #       dBool: if century True, if decade (False)
        #
        #************************************************************************

        self.numVars  = totVars##number of variables
        self.y        = dependent##explaintory variable
        self.X        = independent##independent variables
        self.totYears = totYears##total number of years
        self.tSpan    = dBool##check as to whether century or decade
        self.author   = 'Kevin Smalley'##author

        self.multiInitiate()##initialize multivarite linear regression
        self.multiRegress()##do all the various calculations
        self.multiStats()##calculate other stats
        self.confInter()##calculate confidence interval on regression coefficients

    def multiInitiate(self):
        #*************************************************************************
        #Purpose: initiate multivariate linear regression
        #
        #Inputs:
        #  self: this pyObject
        #*************************************************************************

        from numpy import zeros,dot,linalg

        #-----------------do all matrix calculations required--------------------
        self.XtimesXprime = zeros((self.numVars,self.totYears))
        self.Xprime_timesY = zeros((self.numVars,self.totYears))
        self.inverseX = zeros((self.numVars,self.totYears))
        self.coefficents = zeros(self.numVars)
        self.upperConfidenceBound = zeros(self.numVars)
        self.lowerConfidenceBound = zeros(self.numVars)
        self.plusMinusConfidence = zeros(self.numVars)
        self.residual = zeros(self.totYears)
        self.XtimesXprime  = dot((self.X).T,self.X)
        self.inverseX      = linalg.inv(self.XtimesXprime)
        self.Xprime_timesY = dot((self.X).T,self.y)
        #------------------------------------------------------------------------

        self.autoEST = 0
        self.adjx = 0

        self.coefficents = dot(self.inverseX,self.Xprime_timesY)##coefficents contains the regressor info

    def multiRegress(self):
        #************************************************************************
        #Purpose: Calculate Full Regression
        #
        #Inputs: this pyObject
        #************************************************************************

        from numpy import dot
        from pandas import Series

        sumy = 0
        for i in range(0,self.numVars):
            allVarTot = self.coefficents[i]*self.X[:,i]
            sumy = allVarTot + sumy

        self.Yhat = sumy##response variable estimator

        self.residual = self.y - dot(self.X,self.coefficents)##Calculate Residuals

        if(self.tSpan):
            self.autoEST = Series(self.residual).autocorr(lag=1)##autocorrelation estimate
        else:
            self.autoEST = Series(self.residual).autocorr(lag=1)##autocorrelation estimate

        self.adjx = self.autoEST
        self.adjx=(1-self.adjx)/(1+self.adjx)##adjustment factor for the degrees of freedom

        if(self.tSpan):
            self.adjx = self.adjx
        else:
            self.adjx = 1.

        self.effectiveDF = self.totYears*self.adjx##effective degrees of freedom
        #self.effectiveDF = self.totYears##effective degrees of freedom

    def multiStats(self):
        #*************************************************************************************
        #Purpose: calculate multivariate linear regression statistics
        #
        #Inputs: this pyObject
        #
        #*************************************************************************************

        from numpy import average,size,square,sqrt,zeros
        from scipy import stats

        self.Ybar = average(self.y)
        self.YST = self.y - self.Ybar

        self.YSR = self.Yhat - self.Ybar

        ##Square Sum Info
        r2 = self.residual*self.residual
        self.SSE = sum(r2)

        ST2 = square(self.YST)
        self.SST = sum(ST2)

        SR2 = square(self.YSR)
        self.SSR = sum(SR2)

        ##Mean Sum of Squares Info
        self.MST = self.SST/(self.effectiveDF - 1)
        self.MSE = self.SSE/(self.effectiveDF - self.numVars)
        self.MSR = self.SSR/self.numVars

        ##Root Mean Squares of Sum Info:
        self.RMST = sqrt(self.MST)
        self.RMSE = sqrt(self.MSE)
        self.RMSR = sqrt(self.MSR)

        self.NRMSE = self.RMSE/(max(self.y) - min(self.y))

        ##Standard Deviation of Yhat
        self.SD = sqrt(self.MSE)

        ##Proportion of explaned variance
        PVE = self.SSR/self.SST

        ##Coefficient Standard Errors model
        self.ste = sqrt(self.inverseX.diagonal()*self.SD*self.SD)

        ##Variance-Covariance matrix
        self.varCovar = self.inverseX*self.inverseX*self.MSE

        ##STE for individual points on multiple linear regression
        sumZ = zeros(size(self.X[:,0]))
        xT = self.X.T
        for i in range(0,size(self.varCovar[:,0])):
            for ii in range(0,size(self.varCovar[0,:])):
                sumZ[:] += self.varCovar[i,ii]*xT[i,:]*self.X[:,ii]
        sumZ = sqrt(sumZ)
        self.pointSTE = sumZ

        ##Standardized Coefficients (T-Statistic)
        self.t = self.coefficents/self.ste
        self.oT = stats.t((self.totYears*self.adjx) - self.numVars-1).isf(0.025)

        ##F-test (test for significance of the regression)
        self.f  = self.MSR/self.MSE
        self.f0 = stats.f(self.numVars-1,(self.totYears*self.adjx)-self.numVars-1).isf(0.05)

        ##Coeffient of Multiple Determination
        self.R2 = PVE
        self.adjR2 = 1-(((float(self.effectiveDF - 1))/(float(self.effectiveDF - self.numVars)))*(1-self.R2))

    def confInter(self):
        #******************************************************************************************
        #Purpose: Confidence interval
        #
        #Inputs:
        #  self: this pyObject
        #
        #******************************************************************************************

        from scipy import stats

        lowerConfidenceBound = self.coefficents - (stats.t((self.totYears*self.adjx) - self.numVars-1).isf(0.025)*self.ste)
        upperConfidenceBound = self.coefficents + (stats.t((self.totYears*self.adjx) - self.numVars-1).isf(0.025)*self.ste)
        self.Confidence = upperConfidenceBound - self.coefficents


def hybridPressure(corr1,corr2,surfP,dS,fullVar,VertVar,latVar,initialP = 1,check = False):
    #**************************************************************************************
    #Purpose: approximate pressure levels given hybrid pressure surfaces
    #
    #Inputs:
    #  corr1: a variable from netcdf file
    #  corr2: b variable from netcdf file
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

    from numpy import cos,pi,zeros,average

    a = dS.variables[corr1][:]
    b = dS.variables[corr2][:]
    lat = dS.variables[latVar][:]
    lev = dS.variables[VertVar][:]
    if(check):
        p0 = dS.variables[initialP][:]
    else:
        p0 = initialP

    ps = dS.variables[surfP][:,:,:]

    cosLat = cos(lat*(pi/180))

    aP = a*p0

    bPa = zeros(fullVar.shape)

    for x in range(lev.shape[0]):
        bPa[:,x,:,:] = aP[x] + (b[x]*ps[:,:,:])

    bPa = average(bPa,axis = 3)##zonal average
    bPa = average(bPa,axis = 0)##temporal average
    bPa = average(bPa,axis = 1,weights = cosLat)##meridional average
    bPa = bPa/100.##convert from Pa to hPa

    return bPa

def netConfig(latRange,vertRange,var,lat,lev,weight = True):
    #*******************************************************************************************
    #Purpose: configure each netcdf variable
    #
    #Inputs:
    #  latRange: latitude range
    #  vertRange: pressure levels
    #  var: array of interest
    #  lat: latitude array
    #  lev: pressure array
    #
    #Outputs:
    #  squeeze(var): squeeze out any 1-element dimensions
    #********************************************************************************************

    from numpy import cos,pi,max,min,squeeze,average,deg2rad

    cosLat = cos(lat*(pi/180.))
    lev = lev
    if(weight): var = average(var[:,:,(lat>=min(latRange))&(lat<=max(latRange))],axis = 2,weights = cosLat[(lat>=min(latRange))&(lat<=max(latRange))])
    else:       var = average(var[:,:,(lat>=min(latRange))&(lat<=max(latRange))],axis = 2,)
    var = var[:,(lev>=min(vertRange))&(lev<=max(vertRange))]

    return squeeze(var)

def dates(timeVar,year1,year2,month1 = 1,month2 = 12):
    #************************************************************************************
    #Purpose: go from nc date timestamps to actual dates
    #
    #Inputs:
    #  timeVar: netcdf time variable
    #  year1: initial year in analysis period
    #  year2: final year in analysis period
    #  month1: first month in initial year, set to January
    #  month2: last month in final year, set to December
    #
    #Outputs:
    #  yMask: boolean array used to determine if dates fall within analysis period
    #  unique(years): unique years in the analysis period
    #  allYmask: boolean array, always true (size of the simulation)
    #
    #************************************************************************************

    from netCDF4 import num2date##convert time-stamp to date
    from numpy import asarray,unique#asarray: convert python list to array (for masking)
                                    #unique: take only unique elements of input array

    newDates = num2date(timeVar[:],units = timeVar.units)##dates in simulation

    years  = asarray([y.year for y in newDates])##years in simulation
    months = asarray([m.month for m in newDates])##months in simulation

    allYmask = asarray([True for y in newDates])
    yMask    = (years >= year1) & (years <= year2)

    years  = years[yMask]##years after maskin
    months = months[yMask]##months after masking

    return yMask,unique(years),allYmask##return values

def leastSquares1D(x,y):
    #************************************************************************************
    #Purpose: calculate slope and intercept for a linear least-squares regression
    #
    #Input:
    #  x: input array
    #  y: comparison input array
    #
    #Output:
    #  beta0: y-intercept estimate
    #  beta1: linear slope estimate
    #
    #************************************************************************************

    from numpy import average,sum#average: average arrays
                           #sum: sum elements of an array

    n = len(y)##number of elements in BOTH arrays

    xAvg = average(x)##calculate x array average
    yAvg = average(y)##calculate y array average

    Sxy = sum((x - xAvg)*(y - yAvg))##calculate the cross-sum of squares
    Sxx = sum((x - xAvg)*(x - xAvg))##calculate the sum of squares for x

    beta1 = Sxy/Sxx
    beta0 = yAvg - (beta1*xAvg)

    return beta0,beta1

def leastSquaresSeries(b0,b1,x):
    #***********************************************************************************
    #Purpose: calculate a least squared best-fit line
    #
    #Inputs:
    #  b0: y-intercept value
    #  b1: slope value
    #  x: independent variable
    #
    #outputs:
    #  b0 + (b1 * x): best fit values
    #***********************************************************************************

    return b0 + (b1 * x)

def monAmoms(var):
    #**********************************************************************************
    #Purpose: calculate monthly anomolies for a given dataset
    #
    #Inputs:
    #  var: input array
    #
    #Outputs:
    #  var1: monthly anomoly values
    #**********************************************************************************

    from numpy import arange,ma,where,zeros

    var1 = zeros(var.size)
    indx = arange(var.size) % 12##use to separate months

    for i in range(12):
        xx = where(indx == i)[0]
        var1[xx] = var[xx] - ma.average(var[xx],axis = 0)

    return var1

def annualAnoms(var):
    #**********************************************************************************
    #Purpose: Calculate yearly anomolies for a given dataset
    #
    #Inputs:
    #  var: input array
    #
    #Output:
    #  var - varAvg: yearly anomolies
    #
    #**********************************************************************************

    if(len(var.shape) != 2):
        raise ValueError('must be 2-dimensional')

    from numpy import average

    var = average(var,axis = 1)##average along year dimension
    varAvg = average(var)

    return var - varAvg

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

    from numpy import fft

    cuttoff    = cut
    varFFT     = fft.fft(anom)
    period     = 1/fft.fftfreq(varFFT.size,d = 1.0)
    varFFT    *= (abs(period) < cut)
    inverseFFT = fft.ifft(varFFT)
    varfft     = inverseFFT
    varfft     = (varfft).real

    return varfft

def trendCalc(X,Y):
    #***************************************************************************************************************
    #Purpose: return a,b in solution to y = ax + b such that root mean square distance between trend line and
    #         original points is minimized
    #
    #inputs:
    #  X: one array
    #  Y: other array
    #
    #outputs:
    #  (Sxy * N - Sy * Sx)/det:trend slope
    #  (Sxx * Sy - Sx * Sxy)/det:trend intercept
    #  ste * stats.t(N - 2).isf(0.025):2-tailed 95% confidence
    #
    #***************************************************************************************************************

    from math import sqrt
    from numpy import sum
    from scipy import stats

    N = len(X)
    Sx = Sy = Sxx = Syy = Sxy = 0.0
    for x, y in zip(X, Y):
        Sx = Sx + x
        Sy = Sy + y
        Sxx = Sxx + x*x
        Syy = Syy + y*y
        Sxy = Sxy + x*y

    det = Sxx * N - Sx * Sx

    NewY = ((Sxy * N - Sy * Sx)/det * X) + (Sxx * Sy - Sx * Sxy)/det

    SSe = (Y - NewY) * (Y - NewY)
    SSe = sum(SSe)

    MSe = SSe/(N - 2)

    ste = sqrt(MSe/Sxx)


    return (Sxy * N - Sy * Sx)/det, (Sxx * Sy - Sx * Sxy)/det, ste * stats.t(N - 2).isf(0.025)

def leastSquaresSeries(b0,b1,x):
    #**************************************************************************
    #Purpose: calculate a least-squares line
    #
    #Inputs:
    #  b0: intercept value
    #  b1: slope value
    #
    #Outputs:
    #  y: line values
    #
    #**************************************************************************
    from numpy import zeros,size

    y = zeros(size(x))
    y = b0 + (b1*x)

    return y

def regressionPlot(model,waterVapor,waterVaporEstimator,waterVapor_p1,waterVapor_p2,waterVapor_p3,years,y0 = 2000,y1 = 2100,run = True):
    #***************************************************************************
    #Purpose: Plot the full 21st century regression and estimator as a result
    #         of each predictor on the response variable
    #
    #Inputs:
    #  model: list of the individual models
    #  waterVapor: simulated water vapor
    #  waterVaporEstimator: estimated water vapor by the regression
    #  waterVapor_p1: estimated water vapor by tropospheric temperature alone
    #  waterVapor_p2: estimated water vapor by brewer-dobson alone
    #  waterVapor_p3: estimated water vapor by quasi-biennial oscillation alone
    #  years: temporal variable (years)
    #  y0: initial year
    #  y1: final year
    #  run: determine if to create plot
    #
    #Outputs:
    #  create plot image
    #
    #****************************************************************************

    if not run:##if run is false, don't create plot
        return

    import matplotlib as mpl
    from matplotlib import pyplot as plt
    import matplotlib.ticker as mtick

    #--------------------set plot font settings-----------------------------------
    font = {
            'family' : 'Bitstream Vera Sans',
            'weight' : 'normal',
            'size'   : 24
           }

    mpl.rc('font', **font)
    #----------------------------------------------------------------------------

    fig = plt.figure(0)##initialize a new figure

    ax1 = fig.add_subplot(1,1,1)##include only one plot on figure

    ax1.plot(
             years,##x-axis variable
             waterVapor,##y-axis variable
             linewidth = 2.5,##line-width
             color = 'black',##line-color
             linestyle = 'solid',##plot solid line
             label = ' $\mathregular{[H_{2}O]_{entry}}$'##set line-label
            )
    ax1.plot(
             years,##x-axis variables
             waterVaporEstimator,##y-axis variable
             linewidth = 2.5,##line-width
             color = 'tan',##line-color
             linestyle = 'solid',##plot solid line
             label = '$\mathregular{\\beta_{\Delta{T}}\Delta{T}}$ + $\mathregular{\\beta_{BDC}BDC}$ + $\mathregular{\\beta_{QBO}QBO}$'##set line-label
            )
    ax1.plot(
             years,##x-axis variables
             waterVapor_p1,##y-axis variable
             linewidth = 2.5,##line-width
             color = 'red',##line-color
             linestyle = 'solid',##plot solid line
             label = '$\mathregular{\\beta_{\Delta{T}}\Delta{T}}$'##set line-label
            )
    ax1.plot(
             years,##x-axis variables
             waterVapor_p2,##y-axis variable
             linewidth = 2.5,##line-width
             color = 'green',##line-color
             linestyle = 'solid',##plot solid line
             label = '$\mathregular{\\beta_{BDC}BDC}$'##set line-label
            )
    ax1.plot(
             years,##x-axis variables
             waterVapor_p3,##y-axis variable
             linewidth = 2.5,##line-width
             color = 'blue',##line-color
             linestyle = 'solid',##plot solid line
             label = '$\mathregular{\\beta_{QBO}QBO}$'##set line-label
            )

    ax1.set_ylabel(##y-axis label
                   'Annual Anomolies (ppmv)',
                   fontsize = 32
                  )

    ax1.set_xlim([y0, y1])##confine y-axis to y0 and y1 limits

    ax1.xaxis.grid(True,'major')##set x-axis grid-lines
    ax1.yaxis.grid(True,'major')##set y-axis grid-lines

    ax1.legend(loc='upper left',fontsize = 24.)##setup a legend for plot (place in best place according to matplotlib

    plt.show()##show the plot

def centuryAdjR2(pModels,trend_adj,detrend_adj,run = True):
    #************************************************************************************************
    #Purpose: To plot the century regression adjusted R2 values for both the trended and detrended
    #         regressions
    #
    #Inputs:
    #  pModels: models analyzed
    #  trend_adj: trended century adjusted r2 values
    #  detrend_adj: detrended century adjusted r2 values
    #  run: determine if the plot should be created
    #
    #Outputs:
    #  show century adjusted r2 plot
    #
    #************************************************************************************************

    if not run:##if run is false, don't create plot
        return

    from numpy import arange,mean,std
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    import matplotlib.ticker as mtick

    #--------------------set plot font settings-----------------------------------
    font = {
            'family' : 'Bitstream Vera Sans',
            'weight' : 'normal',
            'size'   : 24
           }

    mpl.rc('font', **font)
    #----------------------------------------------------------------------------

    ind = arange(1,len(pModels)+1)##plot index depending on model
    width = 0.4##bar width, and use to space bars

    fig1 = plt.figure(0)
    ax = plt.subplot2grid((6,6), (0, 0),rowspan = 6,colspan = 6)

    ax.bar(##century trended adjusted r2 values plotted as bars for each model
           ind-width,##bar position
           trend_adj,##adj r2 values
           width,##bar width
           color = '#f5f5f5'##plot off-white bars
          )
    ax.plot(
            (ind[-1] + 2)-width/2.,##marker postion
            mean(trend_adj),##average of the century trended adjusted r2 values
            marker = 'o',##marker type 'circle'
            color = '#f5f5f5',##plot off-white marker
            markersize = 20.,##marker size
            linestyle = 'None'##don't plot a line
           )
    ax.errorbar(
                (ind[-1] + 2)-width/2.,##error-bar position
                mean(trend_adj),
                yerr=std(trend_adj),##standard deviation of century trended adjusted r2 values, repressents error-bars
                linestyle="None",
                elinewidth=3.5,##errorbar linewidth
                capsize = 15,##errorbar cap size
                capthick=2,##errorbar cap thickness
                color='black'##errorbar color
               )

    ax.bar(
           ind,
           detrend_adj,##detrended century adjusted r2 values
           width,
           color = 'lightgrey'
         )
    ax.plot(
            (ind[-1] + 2)+width/2.,
            mean(detrend_adj),
            marker = 'o',
            markersize = 20.,
            linestyle = 'None',
            color = 'lightgrey'
           )
    ax.errorbar(
                (ind[-1] + 2)+width/2.,
                mean(detrend_adj),
                yerr=std(detrend_adj),
                linestyle="None",
                elinewidth=3.5,
                capsize = 15,
                capthick=2,
                color='black'
               )

    ax.axvline(##solid line separating individual model values from ensemble average values
               x=ind[-1] + 1,
               color = 'black',
               linestyle = 'solid',
               linewidth = 3.
              )

    ax.set_ylabel('Adjusted $\mathregular{R^{2}}$',fontsize = 32)##y-axis label

    ax.xaxis.set_ticks(list(range(1,len(pModels)+3,1)))##x-axis tickmarks
    ax.yaxis.set_ticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])##y-axis tickmarks

    ax.set_ylim([0,1])##y-axis limit
    ax.set_xlim([0,15])##x-axis limit

    ax.xaxis.grid(True,'major')##x-axis gridlines
    ax.yaxis.grid(True,'major')##y-axis gridlines

    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))##format y-axis labels to have 2 decimals

    pModels.append('')##append a empty string to model list in preparation for next step
    pModels.append('Ensemble Averages')##append a new value to the model list for the ensemble average values
    ax.set_xticklabels(pModels,rotation = 90)##set x-axis tickmarks to the name of each model

    plt.show()##show plot

def decadeAdjR2(pModels,adj,era,merra,run = True):
    #******************************************************************************************
    #Purpose: plot decadal adjusted r2 values for models, observations, and model averages
    #
    #Inputs:
    #  pModels: list of models analyzed
    #  adj: decadal adjusted r2 values
    #  era: eraI regressed by MLS water vapor adjusted r2 as calculated by Dessler et al. (2014)
    #  merra: merra regressed by MLS water vapor adjusted r2 as calculated by Dessler et al. (2014)
    #  run: determine if plot should be created
    #
    #Outputs:
    #  plot image
    #
    #******************************************************************************************

    if not run:##if run is false, don't create plot
        return

    from numpy import arange,median,mean,std
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    import matplotlib.ticker as mtick

    #--------------------set plot font settings-----------------------------------
    font = {
            'family' : 'Bitstream Vera Sans',
            'weight' : 'normal',
            'size'   : 24
           }

    mpl.rc('font', **font)
    #----------------------------------------------------------------------------

    fig = plt.figure(0)

    ax = fig.add_subplot(1,1,1)

    ensemble = []##use for plotting ensemble values
    for i,r in enumerate(adj,1):
        ax.plot(##average of all decadal adjusted r2 values for each model
                i,##position corresponding to each model
                median(r),
                marker = 'o',
                markersize = 20.,
                linestyle = 'None',
                color = 'lightgrey'
               )

        ax.errorbar(##standard deviation of all adjusted r2 values for each model
                    i,
                    median(r),
                    yerr=std(r),
                    linestyle="None",
                    elinewidth=3.5,
                    capsize = 15,
                    capthick=2,
                    color='black'
                   )
        for en in r:
            ensemble.append(en)
    i += 1
    ax.axvline(##solid line separating individual model values from ensemble average values
               x=i,
               color = 'black',
               linestyle = 'solid',
               linewidth = 3.
              )
    i += 1
    ax.plot(##average of all decadal adjusted r2 values for ensemble
            i,##position corresponding to the ensemble
            mean(ensemble),
            marker = 'o',
            markersize = 20.,
            linestyle = 'None',
            color = 'lightgrey'
           )

    ax.errorbar(##standard deviation of all adjusted r2 values for ensemble
                i,
                mean(ensemble),
                yerr=std(ensemble),
                linestyle="None",
                elinewidth=3.5,
                capsize = 15,
                capthick=2,
                color='black'
               )
    ax.axhline(##dashed line representing MERRA
               y = era,
               color = 'black',
               linestyle = 'dotted',
               linewidth = 3.
              )
    ax.axhline(##dashed line representing MERRA
               y = merra,
               color = 'black',
               linestyle = 'dashed',
               linewidth = 3.
              )
    ax.set_ylabel('Decadal Adjusted $\mathregular{R^{2}}$',fontsize = 32)##y-axis label

    pModels.append('')##append a empty string to model list in preparation for next step
    pModels.append('Ensemble Averages')##append a new value to the model list for the ensemble average values
    ax.set_xticklabels(pModels,rotation = 90)##set x-axis tickmarks to the name of each model

    ax.xaxis.set_ticks(list(range(1,len(pModels)+3,1)))##x-axis tickmarks
    ax.yaxis.set_ticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])##y-axis tickmarks

    ax.set_ylim([0,1])##y-axis limit
    ax.set_xlim([0,15])##x-axis limit

    ax.xaxis.grid(True,'major')##x-axis gridlines
    ax.yaxis.grid(True,'major')##y-axis gridlines

    plt.show()

def centuryRegressionCoeffPlot(pModels,newTemp,newTempSTD,newTemp_det,newTempSTD_det,newbd,newbdSTD,newbd_det,newbdSTD_det,newqbo,newqboSTD,newqbo_det,newqboSTD_det,run = True):
    #******************************************************************************************
    #
    #******************************************************************************************

    if not run:
        return##don't print stats

    from numpy import arange,mean,std
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    import matplotlib.ticker as mtick

    #--------------------set plot font settings-----------------------------------
    font = {
            'family' : 'Bitstream Vera Sans',
            'weight' : 'normal',
            'size'   : 24
           }

    mpl.rc('font', **font)
    #----------------------------------------------------------------------------

    ind   = arange(1,len(pModels) + 1)
    width = 0.4

    fig2 = plt.figure(3)

    ax1 = plt.subplot2grid((12,12), (0, 0),rowspan = 4,colspan = 12)
    ax2 = plt.subplot2grid((12,12), (4,0),rowspan = 4,colspan = 12)
    ax3 = plt.subplot2grid((12,12), (8,0),rowspan = 4,colspan = 12)

    #--------------------------------century temperature coefficients------------
    ax1.plot(##trended temperature century coefficients
             ind - width/2.,
             newTemp,
             marker = 'o',
             color = '#f5f5f5',
             markersize = 20.,
             linestyle = 'None',
            )
    ax1.errorbar(##trended temperautre coefficient 95% confidence errorbars
                 ind - width/2.,
                 newTemp,
                 yerr = newTempSTD,
                 linestyle="None",
                 elinewidth=3.5,
                 capsize = 15,
                 capthick=2,
                 color='black'
                )
    ax1.plot(##detrended temperature century coefficients
             ind + width/2.,
             newTemp_det,
             marker = 'o',
             color = 'lightgrey',
             markersize = 20.,
             linestyle = 'None',
            )
    ax1.errorbar(##detrended temperature coefficient 95% confidence errorbars
                 ind + width/2.,
                 newTemp_det,
                 yerr = newTempSTD_det,
                 linestyle="None",
                 elinewidth=3.5,
                 capsize = 15,
                 capthick=2,
                 color='black'
                )
    ax1.axvline(##solid line separating individual model values from ensemble average values
                x=ind[-1] + 1,
                color = 'black',
                linestyle = 'solid',
                linewidth = 3.
               )
    ax1.plot(##mean value of the trended century temperature coefficients
             (ind[-1] + 2) - width/2.,
             mean(newTemp),
             marker = 'o',
             color = '#f5f5f5',
             markersize = 20.,
             linestyle = 'None',
            )
    ax1.errorbar(##standard deviation of all trended century temperature coefficients
                 (ind[-1] + 2) - width/2.,
                 mean(newTemp),
                 yerr = std(newTemp),
                 linestyle="None",
                 elinewidth=3.5,
                 capsize = 15,
                 capthick=2,
                 color='black'
                )
    ax1.plot(##mean value of the detrended century temperature coefficients
             (ind[-1] + 2) + width/2.,
             mean(newTemp_det),
             marker = 'o',
             color = 'lightgrey',
             markersize = 20.,
             linestyle = 'None',
            )
    ax1.errorbar(##standard deviation of all detrended century temperature coefficients
                 (ind[-1] + 2) + width/2.,
                 mean(newTemp_det),
                 yerr = std(newTemp_det),
                 linestyle="None",
                 elinewidth=3.5,
                 capsize = 15,
                 capthick=2,
                 color='black'
                )
    ax1.axhline(##solid zero line
                y = 0,
                color = 'black',
                linestyle = 'solid',
                linewidth = 3.
               )

    ax1.set_ylabel("$\mathregular{\\beta_{\Delta{T}}}$",fontsize = 22)

    ax1.xaxis.grid(True,'major')
    ax1.yaxis.grid(True,'major')

    ax1.xaxis.set_ticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
    ax1.yaxis.set_ticks([0.0,0.15,0.3,0.45,0.6,0.75])

    ax1.set_ylim([0.0,0.75])
    ax1.set_xlim([0,15])

    ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

    plt.setp(ax1.get_xticklabels(), visible=False)##don't show x-axis labels for this plot
    #------------------------------------------------------------------------------------------
    #---------------------------brewer-dobson coefficient plot---------------------------------
    ax2.plot(##trended brewer-doebson century coefficinets
             ind - width/2.,
             newbd,
             marker = 'o',
             color = '#f5f5f5',
             markersize = 20.,
             linestyle = 'None'
            )
    ax2.errorbar(##trended brewer-dobson coefficient 95% co0nfidence
                 ind - width/2.,
                 newbd,
                 yerr = newbdSTD,
                 linestyle = "None",
                 elinewidth = 3.5,
                 capsize = 15,
                 capthick = 2,
                 color = 'black'
                )
    ax2.plot(##detrended brewer-dobson century coefficients
             ind + width/2.,
             newbd_det,
             marker = 'o',
             color = 'lightgrey',
             markersize = 20.,
             linestyle = 'None'
            )
    ax2.errorbar(##detrended brewer-dobson coefficient 95% confidence
                 ind + width/2.,
                 newbd_det,
                 yerr = newbdSTD_det,
                 linestyle = "None",
                 elinewidth = 3.5,
                 capsize = 15,
                 capthick = 2,
                 color = 'black'
                )
    ax2.axvline(##solid line separating individual model values from ensemble average values
                x=ind[-1] + 1,
                color = 'black',
                linestyle = 'solid',
                linewidth = 3.
               )
    ax2.plot(##average of the century trended brewer-dobson coefficinets
             (ind[-1] + 2) - width/2.,
             mean(newbd),
             marker = 'o',
             color = '#f5f5f5',
             markersize = 20.,
             linestyle = 'None'
            )
    ax2.errorbar(##trended brewer-dobson coefficient standard deviation
                 (ind[-1] + 2) - width/2.,
                 mean(newbd),
                 yerr = std(newbdSTD),
                 linestyle = "None",
                 elinewidth = 3.5,
                 capsize = 15,
                 capthick = 2,
                 color = 'black'
                )
    ax2.plot(#average of the century detrended brewer-dobson coefficients
             (ind[-1] + 2) + width/2.,
             mean(newbd_det),
             marker = 'o',
             color = 'lightgrey',
             markersize = 20.,
             linestyle = 'None'
            )
    ax2.errorbar(##detrended brewer-dobson coefficient standard deviation
                 (ind[-1] + 2) + width/2.,
                 mean(newbd_det),
                 yerr = std(newbdSTD_det),
                 linestyle = "None",
                 elinewidth = 3.5,
                 capsize = 15,
                 capthick = 2,
                 color = 'black'
                )
    ax2.axhline(##solid zero line
                y = 0,
                color = 'black',
                linestyle = 'solid',
                linewidth = 3.
               )

    ax2.set_ylabel("$\mathregular{\\beta_{\Delta{BDC}}}$",fontsize = 22)

    ax2.xaxis.grid(True,'major')
    ax2.yaxis.grid(True,'major')

    ax2.xaxis.set_ticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])

    ax2.set_ylim([-15,10])
    ax2.set_xlim([0,15])

    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

    plt.setp(ax2.get_xticklabels(), visible=False)
    #------------------------------------------------------------------------------------------
    #---------------------------------qbo century coefficients---------------------------------
    ax3.plot(##century trended qbo coefficients
             ind - width/2.,
             newqbo,
             marker = 'o',
             color = '#f5f5f5',
             markersize = 20.,
             linestyle = 'None'
            )
    ax3.errorbar(##century tended qbo coefficient 95% confidence
                 ind - width/2.,
                 newqbo,
                 yerr = newqboSTD,
                 linestyle="None",
                 elinewidth=3.5,
                 capsize = 15,
                 capthick=2,
                 color='black'
                )
    ax3.plot(##century detrended qbo coefficients
             ind + width/2.,
             newqbo_det,
             marker = 'o',
             color = 'lightgrey',
             markersize = 20.,
             linestyle = 'None'
            )
    ax3.errorbar(##century detrended qbo coefficient 95% confidence
                 ind + width/2.,
                 newqbo_det,
                 yerr = newqboSTD_det,
                 linestyle="None",
                 elinewidth=3.5,
                 capsize = 15,
                 capthick=2,
                 color='black'
                )
    ax3.axvline(##solid line separating individual model values from ensemble average values
                x=ind[-1] + 1,
                color = 'black',
                linestyle = 'solid',
                linewidth = 3.
               )
    ax3.plot(##average of the century trended brewer-dobson coefficinets
             (ind[-1] + 2) - width/2.,
             mean(newqbo),
             marker = 'o',
             color = '#f5f5f5',
             markersize = 20.,
             linestyle = 'None'
            )
    ax3.errorbar(##trended brewer-dobson coefficient standard deviation
                 (ind[-1] + 2) - width/2.,
                 mean(newqbo),
                 yerr = std(newqboSTD),
                 linestyle = "None",
                 elinewidth = 3.5,
                 capsize = 15,
                 capthick = 2,
                 color = 'black'
                )
    ax3.plot(#average of the century detrended brewer-dobson coefficients
             (ind[-1] + 2) + width/2.,
             mean(newqbo_det),
             marker = 'o',
             color = 'lightgrey',
             markersize = 20.,
             linestyle = 'None'
            )
    ax3.errorbar(##detrended brewer-dobson coefficient standard deviation
                 (ind[-1] + 2) + width/2.,
                 mean(newqbo_det),
                 yerr = std(newqboSTD_det),
                 linestyle = "None",
                 elinewidth = 3.5,
                 capsize = 15,
                 capthick = 2,
                 color = 'black'
                )
    ax3.axhline(##solid zero line
                y = 0,
                color = 'black',
                linestyle = 'solid',
                linewidth = 3.
               )

    ax3.set_ylabel("$\mathregular{\\beta_{\Delta{QBO}}}$",fontsize = 22)

    ax3.xaxis.grid(True,'major')
    ax3.yaxis.grid(True,'major')

    ax3.xaxis.set_ticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])

    ax3.set_ylim([-0.10,0.15])
    ax3.set_xlim([0,15])

    ax3.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

    pModels.append('')
    pModels.append('Ensemble Averages')
    ax3.set_xticklabels(pModels,rotation=90)##use model names as tick labels for all three
                                            ##plots
    #------------------------------------------------------------------------------------------

    plt.show()

def decadeRegressionCoeffPlot(pModels,temp,bdc,qbo,era,eraSTD,merra,merraSTD,run = True):
    #*****************************************************************************************
    #Purpose: create decadal regression plot
    #
    #Inputs:
    #  pModels: list of models analyzed
    #  temp: temperature decadal coefficients for each individual model
    #  bdc: bd decadal coefficients for each individual model
    #  qbo: qbo decadal coefficients for each individual model
    #  era: eraI coefficient as calculated by Dessler et al. (2014)
    #  eraSTD: eraI coefficient 95% confidence as calculated by Dessler et al. (2014)
    #  merra: merra coefficient as calculated by Dessler et al. (2014)
    #  merraSTD: merra coefficient 95% confidence as calculated by Dessler et al. (2014)
    #  run: determine if plot should be created
    #
    #Outputs:
    #  create plot image
    #
    #*****************************************************************************************

    if not run:
        return##don't print stats

    from numpy import arange,median,mean,std
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    import matplotlib.ticker as mtick

    #--------------------set plot font settings-----------------------------------
    font = {
            'family' : 'Bitstream Vera Sans',
            'weight' : 'normal',
            'size'   : 24
           }

    mpl.rc('font', **font)
    #----------------------------------------------------------------------------

    fig = plt.figure(0)

    ax1 = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2)
    ax3 = fig.add_subplot(3,1,3)

    ensembleT = []##all temperature coefficients from all models
    ensembleB = []##all bd coefficients from all models
    ensembleQ = []##all qbo coefficients from all models
    i = 1##postion of each model on plot
    for a,b,c in zip(temp,bdc,qbo):##zip temperature, bdc, and qbo coefficients into tuples for plotting
        ax1.plot(##average temperature coefficient for each model
                 i,
                 median(a),
                 marker = 'o',
                 markersize = 20.,
                 linestyle = 'None',
                 color = 'lightgrey'
                )
        ax1.errorbar(##standard deviation of temperature coefficients for each model
                     i,
                     median(a),
                     yerr=std(a),
                     linestyle="None",
                     elinewidth = 3.5,
                     capsize = 20,
                     capthick = 2,
                     color='black'
                    )

        ax2.plot(##average bd coefficient for each model
                 i,
                 median(b),
                 marker = 'o',
                 markersize = 20.,
                 linestyle = 'None',
                 color = 'lightgrey',
                )
        ax2.errorbar(##standard deviation of bd coefficients for each model
                     i,
                     median(b),
                     yerr=std(b),
                     linestyle="None",
                     elinewidth = 3.5,
                     capsize = 20,
                     capthick = 2,
                     color='black'
                    )

        ax3.plot(##average qbo coefficient for each model
                 i,
                 median(c),
                 marker = 'o',
                 markersize = 20.,
                 linestyle = 'None',
                 color = 'lightgrey',
                )
        ax3.errorbar(##standard deviation of qbo coefficients for each model
                     i,
                     median(c),
                     yerr=std(c),
                     linestyle="None",
                     elinewidth = 3.5,
                     capsize = 20,
                     capthick = 2,
                     color='black'
                    )

        for aa,bb,cc in zip(a,b,c):
            ensembleT.append(aa)
            ensembleB.append(bb)
            ensembleQ.append(cc)

        i += 1
    ax1.axvline(##solid line separating individual model values from ensemble average values
                x=i,
                color = 'black',
                linestyle = 'solid',
                linewidth = 3.
               )
    ax2.axvline(##solid line separating individual model values from ensemble average values
                x=i,
                color = 'black',
                linestyle = 'solid',
                linewidth = 3.
               )
    ax3.axvline(##solid line separating individual model values from ensemble average values
                x=i,
                color = 'black',
                linestyle = 'solid',
                linewidth = 3.
               )
    i += 1
    ax1.plot(##average temperature coefficient
             i,
             mean(ensembleT),
             marker = 'o',
             markersize = 20.,
             linestyle = 'None',
             color = 'lightgrey'
            )
    ax1.errorbar(##standard deviation of temperature coefficients
                 i,
                 mean(ensembleT),
                 yerr=std(ensembleT),
                 linestyle="None",
                 elinewidth = 3.5,
                 capsize = 20,
                 capthick = 2,
                 color='black'
                )

    ax2.plot(##average bd coefficient
             i,
             mean(ensembleB),
             marker = 'o',
             markersize = 20.,
             linestyle = 'None',
             color = 'lightgrey',
            )
    ax2.errorbar(##standard deviation of bd coefficients
                 i,
                 mean(ensembleB),
                 yerr=std(ensembleB),
                 linestyle="None",
                 elinewidth = 3.5,
                 capsize = 20,
                 capthick = 2,
                 color='black'
                )

    ax3.plot(##average qbo coefficient
             i,
             mean(ensembleQ),
             marker = 'o',
             markersize = 20.,
             linestyle = 'None',
             color = 'lightgrey',
            )
    ax3.errorbar(##standard deviation of qbo coefficients
                 i,
                 mean(ensembleQ),
                 yerr=std(ensembleQ),
                 linestyle="None",
                 elinewidth = 3.5,
                 capsize = 20,
                 capthick = 2,
                 color='black'
                )
    i += 1
    ax1.axvline(##solid line separating individual model values from ensemble average values
                x=i,
                color = 'black',
                linestyle = 'solid',
                linewidth = 3.
               )
    ax2.axvline(##solid line separating individual model values from ensemble average values
                x=i,
                color = 'black',
                linestyle = 'solid',
                linewidth = 3.
               )
    ax3.axvline(##solid line separating individual model values from ensemble average values
                x=i,
                color = 'black',
                linestyle = 'solid',
                linewidth = 3.
               )
    i += 1
    ax1.plot(##era temperature coefficient
             i,
             era[0],
             marker = 'o',
             markersize = 20.,
             linestyle = 'None',
             color = 'lightgrey'
            )
    ax1.errorbar(##era temperature coefficient 95% confidence
                 i,
                 era[0],
                 yerr=eraSTD[0],
                 linestyle="None",
                 elinewidth = 3.5,
                 capsize = 20,
                 capthick = 2,
                 color='black'
                )

    ax2.plot(##era bd coefficient
             i,
             era[1],
             marker = 'o',
             markersize = 20.,
             linestyle = 'None',
             color = 'lightgrey',
            )
    ax2.errorbar(##era bd coefficient 95% confidence
                 i,
                 era[1],
                 yerr=eraSTD[1],
                 linestyle="None",
                 elinewidth = 3.5,
                 capsize = 20,
                 capthick = 2,
                 color='black'
                )

    ax3.plot(##era qbo coefficient
             i,
             era[2],
             marker = 'o',
             markersize = 20.,
             linestyle = 'None',
             color = 'lightgrey',
            )
    ax3.errorbar(##era qbo coefficients 95% confidence
                 i,
                 era[2],
                 yerr=eraSTD[2],
                 linestyle="None",
                 elinewidth = 3.5,
                 capsize = 20,
                 capthick = 2,
                 color='black'
                )
    i += 1
    ax1.axvline(##solid line separating era from merra
                x=i,
                color = 'black',
                linestyle = 'solid',
                linewidth = 3.
               )
    ax2.axvline(##solid line separating era from merra
                x=i,
                color = 'black',
                linestyle = 'solid',
                linewidth = 3.
               )
    ax3.axvline(##solid line separating era from merra
                x=i,
                color = 'black',
                linestyle = 'solid',
                linewidth = 3.
               )
    i += 1
    ax1.plot(##merra temperature coefficient
             i,
             merra[0],
             marker = 'o',
             markersize = 20.,
             linestyle = 'None',
             color = 'lightgrey'
            )
    ax1.errorbar(##merra temperature coefficient 95% confidence
                 i,
                 merra[0],
                 yerr=merraSTD[0],
                 linestyle="None",
                 elinewidth = 3.5,
                 capsize = 20,
                 capthick = 2,
                 color='black'
                )

    ax2.plot(##merra bd coefficient
             i,
             merra[1],
             marker = 'o',
             markersize = 20.,
             linestyle = 'None',
             color = 'lightgrey',
            )
    ax2.errorbar(##merra bd coefficient 95% confidence
                 i,
                 merra[1],
                 yerr=merraSTD[1],
                 linestyle="None",
                 elinewidth = 3.5,
                 capsize = 20,
                 capthick = 2,
                 color='black'
                )

    ax3.plot(##merra qbo coefficient
             i,
             merra[2],
             marker = 'o',
             markersize = 20.,
             linestyle = 'None',
             color = 'lightgrey',
            )
    ax3.errorbar(##merra qbo coefficients 95% confidence
                 i,
                 merra[2],
                 yerr=merraSTD[2],
                 linestyle="None",
                 elinewidth = 3.5,
                 capsize = 20,
                 capthick = 2,
                 color='black'
                )
    ax1.axhline(##solid line separating era from merra
                y = 0,
                color = 'black',
                linestyle = 'solid',
                linewidth = 3.
               )
    ax2.axhline(##solid line separating era from merra
                y = 0,
                color = 'black',
                linestyle = 'solid',
                linewidth = 3.
               )
    ax3.axhline(##solid line separating era from merra
                y = 0,
                color = 'black',
                linestyle = 'solid',
                linewidth = 3.
               )

    ax1.set_ylabel('$\mathregular{\\beta_{\Delta{T}}}$',fontsize = 24)
    ax2.set_ylabel('$\mathregular{\\beta_{BDC}}$',fontsize = 24)
    ax3.set_ylabel('$\mathregular{\\beta_{QBO}}$',fontsize = 24)

    ax1.xaxis.set_ticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
    ax2.xaxis.set_ticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
    ax3.xaxis.set_ticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])

    ax1.yaxis.set_ticks([-0.3,0.0,0.3,0.6])
    ax2.yaxis.set_ticks([-12,-6,0,6])
    ax3.yaxis.set_ticks([-0.1,0.0,0.1,0.2])

    ax1.set_ylim([-0.3,0.6])
    ax2.set_ylim([-12,6])
    ax3.set_ylim([-0.1,0.2])

    ax1.set_xlim([0,19])
    ax2.set_xlim([0,19])
    ax3.set_xlim([0,19])

    ax1.xaxis.grid(True,'major')
    ax2.xaxis.grid(True,'major')
    ax3.xaxis.grid(True,'major')

    ax1.yaxis.grid(True,'major')
    ax2.yaxis.grid(True,'major')
    ax3.yaxis.grid(True,'major')

    ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    ax3.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

    pModels.append('')
    pModels.append('Ensemble Average')
    pModels.append('')
    pModels.append('MLS/ERAI')
    pModels.append('')
    pModels.append('MLS/MERRA')
    ax3.set_xticklabels(pModels,rotation = 90)

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)


    plt.show()
def trendPlot(pModels,allModels,allModelsConf,allTemp,allTempConf,allbdc,allbdcConf,allqbo,allqboConf,run = True):
    #******************************************************************************************
    #Purpose: Plot the trends of each component of the regression
    #
    #Inputs:
    #  pModels: names of models analyzed
    #  allModels: simulated water vapor trend
    #  allModelsConf: simulated water vapor trend 95% confidence
    #  allTemp: simulated water vapor trend due only to tropospheric temperature
    #  allTempConf: simulated water vapor trend 95% confidence due only to tropospheric temperature
    #  allbdc: simulated water vapor trend due only to bdc
    #  allbdcConf: simulated water vapor trend 95% confidence due only to bdc
    #  allqbo: simulated water vapor trend due only to qbo
    #  allqboConf: simulated water vapor trend 95% confidence due only to qbo
    #
    #Outputs:
    #  plot image
    #
    #******************************************************************************************

    if not run:
        return##don't print stats

    from numpy import arange,mean,std
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    import matplotlib.ticker as mtick

    #--------------------set plot font settings-----------------------------------
    font = {
            'family' : 'Bitstream Vera Sans',
            'weight' : 'normal',
            'size'   : 24
           }

    mpl.rc('font', **font)
    #----------------------------------------------------------------------------

    ind = arange(1,len(pModels)+1)##plot index depending on model
    width = 0.2##bar width, and use to space bars

    fig = plt.figure(0)

    ax = fig.add_subplot(1,1,1)

    ax.bar(##water vapor 21st century trend
           ind - (2 * width),
           allModels,
           width,
           color = '#FFFFFF'
          )
    ax.errorbar(
                ind - (2 * width) + 0.1,##error-bar position
                allModels,
                yerr=allModelsConf,##standard deviation of century trended adjusted r2 values, repressents error-bars
                linestyle="None",
                elinewidth=3.5,##errorbar linewidth
                capsize = 15,##errorbar cap size
                capthick=2,##errorbar cap thickness
                color='black'##errorbar color
               )
    ax.bar(##water vapor 21st century trend due to troposheric temperature change
           ind - width,
           allTemp,
           width,
           color = '#FFD034'
          )
    ax.errorbar(
                ind - width + 0.1,##error-bar position
                allTemp,
                yerr=allTempConf,##standard deviation of century trended adjusted r2 values, repressents error-bars
                linestyle="None",
                elinewidth=3.5,##errorbar linewidth
                capsize = 15,##errorbar cap size
                capthick=2,##errorbar cap thickness
                color='black'##errorbar color
               )
    ax.bar(##water vapor 21st century trend due to bdc change
           ind,
           allbdc,
           width,
           color = '#FF4C3B'
          )
    ax.errorbar(
                ind + 0.1,##error-bar position
                allbdc,
                yerr=allbdcConf,##standard deviation of century trended adjusted r2 values, repressents error-bars
                linestyle="None",
                elinewidth=3.5,##errorbar linewidth
                capsize = 15,##errorbar cap size
                capthick=2,##errorbar cap thickness
                color='black'##errorbar color
               )
    ax.bar(##water vapor 21st century trend due to qbo change
           ind + width,
           allqbo,
           width,
           color = '#0072BB'
          )
    ax.errorbar(
                ind + width + 0.1,##error-bar position
                allqbo,
                yerr=allqboConf,##standard deviation of century trended adjusted r2 values, repressents error-bars
                linestyle="None",
                elinewidth=3.5,##errorbar linewidth
                capsize = 15,##errorbar cap size
                capthick=2,##errorbar cap thickness
                color='black'##errorbar color
               )
    ax.axvline(##solid line separating individual model values from ensemble average values
                x=ind[-1] + 1,
                color = 'black',
                linestyle = 'solid',
                linewidth = 3.
               )
    ax.bar(##average water vapor 21st century trend
           (ind[-1] + 2) - (2 * width),
           mean(allModels),
           width,
           color = '#FFFFFF'
          )
    ax.errorbar(
                (ind[-1] + 2) - (2 * width) + 0.1,##error-bar position
                mean(allModels),
                yerr = std(allModels),##standard deviation of century trended adjusted r2 values, repressents error-bars
                linestyle="None",
                elinewidth=3.5,##errorbar linewidth
                capsize = 15,##errorbar cap size
                capthick=2,##errorbar cap thickness
                color='black'##errorbar color
               )
    ax.bar(##average water vapor 21st century trend due to troposheric temperature change
           (ind[-1] + 2) - width,
           mean(allTemp),
           width,
           color = '#FFD034'
          )
    ax.errorbar(
                (ind[-1] + 2) - width + 0.1,##error-bar position
                mean(allTemp),
                yerr=mean(allTemp),##standard deviation of century trended adjusted r2 values, repressents error-bars
                linestyle="None",
                elinewidth=3.5,##errorbar linewidth
                capsize = 15,##errorbar cap size
                capthick=2,##errorbar cap thickness
                color='black'##errorbar color
               )
    ax.bar(##average water vapor 21st century trend due to bdc change
           ind[-1] + 2,
           mean(allbdc),
           width,
           color = '#FF4C3B'
          )
    ax.errorbar(
                (ind[-1] + 2) + 0.1,##error-bar position
                mean(allbdc),
                yerr=mean(allbdc),##standard deviation of century trended adjusted r2 values, repressents error-bars
                linestyle="None",
                elinewidth=3.5,##errorbar linewidth
                capsize = 15,##errorbar cap size
                capthick=2,##errorbar cap thickness
                color='black'##errorbar color
               )
    ax.bar(##aveagewater vapor 21st century trend due to qbo change
           (ind[-1] + 2) + width,
           mean(allqbo),
           width,
           color = '#0072BB'
          )
    ax.axhline(##solid zero line
                y = 0,
                color = 'black',
                linestyle = 'solid',
                linewidth = 3.
               )

    ax.set_ylabel("ppmv $\mathregular{year^{-1}}$",fontsize = 22)

    ax.xaxis.grid(True,'major')
    ax.yaxis.grid(True,'major')

    ax.xaxis.set_ticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])

    ax.set_xlim([0,15])

    pModels.append('')
    pModels.append('Consensus')
    ax.set_xticklabels(pModels,rotation=90)##use model names as tick labels for all three
                                            ##plots

    plt.show()

def decadeCenturyComparisonPlot(tempDecade,tempCentury,bdDecade,bdCentury,qboDecade,qboCentury,era,merra,run = True):
    #******************************************************************************************
    #Purpose: Plot the decadal median and standard deviation of each coefficient
    #
    #Inputs:
    #tempDecade:decadal temperature coefficients
    #tempCentury:century trended temperature coefficients
    #bdDecade: decadal bd coefficients
    #bdCentury: century trended bd coefficients
    #qboDecade: decadal qbo coefficients
    #qboCentury: century trended qbo coefficients
    #era: eraI coefficient as calculated by Dessler et al. (2014)
    #merra:merra coefficient as calculated by Dessler et al. (2014)
    #run: determine if plot should be created
    #
    #Outputs:
    #  create plot image
    #******************************************************************************************

    if not run:
        return##don't print stats

    from numpy import arange,linspace,mean,std
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    import matplotlib.ticker as mtick

    #--------------------set plot font settings-----------------------------------
    font = {
            'family' : 'Bitstream Vera Sans',
            'weight' : 'normal',
            'size'   : 24
           }

    mpl.rc('font', **font)
    #----------------------------------------------------------------------------

    oneToOneT = linspace(-0.2,0.5,num = 100)
    oneToOneB = linspace(-10.,5.,num = 100)
    oneToOneQ = linspace(-0.15,0.2,num = 100)

    tempVals = leastSquares1D(tempDecade,tempCentury)
    bdVals   = leastSquares1D(bdDecade,bdCentury)
    qboVals  = leastSquares1D(qboDecade,qboCentury)

    tempLine = leastSquaresSeries(tempVals[0],tempVals[1],oneToOneT)
    bdLine   = leastSquaresSeries(bdVals[0],bdVals[1],oneToOneB)
    qboLine  = leastSquaresSeries(qboVals[0],qboVals[1],oneToOneQ)

    eraTemp   = (tempVals[1] * era[0]) + tempVals[0]##location of eraI coefficients on one-to-one temperature
    merraTemp = (tempVals[1] * merra[0]) + tempVals[0]##location of merra coefficients on one-to-one temperature line

    eraBd   = (bdVals[1] * era[1]) + bdVals[0]##location of eraI coefficients on one-to-one bd
    merraBd = (bdVals[1] * merra[1]) + bdVals[0]##location of merra coefficients on one-to-one bd line

    eraQbo   = (qboVals[1] * era[2]) + qboVals[0]##location of eraI coefficients on one-to-one qbo
    merraQbo = (qboVals[1] * merra[2]) + qboVals[0]##location of merra coefficients on one-to-one qbo line

    fig = plt.figure(0)

    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = plt.subplot2grid((4,4), (2, 1),rowspan = 2,colspan = 2)

    ax1.plot(##one-to-one line for comparing temperature coefficients
             oneToOneT,
             tempLine,
             color = 'black',
             linewidth = 5
            )
    ax1.plot(##scatterplot of decadal and century temperature coefficients
             tempDecade,
             tempCentury,
             marker = 'o',
             markersize = 20.,
             linestyle = 'None',
             color = 'lightgrey'
            )
    ax1.plot(##eraI temperature coefficients
             era[0],
             eraTemp,
             marker = 's',
             markersize = 20.,
             markeredgecolor = 'grey',
             markeredgewidth = 5.,
             markerfacecolor = 'None',
             linestyle = 'None',
             color = 'lightgrey'
            )
    ax1.plot(##merra temperature coefficients
             merra[0],
             merraTemp,
             marker = 'd',
             markersize = 20.,
             markeredgecolor = 'grey',
             markeredgewidth = 5.,
             markerfacecolor = 'None',
             linestyle = 'None',
             color = 'lightgrey'
            )
    ax2.plot(##one-to-one line for comparing bd coefficients
             oneToOneB,
             bdLine,
             color = 'black',
             linewidth = 5
            )
    ax2.plot(##scatterplot of decadal and century bd coefficients
             bdDecade,
             bdCentury,
             marker = 'o',
             markersize = 20.,
             linestyle = 'None',
             color = 'lightgrey'
            )
    ax2.plot(##eraI bd coefficients
             era[1],
             eraBd,
             marker = 's',
             markersize = 20.,
             markeredgecolor = 'grey',
             markeredgewidth = 5.,
             markerfacecolor = 'None',
             linestyle = 'None',
             color = 'lightgrey'
            )
    ax2.plot(##merra bd coefficients
             merra[1],
             merraBd,
             marker = 'd',
             markersize = 20.,
             markeredgecolor = 'grey',
             markeredgewidth = 5.,
             markerfacecolor = 'None',
             linestyle = 'None',
             color = 'lightgrey'
            )
    ax3.plot(##one-to-one line for comparing qbo coefficients
             oneToOneQ,
             qboLine,
             color = 'black',
             linewidth = 5
            )
    ax3.plot(##scatterplot of decadal and century qbo coefficients
             qboDecade,
             qboCentury,
             marker = 'o',
             markersize = 20.,
             linestyle = 'None',
             color = 'lightgrey'
            )
    ax3.plot(##eraI qbo coefficients
             era[2],
             eraQbo,
             marker = 's',
             markersize = 20.,
             markeredgecolor = 'grey',
             markeredgewidth = 5.,
             markerfacecolor = 'None',
             linestyle = 'None',
             color = 'lightgrey'
            )
    ax3.plot(##merra qbo coefficients
             merra[2],
             merraQbo,
             marker = 'd',
             markersize = 20.,
             markeredgecolor = 'grey',
             markeredgewidth = 5.,
             markerfacecolor = 'None',
             linestyle = 'None',
             color = 'lightgrey'
            )

    ax1.set_ylabel('Century $\mathregular{\\beta_{\Delta{T}}}$',fontsize = 32)
    ax2.set_ylabel('Century $\mathregular{\\beta_{BDC}}$',fontsize = 32)
    ax3.set_ylabel('Century $\mathregular{\\beta_{QBO}}$',fontsize = 32)

    ax1.set_xlabel('Median Decadal $\mathregular{\\beta_{\Delta{T}}}$',fontsize = 32)
    ax2.set_xlabel('Median Decadal $\mathregular{\\beta_{BDC}}$',fontsize = 32)
    ax3.set_xlabel('Median Decadal $\mathregular{\\beta_{QBO}}$',fontsize = 32)

    ax1.yaxis.set_ticks([0,0.3,0.6,0.9])
    ax2.yaxis.set_ticks([-14,-9,-4,1,6])
    ax3.yaxis.set_ticks([-0.04,0.01,0.06,0.11,0.16])

    ax1.set_xlim([-0.2,0.4])
    ax2.set_xlim([-10.,4.])
    ax3.set_xlim([-0.02,0.14])
    ax3.set_xlim([-0.02,0.14])

    ax1.xaxis.grid(True,'major')
    ax2.xaxis.grid(True,'major')
    ax3.xaxis.grid(True,'major')

    ax1.yaxis.grid(True,'major')
    ax2.yaxis.grid(True,'major')
    ax3.yaxis.grid(True,'major')

    ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))


    plt.show()

def printStatsCentury(model,regression,rType = 'Trended Century Regression',run = True):
    #******************************************************************************************
    #Purpose: Print statistics for the century regressions (trended, detrended, or either
    #         standardized
    #
    #Inputs:
    #  model: list of models analyzed
    #  regression: multiregress pyObject
    #  rType: type of stats to display
    #  run: determine if to print stats to screen
    #
    #Outputs:
    #  print stats to screen
    #
    #******************************************************************************************

    if not run:
        return##don't print stats

    print('-----',model,'-----')
    print('')
    print('\t',rType)
    print('')
    print('\ttemperature coefficient                %0.4f' % regression.coefficents[1],'ppmv/K')
    print('\tbrewer-dobson coefficient              %0.4f' % regression.coefficents[2],'ppmv/(K/Day)')
    print('\tquasi-biennial oscillation coefficient %0.4f' % regression.coefficents[3],'ppmv')
    print('')
    print('\tadjusted R^2                           %0.4f' % (regression.adjR2 * 100.),'%')
    print('')

def printStatsDecade(model,temp,bd,qbo,r2,rType = 'Decadal Regression',run = True):
    #******************************************************************************************
    #Purpose: Print statistics for the decadal regressions (trended, detrended, or either
    #         standardized
    #
    #Inputs:
    #model: list of models analyzed
    #temp: temperature values
    #bd: bd values
    #qbo: qbo values
    #r2: adjusted r2 values
    #rType: type of stats to display
    #run: determine if to print stats to screen
    #Outputs:
    #  print stats to screen
    #
    #******************************************************************************************

    if not run:
        return##don't print stats

    from numpy import max,min

    print('-----',model,'-----')
    print('')
    print('\t',rType)
    print('')
    print('\ttemperature Coefficient Range                %0.4f' % min(temp),'to %0.4f' % max(temp))
    print('')
    print('\tbrewer-dobson Coefficient Range              %0.4f' % min(bd),'to %0.4f' % max(bd))
    print('')
    print('\tquasi-biennial oscillation Coefficient Range %0.4f' % min(qbo),'to %0.4f' % max(qbo))
    print('')
    print('\tadjusted R^2 Range                           %0.4f' % min(r2),'to %0.4f' % max(r2))
    print('')

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
    from pandas import Series,concat
    from scipy.signal import detrend
    ##models to loop over
    model = [
             #'MRI',
             #'WACCM',
             #'CMAM',
             #'CCSRNIES',
             #'LMDZrepro',
             #'GEOSCCM',
             #'CCSRNIES-MIROC3.2',
             'CNRM-CM5-3',
             #'NIWA-UKCA',
             #'CMAM_CCMI',
             #'GEOSCCM_CCMI',
             #'MRI-ESM1r1'
            ]

    model = sorted(model)##sort the models being evaluated alphebetically

    ###boolian directors
    teraD = True##true if we want decadal analysis
    teraA = True##true if we want century analysis

    seg = 10##length of a decade in years

    num_years = 98##number of years
    year0     = 2000##beginning year
    yearF     = 2097##ending year

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

    adjR111 = OrderedDict([(cv,0) for cv in model])##full 21st century adjusted r2 values for the trended anomolies for each model
    adjR222 = OrderedDict([(cv,0) for cv in model])##full 21st century adjusted r2 values for the detrended anomolies for each model

    tempCoeffA  = OrderedDict([(cv,0) for cv in model])##full 21st century temperature coefficients for the trended anomolies for each model
    tempSCoeffA = OrderedDict([(cv,0) for cv in model])##full 21st century temperature coefficients for the detrended anomolies for each model

    tempConfA  = OrderedDict([(cv,0) for cv in model])##full 21st century temperature coefficient 95% confidence of the trended anomolies
    tempSConfA = OrderedDict([(cv,0) for cv in model])##full 21st century temperature coefficient 95% confidence of the detrended anomolies

    bdCoeffA  = OrderedDict([(cv,0) for cv in model])##full 21st century brewer-dobson coefficients for the trended anomolies for each model
    bdSCoeffA = OrderedDict([(cv,0) for cv in model])##full 21st century brewer-dobson coefficients for the detrended anomolies for each model

    bdConfA  = OrderedDict([(cv,0) for cv in model])##full 21st century brewer-dobson coefficient 95% confidence of the trended anomolies
    bdSConfA = OrderedDict([(cv,0) for cv in model])##full 21st century brewer-dobson coefficient 95% confidence of the detrended anomolies

    qboCoeffA  = OrderedDict([(cv,0) for cv in model])##full 21st century quasi-biennial oscillation coefficients for the trended anomolies for each model
    qboSCoeffA = OrderedDict([(cv,0) for cv in model])##full 21st century quasi-biennial oscillation coefficients for the detrended anomolies for each model

    qboConfA  = OrderedDict([(cv,0) for cv in model])##full 21st century quasi-biennial oscillationcoefficient 95% confidence of the trended anomolies
    qboSConfA = OrderedDict([(cv,0) for cv in model])##full 21st century quasi-biennial oscillation coefficient 95% confidence of the detrended anomolies

    modelTrendSlope = OrderedDict([(cv,0) for cv in model])##simulated water vapor trends for each model
    modelTrendConf  = OrderedDict([(cv,0) for cv in model])##simulated water vapor trends for each model 95% confidence

    tempTrendSlope = OrderedDict([(cv,0) for cv in model])##simulated water vapor trends resulting from tropospheric temperature for each model
    tempTrendConf  = OrderedDict([(cv,0) for cv in model])##simulated water vapor trends resulting from troposheric temperature for each model 95% confidence

    bdTrendSlope = OrderedDict([(cv,0) for cv in model])##simulated water vapor trends resulting from bd for each model
    bdTrendConf  = OrderedDict([(cv,0) for cv in model])##simulated water vapor trends resulting from bd for each model 95% confidence

    qboTrendSlope = OrderedDict([(cv,0) for cv in model])##simulated water vapor trends resulting from qbo for each model
    qboTrendConf  = OrderedDict([(cv,0) for cv in model])##simulated water vapor trends resulting from qbo for each model 95% confidence

    tempCoeffD = OrderedDict([(cv,[]) for cv in model])##decadal temperature coefficients for the monthly anomolies for each model
    bdCoeffD   = OrderedDict([(cv,[]) for cv in model])##decadal bd coefficients for the monthly anomolies for each model
    qboCoeffD  = OrderedDict([(cv,[]) for cv in model])##decadal qbo coefficients for the monthly anomolies for each model
    adjR2D     = OrderedDict([(cv,[]) for cv in model])##decadal adjusted r2 for the monthly anomolies for each model

    for cv in model:
        ##***********************************keep in main function*************************************
        ##determine if simulation is in CCMVal-2 or CCMI-1 experiments
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
        #continue
        if(
           cv != 'GEOSCCM'
       and experiment == 'CCMVal-2'
          ):
            uaD  = Dataset(finUa,'r')##zonal wind netcdf4 dataset
            taD  = Dataset(finT,'r')##temperature netcdf4 dataset
            h2oD = Dataset(finW,'r')##water vapor netcdf4 dataset
            lD = Dataset(finL,'r')##longwave netcdf4 dataset
            sD = Dataset(finS,'r')##shortwave netcdf4 dataset
        elif(cv == 'NIWA-UKCA'):
            aD  = Dataset(finA,'r')##all variable netcdf4 dataset
            aDD = Dataset(finAA,'r')##water vapor variable netcdf4 dataset
        else:
            #aD = open_dataset(finA)##all variable netcdf4 dataset
            aD = Dataset(finA,'r')##all variable netcdf4 dataset

        ###Read Variable for hybrid pressure adjustment
        if(
           cv == 'CCSRNIES-MIROC3.2'
        or cv == 'GEOSCCM_CCMI'
        or cv == 'CMAM_CCMI'
        or cv == 'MRI-ESM1r1'
          ):
            ua = aD.variables['ua'][:,:,:,:]
            #print(ua)
            #return

        ##Pressure Variables
        if(cv == 'UMUKCA-UCAM'):
            pp = 'p'
        elif(cv == 'GEOSCCM'):
            pp = 'lev'
        elif(
             cv == 'CCSRNIES-MIROC3.2'
          or cv == 'CMAM_CCMI'
            ):
            latP,level,aP,bP,pS1,pS2  = 'lat','lev','ap','b','p0','ps'
            hP   = hybridPressure(aP,bP,pS2,aD,ua,level,latP)
        elif(
             cv == 'GEOSCCM_CCMI'
          or cv == 'MRI-ESM1r1'
            ):
            latP,level,aP,bP,pS1,pS2  = 'lat','lev','a','b','p0','ps'
            hP   = hybridPressure(aP,bP,pS2,aD,ua,level,latP,pS1,True)
        elif(cv == 'NIWA-UKCA'):
            pp = 'plev'
            ppp = 'hybridP'
        else:
            pp = 'plev'

        ##pressure arrays
        if(cv == 'GEOSCCM'\
           or cv == 'CNRM-CM5-3'):
            pLev = aD.variables[pp][:]
        elif(cv == 'NIWA-UKCA'):
            hpLev = aDD.variables[ppp][:]
            pLev = aD.variables[pp][:]
        elif(cv == 'CCSRNIES-MIROC3.2'\
             or cv == 'CMAM_CCMI'\
             or cv == 'GEOSCCM_CCMI'\
             or cv == 'MRI-ESM1r1'):
            pLev = hP
        else:
            pLev = uaD.variables[pp][:]

        ##latitude arrays
        if(
           cv != 'GEOSCCM'
       and experiment == 'CCMVal-2'
          ):
            latU = uaD.variables['lat'][:]##latitude
            latRC = cos(latU[(latU<=max(lR))&(latU>=min(lR))]*(pi/180.))##latitude masked to tropics and weighted by cosine of latitude
        else:
            latU = aD.variables['lat'][:]##latitude
            latRc = cos(latU[(latU<=max(lR))&(latU>=min(lR))]*(pi/180.))##latitude masked to tropics and weighted by cosine of latitude

        ##longitude array
        if(
           cv != 'GEOSCCM'
       and experiment == 'CCMVal-2'
          ):
            lon  = taD.variables['lon'][:]##longitude
        elif(cv == 'NIWA-UKCA'):
            lon = aD.variables['lon'][:]##longitude
        else:
            lon = aD.variables['lon'][:]##longitude

        ##format time-stamps into dates (and boolean arrays for masking)
        if(
           cv == 'GEOSCCM'
        or cv == 'CCSRNIES-MIROC3.2'
        or cv == 'CNRM-CM5-3'
        or cv == 'NIWA-UKCA'
        or cv == 'CMAM_CCMI'
        or cv == 'GEOSCCM_CCMI'
        or cv == 'MRI-ESM1r1'
          ):
            time = aD.variables['time']##time netcdf4 variable
            tim  = time[:]##time array
        else:
            time  = uaD.variables['time']##time netcdf4 variable
            tim  = time[:]##time array

        timI,total_years,check22 = dates(time,year0,yearF)##use num2date to calculate dates in dataset

        ##adjust time boolean mask for same array size for each simulation
        if(
           cv == 'WACCM'
        or cv == 'CMAM'
        or cv == 'CMAM_CCMI'
          ):
            timI2 = timI[timI]
            timI2[-1] = False
            timI[timI] = timI2
        elif(cv == 'UMUKCA-UCAM'):
            timI2 = timI[timI]
            timI2[timI2.size-11:timI2.size] = False
            timI[timI] = timI2
        elif(cv == 'NIWA-UKCA'):
            timI2 = timI[timI]
            timI2[timI2.size-17:timI2.size] = False
            timI[timI] = timI2

        ##initialize all arrays for each simulation
        if(
           cv != 'GEOSCCM'
       and experiment == 'CCMVal-2'
          ):
            h2o = h2oD.variables['H2O'][timI,:,:,:]##water vapor variable
            h2o = ma.average(h2o,axis = 3)##create a zonal average (average along the longitude dimension)
            h2o = netConfig(lR,(83,78),h2o,latU,pLev)*10**6##convert to ppmv and average meridionally (along the longitude dimension)
            h2o = h2o.reshape(num_years,12)##reshape from a 1D array of the same size as the time dimension to a 2D array of size (number of years, number of months) i.e. (100,12)

            ta = taD.variables['ta'][timI,:,:,:]##temperature variable
            ta = ma.average(ta,axis = 3)##create a zonal average (average along the longitude dimension)
            ta = netConfig(lR,(501,499),ta,latU,pLev)##average meridionally (along the longitude dimension)
            ta = ta.reshape(num_years,12)##reshape from a 1D array of the same size as the time dimension to a 2D array of size (number of years, number of months) i.e. (100,12)

            tntlw = lD.variables['tntlw'][timI,:,:,:]##longwave heat flux variable
            tntlw = ma.average(tntlw,axis = 3)##create a zonal average (average along the longitude dimension)
            tntlw = netConfig(lR,(79,81),tntlw,latU,pLev)##average meridionally (along the longitude dimension)
            tntlw = tntlw.reshape(num_years,12)##reshape from a 1D array of the same size as the time dimension to a 2D array of size (number of years, number of months) i.e. (100,12)

            tntsw = sD.variables['tntsw'][timI,:,:,:]##shortwave heat flux variable
            tntsw = ma.average(tntsw,axis = 3)##create a zonal average (average along the longitude dimension)
            tntsw = netConfig(lR,(79,81),tntsw,latU,pLev)##average meridionally (along the longitude dimension)
            tntsw = tntsw.reshape(num_years,12)##reshape from a 1D array of the same size as the time dimension to a 2D array of size (number of years, number of months) i.e. (100,12)

            ua = uaD.variables['ua'][timI,:,:,:]##zonal wind variable
            ua = ma.average(ua,axis = 3)##create a zonal average (average along the longitude dimension)
            ua = netConfig((-2,2),(48,52),ua,latU,pLev)##average meridionally (along the longitude dimension)
            ua = monAmoms(ua)##seasonalize 50-hPa zonal wind
            ua = ua/ua.std()##standardize zonal wind variable, for qbo calculation
            ua = ua.reshape(num_years,12)##reshape from a 1D array of the same size as the time dimension to a 2D array of size (number of years, number of months) i.e. (100,12)

        elif(
             cv == 'GEOSCCM'
         and experiment == 'CCMVal-2'
            ):
            h2o = aD.variables['H2O'][timI,:,:,:]##water vapor variable
            Maskedh2o = (h2o > 10000)
            h2o = ma.array(h2o,mask = Maskedh2o)
            h2o = ma.average(h2o,axis = 3)##create a zonal average (average along the longitude dimension)
            h2o = netConfig(lR,(83,78),h2o,latU,pLev)*10**6##convert to ppmv and average meridionally (along the longitude dimension)
            h2o = h2o.reshape(num_years,12)##reshape from a 1D array of the same size as the time dimension to a 2D array of size (number of years, number of months) i.e. (100,12)

            ta = aD.variables['ta'][timI,:,:,:]##temperature variable
            Maskedta = (ta > 10000)
            ta = ma.array(ta,mask = Maskedta)
            ta = ma.average(ta,axis = 3)##create a zonal average (average along the longitude dimension)
            ta = netConfig(lR,(501,499),ta,latU,pLev)##average meridionally (along the longitude dimension)
            ta = ta.reshape(num_years,12)##reshape from a 1D array of the same size as the time dimension to a 2D array of size (number of years, number of months) i.e. (100,12)

            tntlw = aD.variables['tntlw'][timI,:,:,:]##longwave heat flux variable
            Maskedtntlw = (tntlw > 10000)
            tntlw = ma.array(tntlw,mask = Maskedtntlw)
            tntlw = ma.average(tntlw,axis = 3)##create a zonal average (average along the longitude dimension)
            tntlw = netConfig(lR,(79,81),tntlw,latU,pLev)##average meridionally (along the longitude dimension)
            tntlw = tntlw.reshape(num_years,12)##reshape from a 1D array of the same size as the time dimension to a 2D array of size (number of years, number of months) i.e. (100,12)

            tntsw = aD.variables['tntsw'][timI,:,:,:]##shortwave heat flux variable
            Maskedtntsw = (tntsw > 10000)
            tntsw = ma.array(tntsw,mask = Maskedtntsw)
            tntsw = ma.average(tntsw,axis = 3)##create a zonal average (average along the longitude dimension)
            tntsw = netConfig(lR,(79,81),tntsw,latU,pLev)##average meridionally (along the longitude dimension)
            tntsw = tntsw.reshape(num_years,12)##reshape from a 1D array of the same size as the time dimension to a 2D array of size (number of years, number of months) i.e. (100,12)

            ua = aD.variables['ua'][timI,:,:,:]##zonal wind variable
            Maskedua = (ua > 10000)
            ua = ma.array(ua,mask = Maskedua)
            ua = ma.average(ua,axis = 3)##create a zonal average (average along the longitude dimension)
            ua = netConfig((-2,2),(48,52),ua,latU,pLev)##average meridionally (along the longitude dimension)
            ua = monAmoms(ua)##seasonalize 50-hPa zonal wind
            ua = ua/ua.std()##standardize zonal wind variable, for qbo calculation
            ua = ua.reshape(num_years,12)##reshape from a 1D array of the same size as the time dimension to a 2D array of size (number of years, number of months) i.e. (100,12)

        elif(
             cv == 'CCSRNIES-MIROC3.2'
          or cv == 'CMAM_CCMI'
          or cv == 'GEOSCCM_CCMI'
            ):
            h2o = aD.variables['vmrh2o'][timI,:,:,:]##water vapor variable
            h2o = ma.average(h2o,axis = 3)##create a zonal average (average along the longitude dimension)
            h2o = netConfig(lR,(86,76),h2o,latU,hP)*10**6##convert to ppmv and average meridionally (along the longitude dimension)
            h2o = h2o.reshape(num_years,12)##reshape from a 1D array of the same size as the time dimension to a 2D array of size (number of years, number of months) i.e. (100,12)

            ta = aD.variables['ta'][timI,:,:,:]##temperature variable
            ta = ma.average(ta,axis = 3)##create a zonal average (average along the longitude dimension)
            ta = netConfig(lR,(500,530),ta,latU,hP)##average meridionally (along the longitude dimension)
            ta = ta.reshape(num_years,12)##reshape from a 1D array of the same size as the time dimension to a 2D array of size (number of years, number of months) i.e. (100,12)

            tntlw = aD.variables['tntlw'][timI,:,:,:]##longwave heat flux variable
            tntlw = ma.average(tntlw,axis = 3)##create a zonal average (average along the longitude dimension)
            tntlw = netConfig(lR,(86,76),tntlw,latU,hP)##average meridionally (along the longitude dimension)
            tntlw = tntlw.reshape(num_years,12)##reshape from a 1D array of the same size as the time dimension to a 2D array of size (number of years, number of months) i.e. (100,12)

            tntsw = aD.variables['tntsw'][timI,:,:,:]##shortwave heat flux variable
            tntsw = ma.average(tntsw,axis = 3)##create a zonal average (average along the longitude dimension)
            tntsw = netConfig(lR,(86,76),tntsw,latU,hP)##average meridionally (along the longitude dimension)
            tntsw = tntsw.reshape(num_years,12)##reshape from a 1D array of the same size as the time dimension to a 2D array of size (number of years, number of months) i.e. (100,12)

            ua = aD.variables['ua'][timI,:,:,:]##zonal wind variable
            ua = ma.average(ua,axis = 3)##create a zonal average (average along the longitude dimension)
            ua = netConfig((-2,2),(45,54),ua,latU,hP)##average meridionally (along the longitude dimension)
            ua = monAmoms(ua)##seasonalize 50-hPa zonal wind
            ua = ua/ua.std()##standardize zonal wind variable, for qbo calculation
            ua = ua.reshape(num_years,12)##reshape from a 1D array of the same size as the time dimension to a 2D array of size (number of years, number of months) i.e. (100,12)

        elif(cv == 'MRI-ESM1r1'):
            h2o = aD.variables['vmrh2o'][timI,:,:,:]##water vapor variable
            print(hP[(hP >= 76) & (hP <= 86)])
            h2o = ma.average(h2o,axis = 3)##create a zonal average (average along the longitude dimension)
            h2o = netConfig(lR,(86,76),h2o,latU,hP)*10**6##convert to ppmv and average meridionally (along the longitude dimension)
            h2o = h2o.reshape(num_years,12)##reshape from a 1D array of the same size as the time dimension to a 2D array of size (number of years, number of months) i.e. (100,12)

            ta = aD.variables['ta'][timI,:,:,:]##temperature variable
            ta = ma.average(ta,axis = 3)##create a zonal average (average along the longitude dimension)
            ta = netConfig(lR,(500,530),ta,latU,hP)##average meridionally (along the longitude dimension)
            ta = ta.reshape(num_years,12)##reshape from a 1D array of the same size as the time dimension to a 2D array of size (number of years, number of months) i.e. (100,12)

            tntlw = aD.variables['tntlw'][timI,:,:,:]##longwave heat flux variable
            tntlw = ma.average(tntlw,axis = 3)##create a zonal average (average along the longitude dimension)
            tntlw = netConfig(lR,(86,76),tntlw,latU,hP)##average meridionally (along the longitude dimension)
            tntlw = tntlw.reshape(num_years,12)##reshape from a 1D array of the same size as the time dimension to a 2D array of size (number of years, number of months) i.e. (100,12)

            tntsw = aD.variables['tntsw'][timI,:,:,:]##shortwave heat flux variable
            tntsw = ma.average(tntsw,axis = 3)##create a zonal average (average along the longitude dimension)
            tntsw = netConfig(lR,(86,76),tntsw,latU,hP)##average meridionally (along the longitude dimension)
            tntsw = tntsw.reshape(num_years,12)##reshape from a 1D array of the same size as the time dimension to a 2D array of size (number of years, number of months) i.e. (100,12)

            ua = aD.variables['ua'][timI,:,:,:]##zonal wind variable
            ua = ma.average(ua,axis = 3)##create a zonal average (average along the longitude dimension)
            ua = netConfig((-2,2),(47,52),ua,latU,hP)##average meridionally (along the longitude dimension)
            ua = monAmoms(ua)##seasonalize 50-hPa zonal wind
            ua = ua/ua.std()##standardize zonal wind variable, for qbo calculation
            ua = ua.reshape(num_years,12)##reshape from a 1D array of the same size as the time dimension to a 2D array of size (number of years, number of months) i.e. (100,12)

        elif(cv == 'CNRM-CM5-3'):
            h2o = aD.variables['vmrh2o'][timI,:,:,:]##water vapor variable
            h2o = ma.average(h2o,axis = 3)##create a zonal average (average along the longitude dimension)
            h2o = netConfig(lR,(83,76),h2o,latU,pLev/100.) * 10**6##convert to ppmv and average meridionally (along the longitude dimension)
            h2o = h2o.reshape(num_years,12)##reshape from a 1D array of the same size as the time dimension to a 2D array of size (number of years, number of months) i.e. (100,12)
            ta = aD.variables['ta'][timI,:,:,:]##temperature variable
            ta = ma.average(ta,axis = 3)##create a zonal average (average along the longitude dimension)
            ta = netConfig(lR,(500,510),ta,latU,pLev/100.)##average meridionally (along the longitude dimension)
            ta = ta.reshape(num_years,12)##reshape from a 1D array of the same size as the time dimension to a 2D array of size (number of years, number of months) i.e. (100,12)

            tntlw = aD.variables['tntlw'][timI,:,:,:]##longwave heat flux variable
            tntlw = ma.average(tntlw,axis = 3)##create a zonal average (average along the longitude dimension)
            tntlw = netConfig(lR,(86,76),tntlw,latU,pLev/100.)##average meridionally (along the longitude dimension)
            tntlw = tntlw.reshape(num_years,12)##reshape from a 1D array of the same size as the time dimension to a 2D array of size (number of years, number of months) i.e. (100,12)
            tntsw = aD.variables['tntsw'][timI,:,:,:]##shortwave heat flux variable
            tntsw = ma.average(tntsw,axis = 3)##create a zonal average (average along the longitude dimension)
            tntsw = netConfig(lR,(86,76),tntsw,latU,pLev/100.)##average meridionally (along the longitude dimension)
            tntsw = tntsw.reshape(num_years,12)##reshape from a 1D array of the same size as the time dimension to a 2D array of size (number of years, number of months) i.e. (100,12)
            ua = aD.variables['ua'][timI,:,:,:]##zonal wind variable
            ua = ma.average(ua,axis = 3)##create a zonal average (average along the longitude dimension)
            ua = netConfig((-2,2),(45,52),ua,latU,pLev/100.,weight = False)##average meridionally (along the longitude dimension)
            ua = monAmoms(ua)##seasonalize 50-hPa zonal wind
            ua = ua/ua.std()##standardize zonal wind variable, for qbo calculation
            ua = ua.reshape(num_years,12)##reshape from a 1D array of the same size as the time dimension to a 2D array of size (number of years, number of months) i.e. (100,12)

        elif(cv == 'NIWA-UKCA'):
            h2o = aD.variables['hus'][timI,:,:,:]##water vapor variable
            h2o = ma.average(h2o,axis = 3)##create a zonal average (average along the longitude dimension)
            h2o = netConfig(lR,(83,76),h2o,latU,hpLev)*10**6##convert to ppmv and average meridionally (along the longitude dimension)
            h2o = h2o.reshape(num_years,12)##reshape from a 1D array of the same size as the time dimension to a 2D array of size (number of years, number of months) i.e. (100,12)

            ta = aD.variables['ta'][timI,:,:,:]##temperature variable
            ta = ma.average(ta,axis = 3)##create a zonal average (average along the longitude dimension)
            ta = netConfig(lR,(500,510),ta,latU,pLev/100.)##average meridionally (along the longitude dimension)
            ta = ta.reshape(num_years,12)##reshape from a 1D array of the same size as the time dimension to a 2D array of size (number of years, number of months) i.e. (100,12)

            tntlw = aD.variables['tntlw'][timI,:,:,:]##longwave heat flux variable
            tntlw = ma.average(tntlw,axis = 3)##create a zonal average (average along the longitude dimension)
            tntlw = netConfig(lR,(83,76),tntlw,latU,hpLev)##average meridionally (along the longitude dimension)
            tntlw = tntlw.reshape(num_years,12)##reshape from a 1D array of the same size as the time dimension to a 2D array of size (number of years, number of months) i.e. (100,12)

            tntsw = aD.variables['tntsw'][timI,:,:,:]##shortwave heat flux variable
            tntsw = ma.average(tntsw,axis = 3)##create a zonal average (average along the longitude dimension)
            tntsw = netConfig(lR,(83,76),tntsw,latU,hpLev)##average meridionally (along the longitude dimension)
            tntsw = tntsw.reshape(num_years,12)##reshape from a 1D array of the same size as the time dimension to a 2D array of size (number of years, number of months) i.e. (100,12)

            ua = aD.variables['ua'][timI,:,:,:]##zonal wind variable
            ua = ma.average(ua,axis = 3)##create a zonal average (average along the longitude dimension)
            ua = netConfig((-2,2),(45,52),ua,latU,pLev/100.)##average meridionally (along the longitude dimension)
            ua = monAmoms(ua)##seasonalize 50-hPa zonal wind
            ua = ua/ua.std()##standardize zonal wind variable, for qbo calculation
            ua = ua.reshape(num_years,12)##reshape from a 1D array of the same size as the time dimension to a 2D array of size (number of years, number of months) i.e. (100,12)

        ##century analysis
        if(teraA):
            h2oA  = annualAnoms(h2o)##water vapor trended annual anomolies
            #h2oAD = detrend(h2oA)##water vapor detrended annual anomolies
            h2oAD = detrendAnom(10,h2oA)##water vapor detrended annual anomolies
            h2oAD = Series(h2oAD)##pandas series
            h2oA  = Series(h2oA)##pandas series

            taA  = annualAnoms(ta)##temperature trended annual anomolies
            taAD = detrendAnom(10,taA)##temperature detrended annual anomolies
            #taAD = detrend(taA)##temperature detrended annual anomolies
            taAD = Series(taAD)##pandas series
            taA  = Series(taA)##pandas series

            tntlwA = annualAnoms(tntlw)##longwave heat flux trended annual anomolies
            tntswA = annualAnoms(tntsw)##shortwave heat flux trended annual anomolies
            netRA = (tntlwA + tntswA) * 86400. * ((1000./80.)**(2./7.))
            #print(netRA_test - netRA)
            #exit()
            #netRAD = detrend(netRA)##net heatflux detrended annual anomolies
            netRAD = detrendAnom(10,netRA)##net heatflux detrended annual anomolies
            netRAD = Series(netRAD)##pandas series
            netRA = Series(netRA)##pandas series

            uaAA   = annualAnoms(ua)##zonal wind trended annual anomolies
            #uaAD   = detrend(uaAA)##zonal wind detrended annual anomolies
            uaAD   = detrendAnom(10,uaAA)##zonal wind detrended annual anomolies
            uaAD   = Series(uaAD)##pandas series
            uaAA   = Series(uaAA)##pandas series

            dSet  = concat(##pandas dataframe for trended anomolies
                           {
                            'h2o'     :h2oA,
                            'temp'    :taA,
                            'heating' :netRA,
                            'qbo'     :uaAA
                           },
                           axis = 1
                          )
            dSetD = concat(##pandas dataframe for the detrended anomolies
                           {
                            'h2o'     : h2oAD,
                            'temp'    : taAD,
                            'heating' : netRAD,
                            'qbo'     : uaAD
                           },
                           axis = 1
                          )

            #xSeg   = zeros((dSet.index.size,varNums))##2D array used for trended century regressions
            #xSegS  = zeros((dSet.index.size,varNums))##2D array used for the standardized trended century regressions
            #xSegD  = zeros((dSet.index.size,varNums))##2D array used for the detrended century regressions
            #xSegSD = zeros((dSet.index.size,varNums))##2D array used for the standardized detrended century regressions

            #xSeg[:,0] = ones(dSet.index.values.size)##first column of array is ones for regression calculations
            #xSeg[:,1] = dSet.temp.values##trended temperature anomolies
            #xSeg[:,2] = dSet.heating.values##trended net radiative heating anomolies
            #xSeg[:,3] = dSet.qbo.values##trended qbo anomolies

            #xSegS[:,0] = ones(dSet.index.values.size)##first column of array is ones for regression calculations
            #xSegS[:,1] = dSet.temp.values/dSet.temp.values.std()##trended standardized temperature anomolies
            #xSegS[:,2] = dSet.heating.values/dSet.heating.values.std()##trended standardized net radiative heating anomolies
            #xSegS[:,3] = dSet.qbo.values##trended standardized qbo anomolies

            #xSegD[:,0] = ones(dSet.index.values.size)##first column of array is ones for regression calculations
            #xSegD[:,1] = dSetD.temp.values##detrended temperature anomolies
            #xSegD[:,2] = dSetD.heating.values##detrended net radiative heating anomolies
            #xSegD[:,3] = dSetD.qbo.values##detrended qbo anomolies
            ##xSegD[:,2] = dSetD.qbo.values##detrended qbo anomolies

            #xSegSD[:,0] = ones(dSet.index.values.size)##first column of arrayh is ones for regression calculation
            #xSegSD[:,1] = dSetD.temp.values/dSetD.temp.values.std()##detrended standardized temperature anomolies
            #xSegSD[:,2] = dSetD.heating.values/dSetD.heating.values.std()##detrended standardized net radiative heating anomolies
            #xSegSD[:,3] = dSetD.qbo.values/dSetD.qbo.std()##detrended standardized qbo anomolies

            #regressSEG   = multiregress(##set a multi-linear regression python object for the trended regressions
            #                            dSet.h2o.values,##set a multi-linear regression python object inputs: response variable-water vapor,
            #                            xSeg,##predictor variables-xSeg
            #                            dSet.index.values.size,##length of the regression-dSet
            #                            varNums,##index.values.size,number of variables-varNums
            #                            True
            #                           )
            #regressSEGS  = multiregress(##set a multi-linear regression python object for the standardized trended regressions
            #                            dSet.h2o.values/dSet.h2o.values.std(),
            #                            xSegS,
            #                            dSet.index.values.size,
            #                            varNums,
            #                            True
            #                           )
            #regressSEGD  = multiregress(##set a multi-linear regression python object for the detrended regressions
            #                            dSetD.h2o.values,
            #                            xSegD,
            #                            dSetD.index.values.size,
            #                            varNums,
            #                            #varNums - 1,
            #                            True
            #                           )
            #regressSEGSD = multiregress(##set a multi-linear regression python object for the standardized detrended regressions
            #                            dSetD.h2o.values/dSetD.h2o.values.std(),
            #                            xSegSD,
            #                            dSetD.index.values.size,
            #                            varNums,
            #                            True
            #                           )

            #y1    = regressSEG.y[0]##initial water-vapor value from simulation
            #Yhatc = regressSEG.Yhat[0]##initial water-vapor value estimated from the multi-linar regression

            #Yhat1 = regressSEG.coefficents[0] + regressSEG.coefficents[1]*xSeg[:,1]##predictor (for plotting purposes)
            #Yhat2 = regressSEG.coefficents[0] + regressSEG.coefficents[2]*xSeg[:,2]##predictor (for plotting purposes)
            #Yhat3 = regressSEG.coefficents[0] + regressSEG.coefficents[3]*xSeg[:,3]##predictor (for plotting purposes)

            #printStatsCentury(
            #                  cv,##model of interest
            #                  regressSEGD,##regression object of interest
            #                  run = True
            #                 )

            ################################Trended Regression################################
            X = dSet[['temp','heating','qbo']]
            Y = dSet['h2o']

            regr = linear_model.LinearRegression()
            regr.fit(X, Y)
            #X = sm.add_constant(X) # adding a constant

            #model = sm.OLS(Y, X).fit()
            #predictions = model.predict(X)
            #
            #print_model = model.summary()
            print(f'Trended {cv:s}')
            print(regr.coef_)
            ####################################################################################

            ################################Detrended Regression################################
            X = dSetD[['temp','heating','qbo']]
            Y = dSetD['h2o']

            regr = linear_model.LinearRegression()
            regr.fit(X, Y)
            #X = sm.add_constant(X) # adding a constant

            #model = sm.OLS(Y, X).fit()
            #predictions = model.predict(X)
            #
            #print_model = model.summary()
            print(f'Detrended {cv:s}')
            print(regr.coef_)
            ####################################################################################
            continue

            regressionPlot(##make trended regression plot (change to detrended variables for detrended regression plot
                           cv,##model being plotted
                           regressSEG.y - regressSEG.y[0],##simulated water vapor adjusted for initial water vapor anomolies equal to zero
                           regressSEG.Yhat - regressSEG.Yhat[0],##estimated water vapor adjusted for inital value equal to zero
                           Yhat1 - Yhat1[0],##temperature predictor adjusted for initial value equal to zero
                           Yhat2 - Yhat2[0],#bd predictor adjusted for initial value equal to zero
                           Yhat3 - Yhat3[0],#qbo predictor adjusted for initial value equal to zero
                           total_years,##time variable
                           run = False##don't create plots right now...
                          )

            adjR111[cv] = regressSEG.adjR2##trended regression adjusted R^2 values
            adjR222[cv] = regressSEGD.adjR2##detrended regression adjusted R^2 values

            tempCoeffA[cv]  = regressSEG.coefficents[1]##temperature coefficients for each model
            tempSCoeffA[cv] = regressSEGD.coefficents[1]##detrended temperature coefficients

            tempConfA[cv] = regressSEG.Confidence[1]##trended temperature coefficient 95% confidence
            tempSConfA[cv] = regressSEGD.Confidence[1]##detrended temperature coefficient 95% confidence

            bdCoeffA[cv]  = regressSEG.coefficents[2]##bd coefficients for each model
            bdSCoeffA[cv] = regressSEGD.coefficents[2]##detrended bd coefficients

            bdConfA[cv] = regressSEG.Confidence[2]##bd coefficient 95% confidence
            bdSConfA[cv] = regressSEGD.Confidence[2]##detrended bd coefficient 95% confidence

            qboCoeffA[cv]  = regressSEG.coefficents[3]##qbo coefficients for each model
            qboSCoeffA[cv] = regressSEGD.coefficents[3]##detrended qbo coefficients

            qboConfA[cv]  = regressSEG.Confidence[3]##qbo coefficient 95% confidence
            qboSConfA[cv] = regressSEGD.Confidence[3]##qbo coefficient 95% confidence

            modelTrend = trendCalc(arange(regressSEG.y.size),regressSEG.y)##trend calculation for simulated water vapor
            tempTrend  = trendCalc(arange(regressSEG.y.size),Yhat1)##trend calculation for simulated water vapor only due to tropospheric temperature change
            bdTrend    = trendCalc(arange(regressSEG.y.size),Yhat2)##trend calculation for simulated water vapor only due to brewer-dobson change
            qboTrend   = trendCalc(arange(regressSEG.y.size),Yhat3)##trend calculation for simulated water vapor only due to qbo temperature change

            modelTrendSlope[cv] = modelTrend[0]##trend slope
            modelTrendConf[cv]  = modelTrend[2]##trend 95% confidence

            tempTrendSlope[cv]  = tempTrend[0]##trend slope
            tempTrendConf[cv]   = tempTrend[2]##trend 95% confidence

            bdTrendSlope[cv] = bdTrend[0]##trend slope
            bdTrendConf[cv]  = bdTrend[2]##trend 95% confidence

            qboTrendSlope[cv] = qboTrend[0]##trend slope
            qboTrendConf[cv]  = qboTrend[2]##trend 95% confidence

        if(teraD):
            for j in range(0,size(total_years),segmentOfInterest):
                if(product(h2o[j:j + seg,:].shape) < seg*12):
                    break##break loop after last 10-year segment

                RML1    = tntlw[j:j + seg].reshape(product(h2o[j:j + seg,:].shape))##long-wave heating
                RMS1    = tntsw[j:j + seg].reshape(product(h2o[j:j + seg,:].shape))##short-wave heating
                h2oM1   = h2o[j:j + seg,:].reshape(product(h2o[j:j + seg,:,].shape))##water vapor
                tempM1  = ta[j:j + seg,:].reshape(product(h2o[j:j + seg,:,].shape))##temperature
                windM1  = ua[j:j + seg,:].reshape(product(h2o[j:j + seg,:].shape))##qbo

                h2o1 = monAmoms(h2oM1)##water vapor monthly anomolies
                h2o1 = Series(h2o1)

                ta1 = monAmoms(tempM1)##temperature monthly anomolies
                ta1 = Series(ta1)

                tntlw1 = monAmoms(RML1)##long-wave heating monthly anomolies
                tntsw1 = monAmoms(RMS1)##short-wave heating monthly anomolies
                netR = (tntlw1 + tntsw1)*86400.*((1000./80.)**(2./7.))##net-heating monthly anomolies
                netR = Series(netR)

                ua1 = monAmoms(windM1)##qbo monthly anomolies
                ua1 = Series(ua1)

                dSetDD = concat(
                               {
                                'h2o' : h2o1,
                                'temp' : ta1,
                                'heating' : netR,
                                'qbo' : ua1
                               },
                               axis = 1
                              )

                dSetD.temp    = dSetDD.temp.shift(3)##lag temperature by 3-months
                dSetD.heating = dSetDD.heating.shift(1)##lag net-heating by 1-month
                dSetD.qbo     = dSetDD.qbo.shift(3)##lag the qbo by 3-months
                dSetD         = dSetDD.dropna()##drop nan values due to lag

                xSeg  = zeros((dSetDD.index.values.size,varNums))##2D-array holding predictors of standardized water vapor
                xSegD = zeros((dSetDD.index.values.size,varNums))##2D-array holding predictors of decadal water vapor

                xSeg[:,0] = ones(dSetDD.index.values.size)##ones column, for constant (y-intercept) of linear regression
                xSeg[:,1] = dSetDD.temp.values##temperature monthly anomolies
                xSeg[:,2] = dSetDD.heating.values##bd monthly anomolies
                xSeg[:,3] = dSetDD.qbo.values##qbo monthly anomolies

                xSegD[:,0] = ones(dSetDD.index.values.size)##ones column, for constant (y-intercept) of linear regression
                xSegD[:,1] = dSetDD.temp.values/dSetDD.temp.values.std()##standardized temperature monthly anomolies
                xSegD[:,2] = dSetDD.heating.values/dSetDD.heating.std()##standardized bd monthly anomolies
                xSegD[:,3] = dSetDD.qbo.values##standardized qbo monthly anomolies

                try:##try to make sure no errors
                    regressSEG  = multiregress(##set pyObject of decadal regression
                                               dSetDD.h2o.values,
                                               xSeg,
                                               dSetDD.index.values.size,
                                               varNums,
                                               False
                                              )
                except:
                    break##else-break loop

                try:##try to make sure no errors
                    regressSEGD = multiregress(##set pyObject of standardized decadal regression
                                               dSetDD.h2o.values/dSetDD.h2o.values.std(),
                                               xSegD,
                                               dSetDD.index.values.size,
                                               varNums,
                                               False
                                              )
                except:
                    break##else-break loop

                tempCoeffD[cv].append(regressSEG.coefficents[1])
                bdCoeffD[cv].append(regressSEG.coefficents[2])
                qboCoeffD[cv].append(regressSEG.coefficents[3])

                if(regressSEG.adjR2 < 0):
                    adjR2D[cv].append(0)
                else:
                    adjR2D[cv].append(regressSEG.adjR2)

            printStatsDecade(##print decadal model regression statistics
                             cv,
                             tempCoeffD[cv],
                             bdCoeffD[cv],
                             qboCoeffD[cv],
                             adjR2D[cv],
                             run = False
                            )

    #if(teraA):
    #    centuryAdjR2(##century adjusted r2 plot
    #                 list(adjR111.keys()),
    #                 list(adjR111.values()),
    #                 list(adjR222.values()),
    #                 run = False
    #                )
    #    centuryRegressionCoeffPlot(##century regression coefficient plot
    #                               list(tempCoeffA.keys()),
    #                               list(tempCoeffA.values()),
    #                               list(tempConfA.values()),
    #                               list(tempSCoeffA.values()),
    #                               list(tempSConfA.values()),
    #                               list(bdCoeffA.values()),
    #                               list(bdConfA.values()),
    #                               list(bdSCoeffA.values()),
    #                               list(bdSConfA.values()),
    #                               list(qboCoeffA.values()),
    #                               list(qboConfA.values()),
    #                               list(qboSCoeffA.values()),
    #                               list(qboSConfA.values()),
    #                               run = False
    #                              )

    #    trendPlot(##model 21st century trend plot
    #              list(modelTrendSlope.keys()),
    #              list(modelTrendSlope.values()),
    #              list(modelTrendConf.values()),
    #              list(tempTrendSlope.values()),
    #              list(tempTrendConf.values()),
    #              list(bdTrendSlope.values()),
    #              list(bdTrendConf.values()),
    #              list(qboTrendSlope.values()),
    #              list(qboTrendConf.values()),
    #              run = False
    #             )

    if(teraD):
        decadeAdjR2(##decadal adjusted r2 plot
                    list(adjR2D.keys()),
                    list(adjR2D.values()),
                    ERAIadjR2,
                    MERRAadjR2,
                    run = False
                   )
        decadeRegressionCoeffPlot(##decadal regression coefficient plot
                                  list(tempCoeffD.keys()),
                                  list(tempCoeffD.values()),
                                  list(bdCoeffD.values()),
                                  list(qboCoeffD.values()),
                                  (eraITemp,eraIBD,eraIQBO),
                                  (eraITempConf,eraIBDConf,eraIQBOConf),
                                  (merraTemp,merraBD,merraQBO),
                                  (merraTempConf,merraBDConf,merraQBOConf),
                                  run = False
                                 )

    if(teraA and teraD):
        ensembleT = []##all decadal median temperature coefficients
        ensembleB = []##all decadal median bd coefficients
        ensembleQ = []##all decadal median qbo coefficients
        for v in list(tempCoeffD.values()):
            ensembleT.append(median(v))
        for v in list(bdCoeffD.values()):
            ensembleB.append(median(v))
        for v in list(qboCoeffD.values()):
            ensembleQ.append(median(v))

        decadeCenturyComparisonPlot(##coefficient comparison plot
                                    ensembleT,
                                    list(tempCoeffA.values()),
                                    ensembleB,list(bdCoeffA.values()),
                                    ensembleQ,list(qboCoeffA.values()),
                                    (eraITemp,eraIBD,eraIQBO),
                                    (merraTemp,merraBD,merraQBO)
                                   )

if __name__ == '__main__':
    main()
