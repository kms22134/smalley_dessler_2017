import pandas as pd
from numpy import zeros,ones
from sklearn import linear_model
import statsmodels.api as sm
from smalley2017_regress import * 

Stock_Market = {'Year': [2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016],
                'Month': [12, 11,10,9,8,7,6,5,4,3,2,1,12,11,10,9,8,7,6,5,4,3,2,1],
                'Interest_Rate': [2.75,2.5,2.5,2.5,2.5,2.5,2.5,2.25,2.25,2.25,2,2,2,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75],
                'Unemployment_Rate': [5.3,5.3,5.3,5.3,5.4,5.6,5.5,5.5,5.5,5.6,5.7,5.9,6,5.9,5.8,6.1,6.2,6.1,6.1,6.1,5.9,6.2,6.2,6.1],
                'Stock_Index_Price': [1464,1394,1357,1293,1256,1254,1234,1195,1159,1167,1130,1075,1047,965,943,958,971,949,884,866,876,822,704,719]        
                }

df = pd.DataFrame(Stock_Market,columns=['Year','Month','Interest_Rate','Unemployment_Rate','Stock_Index_Price'])

X = df[['Interest_Rate','Unemployment_Rate']]
Y = df['Stock_Index_Price']

xSeg      = zeros((df.index.size,3))##2D array used for trended century regressions
xSeg[:,0] = ones(df.index.values.size)##first column of array is ones for regression calculations
xSeg[:,1] = df['Interest_Rate'].values##trended temperature anomolies
xSeg[:,2] = df['Unemployment_Rate'].values##trended net radiative heating anomolies

regressSEG   = multiregress(##set a multi-linear regression python object for the trended regressions
			    df['Stock_Index_Price'].values,##set a multi-linear regression python object inputs: response variable-water vapor,
			    xSeg,##predictor variables-xSeg
			    df.index.values.size,##length of the regression-dSet
			    3,##index.values.size,number of variables-varNums
			    True
			   )

print(regressSEG.coefficents)
print()
print(regressSEG.adjR2)
print()
print(1 - (((1 - regressSEG.R2) * (df.index.values.size - 1)) / (df.index.values.size - 2 - 1)))

