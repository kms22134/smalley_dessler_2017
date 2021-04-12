import statsmodels.api as sm

spector_data = sm.datasets.spector.load(as_pandas=True)

spector_data.exog = sm.add_constant(spector_data.exog, prepend=False)

mod = sm.OLS(spector_data.endog, spector_data.exog)
print(help(sm.OLS))
