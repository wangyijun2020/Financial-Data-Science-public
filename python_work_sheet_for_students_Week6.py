import numpy as np
import pandas as pd
import statsmodels

import statsmodels.api as sm

import matplotlib.pyplot as plt

# read the data of interest rate
import csv

from statsmodels.tsa import ar_model

data= pd.read_csv('D:\study\Sose2021\Finance Data Science\Week5\ps5-adj-checker/apple_mthly.csv')
data

data.plot()

data_3=data['3']
data_120=data['120']

cov_3_120= np.cov(data_3,data_120)/(np.std(data_3)*np.std(data_120))


# which lag are significant?
from statsmodels.graphics.tsaplots import plot_pacf
pacf_3= plot_pacf(data_3)
pacf_120= plot_pacf(data_120)

statsmodels.tsa.stattools.pacf(data_3, nlags=5, method='ols' , alpha=0.05)



pacf_120_p= sm.graphics.tsa.plot_pacf(data_120, nlags=5, method='ols' , alpha=0.05)

#AIC Task5


#ar_model=statsmodels.tsa.ar_model.AR(data_3 ,  )
#ar_model.fit=ar_model.fit()

#m3=ar_model.AR(data_3.to_numpy())
#m12=ar_model.AR(data_120.to_numpy())

def AIC(data, max_lags):
    AIC_values = np.zeros(max_lags)
    X, Y = lag_data(data, max_lags)

    for l in range(max_lags):
        x_lag = X[:, 0:l + 1]

        T = len(data)

        log_likelihood = sm.OLS(Y, sm.add_constant(x_lag)).fit().llf

        p = l + 2  # nr of parameters estimated by OLS

        AIC_values[l] = (-2 / T) * log_likelihood + (2 / T) * p

    opt_lags = 1 + np.where(AIC_values == min(AIC_values))[0].item()

    return AIC_values, opt_lags

AIC_values_3, optimal_lags_3 = AIC(data_3,5)
AIC_values_120, optimal_lags = AIC(data_120,5)
# Task6
def lag_data(data, lags):
    n = len(data) #公式中的T
    lagged_data = np.zeros((n - lags, lags))

    for i in range(lags):
        lagged_data[:, -(i + 1)] = data[i:(n - lags + i)]
    ts_data = data[lags:]

    return lagged_data, ts_data


def BIC(data, max_lags):
    BIC_values = np.zeros(max_lags)
    X, Y = lag_data(data, max_lags)

    for l in range(max_lags):
        x_lag = X[:, 0:l + 1]

        T = len(data)

        log_likelihood = sm.OLS(Y, sm.add_constant(x_lag)).fit().llf

        p = l + 2  # nr of parameters estimated by OLS

        BIC_values[l] = (-2 / T) * log_likelihood + (np.log(T) / T) * p

    opt_lags = 1 + np.where(BIC_values == min(BIC_values))[0].item()

    return BIC_values, opt_lags

BIC_values_3, optimal_lags_3 = BIC(data_3,5)
BIC_values_120, optimal_lags = BIC(data_120,5)
