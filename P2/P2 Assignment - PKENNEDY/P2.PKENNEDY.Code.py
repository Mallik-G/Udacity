# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 12:23:47 2015

@author: patrickkennedy
"""

import numpy as np
import pandas as pd
import scipy
import scipy.stats
from sklearn.linear_model import SGDRegressor
import statsmodels.api as sm
import matplotlib.pyplot as plt

def normalized_features(features):
    means = np.mean(features, axis=0)
    std_devs = np.std(features, axis=0)
    normalized_features = (features - means) / std_devs
    return means, std_devs, normalized_features
    
def recover_params(means, std_devs, norm_intercept, norm_params):
    intercept = norm_intercept - np.sum(means * norm_params / std_devs)
    params = norm_params / std_devs
    return intercept, params
    
def linear_regression_GD(features, values):
    means, std_devs, features = normalized_features(features)
    model = SGDRegressor(eta0=0.001)
    results = model.fit(features, values)
    intercept = results.intercept_
    params = results.coef_
    return intercept, params
    
    
def predictionsGD(dataframe):
    features = dataframe[['rain', 'hour']]
    dummy_units = pd.get_dummies(dataframe['UNIT'], prefix='unit')
    features = features.join(dummy_units)
    dummy_units = pd.get_dummies(dataframe['DATEn'])
    features = features.join(dummy_units)
    
    values = dataframe['ENTRIESn_hourly']
    
    features_array = features.values
    values_array = values.values
    
    means, std_devs, normalized_features_array = normalized_features(features_array)
    
    norm_intercept, norm_params = linear_regression_GD(normalized_features_array, values_array)
    
    intercept, params = recover_params(means, std_devs, norm_intercept, norm_params)
    
    predictions = intercept + np.dot(features_array, params)
    print(params)
    return predictions
    
    

def mann_whitney_plus_means(turnstile_weather):
    with_rain = turnstile_weather['ENTRIESn_hourly'][turnstile_weather['rain']==1]
    without_rain = turnstile_weather['ENTRIESn_hourly'][turnstile_weather['rain']==0]
    with_rain_mean = np.mean(with_rain)
    without_rain_mean = np.mean(without_rain)
    U, p = scipy.stats.mannwhitneyu(with_rain, without_rain)
    return with_rain_mean, without_rain_mean, U, p
    

def compute_r_squared(data, predictions):
    mean_data = np.mean(data)
    SST = np.sum((data - mean_data)**2)
    SSres = np.sum((predictions - mean_data)**2)
    r_squared = SSres / SST
    return r_squared




def linear_regression_OLS(features, values):
    features = sm.add_constant(features)
    model = sm.OLS(values, features)
    results = model.fit()
    params = results.params[1:]
    intercept = results.params[0]
    return intercept, params

def predictionsLR(dataframe):
    features = dataframe[['rain']]
    dummy_units = pd.get_dummies(dataframe['UNIT'], prefix='unit')
    features = features.join(dummy_units)
    
    values = dataframe['ENTRIESn_hourly']
    
    intercept, params = linear_regression_OLS(features, values)
    predictions = intercept + np.dot(features, params)
    return predictions
    

def stationMeans(dataframe):    
    
    rain = []
    no_rain = []
    station_list = []
    entriesn_hourly_station = pd.DataFrame()
    
    
    for station in dataframe['station'].unique():
        rain.append(np.mean(dataframe['ENTRIESn_hourly'][dataframe['rain']==1][dataframe['station']==station]))
        no_rain.append(np.mean(dataframe['ENTRIESn_hourly'][dataframe['rain']==0][dataframe['station']==station]))
        station_list.append(station)
    
    entriesn_hourly_station["station"] = station_list
    entriesn_hourly_station["rain"] = rain
    entriesn_hourly_station["no_rain"] = no_rain
    entriesn_hourly_station["diff"] = entriesn_hourly_station["rain"] - entriesn_hourly_station["no_rain"]
    
    
    
    return entriesn_hourly_station



def rain_hists(rain, no_rain):
    
    plt.hist(no_rain.values, bins=20, range=[26000, 50000], color='g', label='No Rain')
    plt.hist(rain.values, bins=20, range=[26000, 50000], color='b', label='Rain')
    plt.title("Turnstile Entries per Hour Given Presence of Rain")
    plt.xlabel("Volume of Ridership in Entries per Hour")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    

def station_barchart(dataFrame):
      
    
    plt.barh(dataFrame['count'][::12], dataFrame['diff'][::12], height=7)
    plt.yticks(dataFrame['count'][::12],dataFrame['station'])
    plt.xlabel('Difference in Mean Entries per Hour')
    plt.title('Station Ridership Difference Given Presence of Rain')    
    
    plt.tight_layout()
    
    plt.show()