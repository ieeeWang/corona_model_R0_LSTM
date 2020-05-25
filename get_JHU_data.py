# -*- coding: utf-8 -*-
"""
Created on Tue May 19 13:43:18 2020
return the confirmedcases (C) and death (D) given an input of country name
@author: lwang
"""

import json
import wget
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
# from china_cities import *  

from pandas.io.json import json_normalize  
from sklearn.model_selection import train_test_split

# below 3 libaries for time stick axis
from datetime import datetime, timedelta, date
import matplotlib.dates as mdates

#%%
def download_JHU_data():
    """
    Function: Extract JHU github data
    https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series
    """

    urls = ['https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv', 
            'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv', 
            'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv',
            ]
    for url in urls:
        filename = wget.download(url)   
        """
        Function: now the above file is already downloaded under your local directory
                  You can check the filename below:
        """
        print('filename', filename)


def pick_country_data(country_name): 
    show_data = False
    """
    Function: read JHU data
        - Number of confirmed case
        - Number of death
        - Number of recovered
    """
    confirmed_df = pd.read_csv('./data/time_series_covid19_confirmed_global.csv')
    deaths_df   = pd.read_csv('./data/time_series_covid19_deaths_global.csv')
    recoveries_df = pd.read_csv('./data/time_series_covid19_recovered_global.csv')        
    if show_data == True:
        print('confirmed  = ', confirmed_df.head(3))
        print('deaths = ', deaths_df.head(3))
        print('recovered =', recoveries_df.head(3))
        # print('cols = ',confirmed_df.keys())
    
    cols = confirmed_df.keys()
    dates = cols[4:]
    # confirmed = confirmed_df.loc[:, cols[4]:cols[-1]]
    # deaths = deaths_df.loc[:, cols[4]:cols[-1]]
    # recoveries = recoveries_df.loc[:, cols[4]:cols[-1]]
    # dates = confirmed.keys()
    
    df_confirmed = confirmed_df.loc[confirmed_df['Country/Region'] == country_name]
    df_deaths = deaths_df.loc[deaths_df['Country/Region'] == country_name]
    df_recoveries = recoveries_df.loc[recoveries_df['Country/Region'] == country_name]
  
    if show_data == True:    
        print('confirmed_df = ', df_confirmed.head())
        print('deaths_df = ', df_deaths.head())
        print('recoveries_df = ', df_recoveries.head())
    
    return dates, df_confirmed, df_deaths, df_recoveries


def pick_NL_data(plot = True):
    #%% country including provinces, e.g., Netherlands   
    dates, df_confirmed, df_deaths, df_recoveries = pick_country_data('Netherlands')
    # print( df_confirmed.head())
    
    # tp_c = df_confirmed.loc[(df_confirmed['Province/State'] == 'Aruba')] # select a province
    tp_c = df_confirmed.iloc[3] # using index. Note 'nan' is not able to be index
    tp_d = df_deaths.iloc[3]
    tp_r = df_recoveries.iloc[3]
    
    C = tp_c[4:].to_numpy()
    D = tp_d[4:].to_numpy()
    R = tp_r[4:].to_numpy()
    
    if plot==True:
        # (B) plot using date sticks as x. It automated scale x-axis!
        Nd = len(dates)
        # ceeat a datetime object
        day0 = datetime(2020, 1, 22)
        print('First day: ', day0)
        xt = [day0 + timedelta(days=i) for i in range(Nd)]
        # print(xt)
        
        plt.figure(figsize=(16,6))
        plt.plot(xt, C,'o-', linewidth=2, markersize=5, label='confirmed') 
        plt.plot(xt, D,'o-', linewidth=2, markersize=5, label='death')
        plt.legend(loc='upper left', fontsize='x-large')
        plt.grid(True)
        plt.gcf().autofmt_xdate()  # 自动旋转日期标记

    return C, D, R
 
def pick_country_no_provices(country_name, plot = True):
    # country including no provinces, e.g., Italy 
    dates, df_confirmed, df_deaths, df_recoveries = pick_country_data(country_name)
    tp_c = df_confirmed.iloc[0] # using index. Note 'nan' is not able to be index
    tp_d = df_deaths.iloc[0]
    tp_r = df_recoveries.iloc[0]
    
    C = tp_c[4:].to_numpy()
    D = tp_d[4:].to_numpy()
    R = tp_r[4:].to_numpy()
    
    if plot==True:
        # (B) plot using date sticks as x. It automated scale x-axis!
        Nd = len(dates)
        # ceeat a datetime object
        day0 = datetime(2020, 1, 22)
        print('First day: ', day0)
        xt = [day0 + timedelta(days=i) for i in range(Nd)]
        # print(xt)
        
        plt.figure(figsize=(16,6))
        plt.plot(xt, C,'o-', linewidth=2, markersize=5, label='confirmed') 
        plt.plot(xt, D,'o-', linewidth=2, markersize=5, label='death')
        plt.legend(loc='upper left', fontsize='x-large')
        plt.grid(True)
        plt.gcf().autofmt_xdate()  # 自动旋转日期标记

    return C, D, R 


if __name__=='__main__':      
    #%% prepare data: after download the data, move them under the 'data' folder
    download_JHU_data()
      
    #%% country including provinces, e.g., Netherlands   
    C, D, R = pick_NL_data(plot=1)
    
    pick_country_no_provices('Italy',plot=1)
          
    print('get_JHU_data was executed...')  
    
    #%% save NL data into .csv file
    # FIRST day with recording
    day0 = date(2020, 1, 22)
    # first case occur in NL
    day1 = date(2020, 2, 27)
    dd =  (day1-day0).days # int 36
    C_nl = C[dd:] # remove firt 10 days with 0 case
    R_nl = R[dd:]
    D_nl = D[dd:]
    # correct a mistake: day 14 = 614 (not 503 from JHU dataset)
    C_nl[14] = 614
    Nd = len(C_nl)
    xt = [day1 + timedelta(days=i) for i in range(Nd)]
    # creat a dateframd
    my_dict = {"date": xt,"confirmed_num": C_nl,"cured_num": R_nl,"dead_num": D_nl}
    myDF = pd.DataFrame(my_dict)
    
    # save dataframe into a .csv file
    myDF.to_csv('covid-19_nl.csv') # will overwrite same-name file
    # df = pd.read_csv("./data/covid-19_nl.csv")

    
    #%% (A) plot using default dates as x
    # plt.figure(figsize=(16,6))
    # plt.plot(dates, C,'o-', linewidth=2, markersize=5, label='confirmed') 
    # plt.plot(dates, D,'o-', linewidth=2, markersize=5, label='death')
    # plt.legend(loc='upper left', fontsize='x-large')
    # plt.grid(True)
    # # plt.xticks(rotation=60)
    # plt.gcf().autofmt_xdate()  # 自动旋转日期标记
    
    # #%% (B) plot using date sticks as x. It automated scale x-axis!
    # Nd = len(dates)
    # # creat a datetime object
    # day0 = datetime(2020, 1, 22)
    # print('First day: ', day0)
    # xt = [day0 + timedelta(days=i) for i in range(Nd)]
    # # print(xt)
    
    # plt.figure(figsize=(16,6))
    # plt.plot(xt, C,'o-', linewidth=2, markersize=5, label='confirmed') 
    # plt.plot(xt, D,'o-', linewidth=2, markersize=5, label='death')
    # plt.legend(loc='upper left', fontsize='x-large')
    # plt.grid(True)
    # plt.gcf().autofmt_xdate()  # 自动旋转日期标记




