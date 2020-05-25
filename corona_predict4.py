# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 15:20:55 2020
for predicting the coronavirus situation in the Netherlands
v3: use natural date instead of day 1, 2,...
v4: use different r0 & observation_day list
@author: lwang
"""

#%% import libraries
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.special as sps
from statistics import median
# below 3 libaries for time stick axis
from datetime import datetime
from datetime import timedelta
import matplotlib.dates as mdates


#%%
# to pad real number (with nan) = observation_day
def pad_real_number(Nd, real_patient_no):
    real_patient_no = real_patient_no.astype(float)
    # Nd = sum(ob_day1)
    real_pad = np.empty(Nd - len(real_patient_no))
    real_pad[:] = np.nan #assign NaN to all values
    real_patient_no = np.concatenate((real_patient_no, real_pad), axis=0) 
    return real_patient_no


# function for mapping R0 to increasing rate/per day
n_days = 5 # here using pror: R0 works on every 5 days
f_increase = lambda x: x**(1/(n_days-1)) -1 # x is R0

real_r0 = lambda x: x + x**2 + x**3 + x**4 # x is R0


# gamma distribution of incubation days with median = 4
def get_gamma_distr(n): # n>=1
    shape, scale = 2.5, 2.  
    np.random.seed(0)
    s = np.random.gamma(shape, scale, n)
    s = np.floor(s)
    print('median:',median(s))
    return s


#%% senario 2: one patient will spread to r0 persons within period_r0
def predict_corona_patients(patient_zero_no = 10, ob_day_list=[18], r0_list=[3]):  
    populaton_NL = 1.7e7 
    
    N_period = len(ob_day_list)

    # incubation days of the initial patients
    patient_remainingdays = get_gamma_distr(patient_zero_no)
    
    # the number of newly-increased potential patient (with full incubation days) every day
    new_patient_list = []
    new_patient_list.append(patient_zero_no)
    
    new_patient_day = [] # the newly confirmed number of patients per day
    accum_patient_day = [] # the accumulated number of patients per day
    incubation_patient_current = [] # the number of current patients in incubation but not been test yet
    for p in range(N_period):
        print('period ', p, '/',N_period)
        observation_day = ob_day_list[p]
        r0 = r0_list[p]
    
        for d in range(observation_day):
            
            f_increase_rate = f_increase(r0)
            # print('incrasing rate per day:',f_increase_rate)
        
            patient_remainingdays -=1
            d+=1
            
            new_infected = new_patient_list[-1]*(1+f_increase_rate)
            new_infected = int(np.floor(new_infected))
            print('day ', d, ': new infected:', new_infected)
            if new_infected < 1: # no new patient anymore
                new_patient_list.append(0)
            else:
                new_patient_list.append(new_infected)
                patient_remainingdays = np.concatenate((patient_remainingdays, 
                                                    get_gamma_distr(new_infected)))

            # count new patients
            tmp1 = sum(patient_remainingdays==0)
            # count accumunated patients
            tmp2 = sum(patient_remainingdays<0)   
            tempN = len(patient_remainingdays)
            tmp3 = tempN -tmp1-tmp2
            # tmp3 = sum(patient_remainingdays>0)
            if tempN > populaton_NL*0.66: # 0.66 = herd imunity threshold
                break # It terminates the nearest enclosing loop
            else:
                new_patient_day.append(tmp1)
                accum_patient_day.append(tmp2) 
                incubation_patient_current.append(tmp3)
   
    
    return new_patient_day, accum_patient_day, incubation_patient_current, patient_remainingdays
   
#%% run this file and avoid this file being run when being imported  
if __name__=='__main__':   

    #%% intitial values
    # ob_day1 = [17, 15, 200]
    # r0_1 = [2.8, 1.3, 0.9] 
    ob_day1 = [17, 10, 20]
    r0_1 = [2.8, 1.3, 0.8] 
    print('R0 in corina_predic4:',r0_1)
    
    # Netherlands: 1st = 2020, 2, 27
    real_patient_no1 = np.array([
        1,2,7,10,18,23,38,82,128,188,265,321,382,503,614,804,959,1135,1413,1705,
        2051,2460,2994,3643,4204,4749,5560,6412,7431,8603,9762,10866,11750,12595
                                 ])
    print ('Netherlands day past:',len(real_patient_no1))
    # Italy: 1128 = Feb 29
    real_patient_no3 = np.array([453, 655, 888,
        1128,1694,2036, 2502, 3089, 3858, 4636,5883, 7375, 9172, 10149,12462,15113,
        # 17660,21157,24747,27980,31506,35713,41035,47021,53578,59138,
        ])
    print ('Italy days past:', len(real_patient_no3))

         
    #%%  predict
    start_time = time.time()
    
    new_patient_day, accum_patient_day, incuba_current, patient_remainingdays = predict_corona_patients(
        ob_day_list = ob_day1, r0_list=r0_1)     
    
    r1=r0_1[1]
    realR0 = real_r0((1+f_increase(r1)))
    print('real R0:', realR0)
    
    elapsed_time = time.time() - start_time
    print('elapsed_time:', elapsed_time)
    
    #%% date stick
    # Nd = sum(ob_day1)
    Nd = len(accum_patient_day)
    
    real_patient_no = pad_real_number(Nd, real_patient_no1)
    real_patient_no3 = pad_real_number(Nd, real_patient_no3)
    # ceeat a datetime object
    day0 = datetime(2020, 2, 27)
    print('First day: ', day0)
    xt = [day0 + timedelta(days=i) for i in range(Nd)]
    # print(xt)
    
    #%% plot
    plt.figure(figsize=(15,10))
    plt.subplot(2,1,1)
    k= 0 # plot prediction data after k days 
    plt.plot(xt, real_patient_no,'o-', linewidth=2, markersize=5, label='real accumulated Netherlands')
    plt.plot(xt, accum_patient_day[k:], label='predict accumulated',linestyle='--',linewidth=2)  
    plt.plot(xt, new_patient_day[k:], label='predict daily increase',
             linestyle='--') 
    plt.plot(xt, incuba_current[k:], label='predict in incubation',linestyle='--')  
    plt.plot(xt, real_patient_no3,'o-', label='real accumulated Italy')
    plt.legend(loc='upper left', fontsize='x-large')
    # plt.legend(loc='lower right', fontsize='x-large')
    # plt.yscale('symlog')
    
    plt.grid(True)
    plt.ylabel("Case number") 
    plt.gcf().autofmt_xdate()  # 自动旋转日期标记
    # plt.title('Netherlands: R0='+str(r0_1[2]) + ' after Apr.1') 
    # plt.title('Netherlands: R0=' +str(r0_1[0]) + ' in first ' +str(ob_day1[0]) +' days, R0='+str(r0_1[1]) + ' in remaining days') 
    plt.title('Netherlands: R0=' + str(r0_1[0]) + ', ' +str(r0_1[1])+ ', ' + str(r0_1[2])+ ' in days ' + str(ob_day1[0]) +', ' + str(ob_day1[1]) + ' and remaining days, respectively') 
    
    
    plt.subplot(2,1,2)
    k= 0 # plot prediction data after k days 
    plt.plot(xt, real_patient_no,'o-', linewidth=2, markersize=5, label='real accumulated Netherlands')
    plt.plot(xt, accum_patient_day[k:],label='predict accumulated',
             linestyle='--',linewidth=2)  
    plt.plot(xt, new_patient_day[k:], label='predict daily increase',
             linestyle='--') 
    plt.plot(xt, incuba_current[k:], label='predict in incubation',linestyle='--')  
    plt.plot(xt, real_patient_no3,'o-', label='real accumulated Italy')
    # plt.legend(loc='upper left', fontsize='x-large')
    # plt.legend(loc='lower right', fontsize='x-large')
    plt.yscale('symlog')
    
    plt.grid(True)
    plt.ylabel("Case number (log)") 
    plt.gcf().autofmt_xdate()  # 自动旋转日期标记
    # plt.title('Netherlands: R0='+str(r0_1[2]) + ' after Apr.1') 
    plt.title('Netherlands: R0=' +str(r0_1[0]) + ' in first ' +str(ob_day1[0]) 
              +' days, R0='+str(r0_1[1]) + ' in remaining days') 
    
    print('corona_predict4py was executed...')
