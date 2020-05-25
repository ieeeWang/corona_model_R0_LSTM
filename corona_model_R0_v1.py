# -*- coding: utf-8 -*-
"""
Created on Tue May 19 17:16:03 2020
corona_case_model_R0 v1
@author: lwang
"""

#%% import libraries
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.special as sps
from statistics import median
# for time stick axis
from datetime import datetime, timedelta, date
import matplotlib.dates as mdates

# from my helper functions:
# from get_data import extract_JHU_data, extract_JHU_data2
from corona_predict4 import predict_corona_patients
from get_JHU_data import pick_NL_data, pick_country_no_provices

#%% prepare data
C_NL, _, _= pick_NL_data(plot=0)
C_Italy, _, _ = pick_country_no_provices('Italy',plot=0)
  
# plt NL vs Italy
Nd = len(C_NL)
# ceeat a datetime object
day0 = date(2020, 1, 22)
print('First day: ', day0)
xt = [day0 + timedelta(days=i) for i in range(Nd)]
print('Last day: ', xt[-1])

        
plt.figure(figsize=(16,6))
plt.plot(xt, C_NL,'o-', linewidth=2, markersize=5, label='NL') 
plt.plot(xt, C_Italy,'o-', linewidth=2, markersize=5, label='Italy')
plt.legend(loc='upper left', fontsize='x-large')
plt.grid(True)
plt.gcf().autofmt_xdate()


#%% prepare date 
# first case occur in NL
day1 = date(2020, 2, 27)
dd =  (day1-day0).days # 10
C_nl = C_NL[dd:] # remove firt 10 days with 0 case

# correct a mistake: day 14 = 614 (not 503 from JHU dataset)
C_nl[14] = 614

# real daily increase
daily_new = np.concatenate(([0], np.diff(C_nl)))
# loc0 = np.where(daily_new == 0)
# print(daily_new[14])


Nd = len(C_nl)+ 60 # N days later
# ceeat a datetime object
print('First day: ', day1)
# previous days
xt1 = [day1 + timedelta(days=i) for i in range(len(C_nl))]
# previous days + N days later
xt2 = [day1 + timedelta(days=i) for i in range(Nd)]

#%%
# case 1: R0 = 0.9 till end
# d1 = 17
# d2 = 10
# day_list = [d1, d2, Nd-d1-d2]
# r0_list = [2.8, 1.3, 0.9] # corresponding to day_list
 
# case 2: R0 = 0.8 till end
d1 = 17
d2 = 10 # 15
d3 = 30
day_list = [d1, d2, d3, Nd-d1-d2-d3]
r0_list = [2.8, 1.3, 0.9, 0.8] # corresponding to day_list

print('1st phase end: ', day1+timedelta(days=d1))
print('2nd phase end: ', day1+timedelta(days=(d1+d2)))
print('3rd phase end: ', day1+timedelta(days=(d1+d2+d3)))
print('4th phase end: ', day1+timedelta(days=Nd))

#%% fit
start_time = time.time()
new_patient_day, accum_patient_day, incuba_current,\
    remainingdays = predict_corona_patients(patient_zero_no = 10,
                                     ob_day_list = day_list, r0_list =r0_list) 
    
elapsed_time = time.time() - start_time
print('elapsed_time:', elapsed_time)

#%% 
plt.figure(figsize=(15,10))
plt.subplot(2,1,1)
k=0 
plt.plot(xt1, C_nl,'o-', linewidth=2, markersize=5, label='real accumulated') 
plt.plot(xt1, daily_new,'o-', linewidth=2, markersize=5, label='real daily increase') 
plt.plot(xt2, accum_patient_day[k: Nd+k], label='estimate accumulated',linestyle='--',linewidth=3)  
plt.plot(xt2, incuba_current[k:Nd+k], label='estimate current incubation',linestyle='--')  
plt.plot(xt2, new_patient_day[k:Nd+k], label='estimate daily increase',linestyle='--')  
plt.legend(loc='best', fontsize='x-large')
# plt.legend(loc='upper left', fontsize='x-large')
plt.grid(True)
# plt.yscale('symlog')
plt.ylabel("Case number") 
plt.gcf().autofmt_xdate()  # 自动旋转日期标记
# plt.title('Netherlands: R0=' + str(r0_list[0]) + ', ' +str(r0_list[1])+ ', ' + 
#           str(r0_list[2]) + ' in days ' + str(day_list[0]) +', ' + 
#           str(day_list[1]) + ' and remaining days, respectively (Feb.27 - May20)') 
plt.title('Netherlands: R0=' + str(r0_list) + ' in days ' + str(day_list) + 
           ', respectively (Feb.27 - May.20)') 


plt.subplot(2,1,2)
plt.plot(xt1, C_nl,'o-', linewidth=2, markersize=5, label='real accumulated') 
plt.plot(xt1, daily_new,'o-', linewidth=2, markersize=5, label='real daily increase') 
plt.plot(xt2, accum_patient_day[k: Nd+k], label='estimate accumulated',linestyle='--',linewidth=3)  
plt.plot(xt2, incuba_current[k:Nd+k], label='estimate current incubation',linestyle='--')  
plt.plot(xt2, new_patient_day[k:Nd+k], label='estimate daily increase',linestyle='--')  

# plt.legend(loc='best', fontsize='x-large')
plt.grid(True)
plt.yscale('symlog')
plt.ylabel("Case number (log)") 
plt.gcf().autofmt_xdate()

#%% 
plt.figure(figsize=(15,10))
plt.plot(xt1, daily_new,'o-', linewidth=2, markersize=5, label='reported daily increase')  
plt.plot(xt2, incuba_current[k:Nd+k], label='estimate current incubation',linestyle='--')  
plt.plot(xt2, new_patient_day[k:Nd+k], label='estimate daily increase',linestyle='--')  
plt.legend(loc='best', fontsize='x-large')
plt.grid(True)
# plt.yscale('symlog')
plt.ylabel("Case number") 
plt.gcf().autofmt_xdate()





