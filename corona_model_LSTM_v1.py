# -*- coding: utf-8 -*-
"""
Created on Thu May 21 11:22:24 2020

@author: lwang
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow.keras import models,layers,losses,metrics,callbacks 
print ('tf:', tf.__version__)

#%% plot cumulated data
df = pd.read_csv("./data/covid-19.csv",sep = "\t") #sep argument: delimit by a tab (not good for open in exel)
df.plot(x = "date",y = ["confirmed_num","cured_num","dead_num"],figsize=(10,6))
plt.xticks(rotation=60)

# df.to_csv('covid-19_copy.csv') # save dataframe into a .csv file
# df = pd.read_csv("./data/covid-19_copy.csv")

# daily statistic
dfdata = df.set_index("date")
dfdiff = dfdata.diff(periods=1).dropna()
dfdiff = dfdiff.reset_index("date")

dfdiff.plot(x = "date",y = ["confirmed_num","cured_num","dead_num"],figsize=(10,6))
plt.xticks(rotation=60)
dfdiff = dfdiff.drop("date",axis = 1).astype("float32")


#%% use previoue 8 days as input to predict the current day
WINDOW_SIZE = 8

def batch_dataset(dataset):
    dataset_batched = dataset.batch(WINDOW_SIZE,drop_remainder=True)
    return dataset_batched

# ds_data = tf.data.Dataset.from_tensor_slices(tf.constant(dfdiff.values,dtype = tf.float32)).window(WINDOW_SIZE,shift=1).flat_map(batch_dataset)
# divide data with window=8 and shift=1
ds_data = tf.data.Dataset.from_tensor_slices(tf.constant(dfdiff.values,dtype = tf.float32)).window(WINDOW_SIZE,shift=1)
# i=0
# for element in ds_data:
#     i+=1
#     print(i, list(element.as_numpy_iterator()))  
# remove windows with length < 8    
ds_data = ds_data.flat_map(batch_dataset)
# i=0
# for element in ds_data.as_numpy_iterator():
#     i+=1
#     print(i, element)  
    
# remove first 8 elements, remaining as labels 
ds_label = tf.data.Dataset.from_tensor_slices(
    tf.constant(dfdiff.values[WINDOW_SIZE:],dtype = tf.float32))
# show label
i=0
for element in ds_label.as_numpy_iterator():
    i+=1
    print(i, element)    
    
# all 38 samples in one batch
ds_train = tf.data.Dataset.zip((ds_data,ds_label)).batch(38).cache()

# i=0
# for element in ds_train.as_numpy_iterator():
#     i+=1
#     print(i, element) 
    
#%% 
class Block(layers.Layer):
    def __init__(self, **kwargs):
        super(Block, self).__init__(**kwargs)
    
    def call(self, x_input,x):
        x_out = tf.maximum((1+x)*x_input[:,-1,:],0.0)
        return x_out
    
    def get_config(self):  
        config = super(Block, self).get_config()
        return config
    
#%% build a model
tf.keras.backend.clear_session()
x_input = layers.Input(shape = (None,3),dtype = tf.float32)
x = layers.LSTM(3,return_sequences = True,input_shape=(None,3))(x_input)
x = layers.LSTM(3,return_sequences = True,input_shape=(None,3))(x)
x = layers.LSTM(3,return_sequences = True,input_shape=(None,3))(x)
x = layers.LSTM(3,input_shape=(None,3))(x)
x = layers.Dense(3)(x)

#考虑到新增确诊，新增治愈，新增死亡人数数据不可能小于0，设计如下结构
# x = tf.maximum((1+x)*x_input[:,-1,:],0.0)
x = Block()(x_input,x)
model = models.Model(inputs = [x_input],outputs = [x])
model.summary()


#自定义损失函数，考虑平方差和预测目标的比值
class MSPE(losses.Loss):
    def call(self,y_true,y_pred):
        err_percent = (y_true - y_pred)**2/(tf.maximum(y_true**2,1e-7))
        mean_err_percent = tf.reduce_mean(err_percent)
        return mean_err_percent
    
    def get_config(self):
        config = super(MSPE, self).get_config()
        return config        
    
#%% model param. set-up
import os
import datetime

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer,loss=MSPE(name = "MSPE"))
# save name and dir of model
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = os.path.join('data', 'autograph', stamp)

## 在 Python3 下建议使用 pathlib 修正各操作系统的路径
# from pathlib import Path
# stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# logdir = str(Path('./data/autograph/' + stamp))

#% add 3 callback functions below
#为Tensorboard可视化保存日志信息。支持评估指标，计算图，模型参数等的可视化:
tb_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
#如果loss在100个epoch后没有提升，学习率减半:
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss",factor = 0.5, patience = 100)
#当loss在200个epoch后没有提升，则提前终止训练:
stop_callback = tf.keras.callbacks.EarlyStopping(monitor = "loss", patience= 200)

callbacks_list = [tb_callback,lr_callback,stop_callback]

#%% train
history = model.fit(ds_train,epochs=500,callbacks = callbacks_list)

#%% show model train loss
import matplotlib.pyplot as plt

def plot_metric(history, metric):
    train_metrics = history.history[metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.title('Training '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric])
    plt.show()    

plt.figure()
plot_metric(history,"loss")
 
#%% use model
dfresult = dfdiff[["confirmed_num","cured_num","dead_num"]].copy()
dfresult.tail() #print last 5



dfresult = dfresult.head(8)
dfresult.tail() #print last 5

#预测此后100天的新增走势,将其结果添加到dfresult中
for i in range(38):
    temp = tf.expand_dims(dfresult.values[-8:,:],axis = 0)
    arr_predict = model.predict(tf.constant(temp))
    print(arr_predict)
    dfpredict = pd.DataFrame(tf.cast(tf.floor(arr_predict),tf.float32).numpy(),
                columns = dfresult.columns)
    print(dfpredict)
    dfresult = dfresult.append(dfpredict,ignore_index=True)


dfresult.query("confirmed_num==0").head()
dfresult.query("dead_num==0").head()
dfresult.query("cured_num==0").head()

# plot 
plt.figure()
dfresult.plot(y = ["confirmed_num","cured_num","dead_num"],figsize=(10,6))
# plt.xticks(rotation=60)


#%% save model
model.save('./data/tf_model_savedmodel/', save_format="tf")
print('export saved model.')

#%% use saved model
model_loaded = tf.keras.models.load_model('./data/tf_model_savedmodel',compile=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model_loaded.compile(optimizer=optimizer,loss=MSPE(name = "MSPE"))
model_loaded.predict(ds_train)
