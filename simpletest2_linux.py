
import pandas as pd
import numpy as np
import xgboost as xgb
from matplotlib import pyplot as plt

load_path = 'rowdata_test1.csv'
regression_path = 'regressiontest_test1.csv'
history_scale = 30# 选择过去15个时间点位加入
pred_pos = 10# 选择预测未来第几个数据
rawdata=pd.read_csv(load_path)
rawdata = rawdata.drop('Unnamed: 0',axis=1) # 去掉时间


def bar_wash0522(kline, fast_length, slow_length, history_scale, pred_pos):
    # get_bar（经macd）的数据 快macd尺度 慢macd尺度 历史信息量 预测位点
    output_size = len(kline) - pred_pos - history_scale
    data = -255 * np.ones((output_size, 6 * history_scale + 6))
    label = -255 * np.ones((output_size, 1))
    label2 = -255 * np.ones((output_size))
    rawdata1 = kline.values
    # label2 = -255*np.ones((output_size,1)) #label2幅度百分比 简单回测用
    for i in range(output_size):
        pos = i + history_scale

        # print(rawdata1[pos,:3])
        # print(rawdata1[i:pos,:4].reshape((1,-1)))
        temp1 = rawdata1[pos, :3]
        temp1 = temp1.reshape((1, -1))
        history_price = np.concatenate([temp1, rawdata1[i:pos, :4].reshape((1, -1))], axis=1)

        macd_feature = (rawdata1[i:pos + 1, 5:7].reshape((1, -1)) / rawdata1[pos, 3] - 1) * 100
        history_price_feature = (history_price / rawdata1[pos, 3] - 1) * 100

        temp1 = np.concatenate([history_price_feature, rawdata1[pos, 4].reshape((1, -1))], axis=1)
        data[i] = np.concatenate([temp1, macd_feature], axis=1)
        # 标签 3档 0跌 1稳 2涨
        if ((rawdata1[pos + pred_pos, 3] - rawdata1[pos, 3]) / rawdata1[pos, 3]) < -0.0025:
            label[i, 0] = 0
        elif ((rawdata1[pos + pred_pos, 3] - rawdata1[pos, 3]) / rawdata1[pos, 3]) > 0.0025:
            label[i, 0] = 2
        else:
            label[i, 0] = 1
        label2[i] = (rawdata1[pos + pred_pos, 3] - rawdata1[pos, 3]) / rawdata1[pos, 3]
    data_all = np.concatenate([data, label], axis=1)
    return data_all, label2

slow_length = 12
data_washed,data_retest = bar_wash0522(rawdata,5,slow_length,history_scale,pred_pos)

# 转dataframe
data_washed = pd.DataFrame(data_washed)
data_washed

from sklearn.model_selection import train_test_split
x = data_washed.drop(np.size(data_washed,1)-1, axis=1)
y = data_washed[np.size(data_washed,1)-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25) # 此处目的仅是调取划分后的

x_test = x[np.size(x_train,0):]
x_train = x[:np.size(x_train,0)]
y_test = y[np.size(x_train,0):]
y_train = y[:np.size(x_train,0)]

#The data is stored in a DMatrix object
#label is used to define our outcome variable
dtrain=xgb.DMatrix(x_train,label=y_train)
dtest=xgb.DMatrix(x_test)

#setting parameters for xgboost
parameters={'max_depth':6, 'eta':0.1, 'silent':1,'objective':'multi:softmax','num_class':3,'eval_metric':'auc','learning_rate':.05}

#training our model
num_round=10
from datetime import datetime
start = datetime.now()
xg=xgb.train(parameters,dtrain,num_round)
stop = datetime.now()

#Execution time of the model
execution_time_xgb = stop-start
print(execution_time_xgb)
ypred=xg.predict(dtest) 
