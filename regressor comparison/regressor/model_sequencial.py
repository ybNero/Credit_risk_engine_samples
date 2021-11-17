# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 09:39:29 2018
@author: zhi.li04
非标准化

标准化
MAE
21.366533333333333
score
0.04470965549720118
"""
from keras.models import Sequential
from keras.layers import Dense
from pandas import Series
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.model_selection import train_test_split


readFileName="data.xlsx"
#读取excel
data=pd.read_excel(readFileName)
X=data.ix[:,"用户实名制是否通过核实":"当月旅游资讯类应用使用次数"]
y=data["score"]


st = StandardScaler()
X_transform = st.fit_transform(X)

#划分训练集和测试集
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_transform, y, test_size=0.3, random_state=42)
#X_train, X_test, y_train, y_test = model_selection.train_test_split(X_transform, y, test_size=0.3, random_state=42)

seq = Sequential()
seq.add(Dense(13, activation='relu',input_dim=X.shape[1]))
seq.add(Dense(13, activation='relu'))
seq.add(Dense(1, activation='relu'))
seq.compile(loss='mse', optimizer='adam')
seq.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=12, batch_size=10)

y_pred = seq.predict(X_test)
#rms = (np.mean((y - y_pred)**2))**0.5
MAE=sum(abs(y_test - y_pred))/len(y_test)
score=1/(1+MAE)
#print ("RF RMS", rms)
print("MAE",MAE)
print("score:",score)


'''
#测试集
data_oot=pd.read_excel("oot.xlsx")
X_oot=data_oot.ix[:,"用户实名制是否通过核实":"当月旅游资讯类应用使用次数"]
st = StandardScaler()
X_oot_transform= st.fit_transform(X_oot)
y_pred_oot = rf.predict(X_oot_transform)
df_y_pred=pd.DataFrame(y_pred_oot)
#最终预测分数
df_y_pred.to_excel("test_score.xlsx")
'''

