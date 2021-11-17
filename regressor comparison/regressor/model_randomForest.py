# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 09:39:29 2018

@author: zhi.li04

100颗树，StandardScaler
MAE 15.3823373333333
score: 0.06104135079463236

100颗树，no StandardScaler
MAE 15.383384666666757
score: 0.061037448631391544

500颗树，no StandardScaler
MAE 15.348868133333287
score: 0.061166313890631104

500颗树，StandardScaler
MAE 15.34924546666663
score: 0.06116490219923802

1000树
MAE 15.336045533333323
score: 0.06121432496986637

500颗树，no StandardScaler,深度6
MAE 16.479017115102845
score: 0.05721145493564076

500颗树，no StandardScaler,深度3
MAE 19.399696242430217
score: 0.04902033775973862

n_estimators=500,max_depth=14,min_samples_split=12
MAE 15.323770036230439
score: 0.06126035822487761

n_estimators=500,max_depth=14,min_samples_split=12,n_jobs=-1
MAE 15.321000114334572
score: 0.06127075503919088

n_estimators=500,max_depth=14,min_samples_split=80,bootstrap=True
MAE 15.329166273278192
score: 0.06124011374888421


best param: {'min_samples_leaf': 7}
MAE 15.363766202982903
score: 0.06111062622110264

criterion="mae"
MAE 16.185653333333303
score: 0.05818807004912635

best param: {'min_samples_split': 20}
MAE 15.344791874801906
score: 0.06118156827323442

best param: {'min_samples_split': 30}
MAE 15.330841332183342
score: 0.06123383233350573

n_estimators=500,max_depth=14,min_samples_split=80
MAE 15.448921607324463
score: 0.06079425897164679

n_estimators=500,max_depth=14,min_samples_split=30,min_samples_leaf=7,bootstrap=True
MAE 15.346146874247953
score: 0.0611764966810264

n_estimators=1000,max_depth=14,min_samples_split=12,n_jobs=-1
MAE 15.319042004230988
score: 0.06127810687298514

n_estimators=1000,max_depth=20,min_samples_split=12,n_jobs=-1
MAE 15.318632686099374
score: 0.06127964390373376

n_estimators=1000,max_depth=25,min_samples_split=12,n_jobs=-1
MAE 15.316957318743183
score: 0.06128593588041726

rf = RandomForestRegressor(n_estimators=1000,min_samples_split=12,n_jobs=-1)
MAE 15.320287975036264
score: 0.061273428601849034

n_estimators=1000,max_depth=14,min_samples_split=30,min_samples_leaf=7,bootstrap=True
MAE 15.346459400777968
score: 0.06117532705292791
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

readFileName="data.xlsx"
#读取excel
data=pd.read_excel(readFileName)
X=data.ix[:,"用户实名制是否通过核实":"当月旅游资讯类应用使用次数"]
y=data["score"]


st = StandardScaler()
X_transform = st.fit_transform(X)
#划分训练集和测试集
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=42)
#X_train, X_test, y_train, y_test = model_selection.train_test_split(X_transform, y, test_size=0.3, random_state=42)

#随机森林模型
#rf=RandomForestRegressor(n_estimators=1000)
#rf = RandomForestRegressor(n_estimators=1000,max_depth=14,min_samples_split=30,min_samples_leaf=7,bootstrap=True)
rf = RandomForestRegressor(n_estimators=1000,max_depth=25,min_samples_split=12,n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
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

