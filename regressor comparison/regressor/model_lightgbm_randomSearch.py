# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 09:39:29 2018
@author: zhi.li
StandardScaler
MAE 19.1058
score: 0.049736891842155004

no StandardScaler
MAE 14.877202547291887
score: 0.06298338747152697

1000树
MAE 15.083816231669893
score: 0.062174299034264434

best param: {'num_leaves': 42, 'max_depth': 7}
MAE 14.860683845758857
score: 0.06304898387262159

best param: {'num_leaves': 65, 'max_depth': 7}
MAE 14.920263092782148
score: 0.0628130323080763

best param: {'num_leaves': 36, 'max_depth': 14}
MAE 14.90316336090575
score: 0.06288057145022284
"""
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import lightgbm as lgb
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

params_test1={
    'max_depth': range(2,20,1),
    'num_leaves':range(10, 100, 1),
    'learning_rate':[0.1,0.05,0.01],
    'bagging_fraction ':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
    'feature_fraction':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
}

#划分训练集和测试集
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=42)
#X_train, X_test, y_train, y_test = model_selection.train_test_split(X_transform, y, test_size=0.3, random_state=42)
#lightgbm
model = lgb.LGBMRegressor(metric='mse')
#model = lgb.LGBMRegressor()
rf = RandomizedSearchCV(estimator = model, param_distributions = params_test1, n_jobs = -1,return_train_score=True,scoring='neg_mean_squared_error')
rf.fit(X_train, y_train)
print("best param:",rf.best_params_)
print( rf.cv_results_)
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

