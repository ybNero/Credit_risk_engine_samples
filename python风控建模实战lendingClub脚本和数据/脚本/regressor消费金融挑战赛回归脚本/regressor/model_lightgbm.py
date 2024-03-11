# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 09:39:29 2018
@author: zhi.li

默认
LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
       learning_rate=0.1, max_depth=-1, metric='mae', min_child_samples=20,
       min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,
       n_jobs=-1, num_leaves=31, objective=None, random_state=None,
       reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
       subsample_for_bin=200000, subsample_freq=0)
MAE 14.882147128963936
score: 0.06296377888203297


max_depth=7,num_leaves=45,bagging_fraction = 0.8,feature_fraction = 0.5,metric='mae',min_child_samples=21
MAE 14.804584335186549
score: 0.06327278078257643

max_depth=7,num_leaves=42,bagging_fraction = 0.8,feature_fraction = 0.5,metric='mae',min_child_samples=21
MAE 14.829490420831611
score: 0.06317322752752672

max_depth=7,num_leaves=42,bagging_fraction = 0.8,feature_fraction = 0.5,metric='mae'
MAE 14.834914655674762
score: 0.0631515875989663

best param: {'num_leaves': 42, 'max_depth': 7}
MAE 14.860683845758857
score: 0.06304898387262159

深度6
MAE 14.888437400878686
score: 0.0629388513652511

深度14
MAE 14.870905248997616
score: 0.0630083781807694

500树
MAE 14.945889991838662
score: 0.06271208446263046

1000树
树=1000
MAE 15.072731121064582
score: 0.06221717967330525

max_depth=14,bagging_fraction = 0.8,feature_fraction = 0.8
MAE 14.902866825412413
score: 0.06288174396342319
"""
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

#划分训练集和测试集
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=42)
#X_train, X_test, y_train, y_test = model_selection.train_test_split(X_transform, y, test_size=0.3, random_state=42)

#lightgbm
#rf=lgb.LGBMRegressor(metric='mae')
rf = lgb.LGBMRegressor(max_depth=7,num_leaves=45,bagging_fraction = 0.8,feature_fraction = 0.5,metric='mae',min_child_samples=21,learning_rate=0.1)
#rf = lgb.LGBMRegressor(max_depth=14,num_leaves=92,bagging_fraction = 0.8,feature_fraction = 0.6)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
#rms = (np.mean((y - y_pred)**2))**0.5
MAE=sum(abs(y_test - y_pred))/len(y_test)
score=1/(1+MAE)
#print ("RF RMS", rms)
print("MAE",MAE)
print("score:",score)


#测试集
data_oot=pd.read_excel("oot.xlsx")
X_oot=data_oot.ix[:,"用户实名制是否通过核实":"当月旅游资讯类应用使用次数"]
y_pred_oot = rf.predict(X_oot)
df_y_pred=pd.DataFrame(y_pred_oot)
#最终预测分数
df_y_pred.to_csv("lightgbm_test_score.csv")


'''
feature_importances=rf.feature_importances_
names=X.columns
list_feature_importances=list(zip(feature_importances,names))
df_feature_importances=pd.DataFrame(list_feature_importances)
df_feature_importances.to_excel("变量信息增益.xlsx")

n_features=X.shape[1]
plt.barh(range(n_features),rf.feature_importances_,align='center')
plt.yticks(np.arange(n_features),X.columns)
plt.title("lightgbm")
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()
'''