# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 09:39:29 2018

@author: zhi.li04

sclf = StackingRegressor(regressors=[lightgbm, xgboost, catboost],  
                          meta_regressor=lr) 
非标准化
MAE 21.338284014244344
score: 0.044766195978273665

标准化
MAE 31.629733333333334
score: 0.030646894652321195


非标准化
sclf = StackingRegressor(regressors=[randomForest, xgboost, catboost],  
                          meta_regressor=lightgbm) 
MAE 15.111416086010365
score: 0.06206779060645735


lightgbm默认参数，非标准化
sclf = StackingRegressor(regressors=[randomForest, xgboost, catboost],  
                          meta_regressor=lightgbm) 
MAE 16.680958228925444
score: 0.0565580206147444

sclf = StackingRegressor(regressors=[randomForest, xgboost, lightgbm],  
                          meta_regressor=catboost) 
MAE 16.76580398268089
score: 0.05628791137034139

sclf = StackingRegressor(regressors=[catboost, xgboost, lightgbm],  
                          meta_regressor=randomForest) 
MAE 15.444432318534236
score: 0.06081085565191066

"""
import xgboost as xg
import catboost as cb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from mlxtend.regressor import StackingRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import linear_model

readFileName="data.xlsx"
#读取excel
data=pd.read_excel(readFileName)
X=data.ix[:,"用户实名制是否通过核实":"当月旅游资讯类应用使用次数"]
y=data["score"]

'''
st = StandardScaler()
X_transform = st.fit_transform(X)
'''
#划分训练集和测试集
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=42)
#X_train, X_test, y_train, y_test = model_selection.train_test_split(X_transform, y, test_size=0.3, random_state=42)

lightgbm = lgb.LGBMRegressor(max_depth=7,num_leaves=45,bagging_fraction = 0.8,feature_fraction = 0.5,metric='mae',min_child_samples=21,learning_rate=0.1)
xgboost = xg.XGBRegressor(n_estimators=1000)
catboost = cb.CatBoostRegressor()

randomForest=rf = RandomForestRegressor(n_estimators=1000,max_depth=25,min_samples_split=12,n_jobs=-1)

sclf = StackingRegressor(regressors=[catboost, xgboost, lightgbm],  
                          meta_regressor=randomForest) 


'''   
print('3-fold cross validation:\n') 

for clf, label in zip([clf1, clf2, clf3, sclf],  
                      ['lightgbm',  
                       'xgboost',  
                       'catboost', 
                       'StackingClassifier']): 
   
    scores = model_selection.cross_val_score(clf, X, y,  
                                              cv=5, scoring='accuracy') 
    print("Accuracy: %0.2f (+/- %0.2f) [%s]"  
          % (scores.mean(), scores.std(), label)) 
'''


sclf.fit(X_train, y_train)
y_pred=sclf.predict(X_test)

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

