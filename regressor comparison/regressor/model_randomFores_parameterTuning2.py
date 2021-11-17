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

500颗树，no StandardScaler,深度6
MAE 16.479017115102845
score: 0.05721145493564076

500颗树，no StandardScaler,深度3
MAE 19.399696242430217
score: 0.04902033775973862

best param: {'min_samples_split': 80, 'max_depth': 12}
MAE 15.497803824172811
score: 0.06061412844143449

best param: {'max_depth': 14}， 100颗树
MAE 15.391470303061913
score: 0.06100733988537933

best param: {'min_samples_split': 12}  100颗树
MAE 15.371690939635355
score: 0.06108104554912107

best param: {'bootstrap': True}
MAE 15.423722666666745
score: 0.060887535688213965

best param: {'min_samples_leaf': 7}
MAE 15.363766202982903
score: 0.06111062622110264

best param: {'min_samples_leaf': 11}
MAE 15.373985827510703
score: 0.06107248476542915
"""
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
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

'''
st = StandardScaler()
X_transform = st.fit_transform(X)
'''
#划分训练集和测试集
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=42)
#X_train, X_test, y_train, y_test = model_selection.train_test_split(X_transform, y, test_size=0.3, random_state=42)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 50)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(2, 50, num =24)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split =[10,20,30,50,80,100]
# Minimum number of samples required at each leaf node
min_samples_leaf = range(1,100,10)
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {#'n_estimators': n_estimators,
               #'max_features': max_features,
               'max_depth': max_depth,
               #'min_samples_split': min_samples_split,
               #'min_samples_leaf': min_samples_leaf,
               #'bootstrap': bootstrap
               }

rf = RandomForestRegressor(n_estimators=100) 
#如果n_jobs=-1，那么机器上所有的核都会被使用
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, verbose=2, random_state=42, n_jobs = -1)

rf_random.fit(X_train,y_train)
print("best param:",rf_random.best_params_)

y_pred = rf_random.predict(X_test)
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




