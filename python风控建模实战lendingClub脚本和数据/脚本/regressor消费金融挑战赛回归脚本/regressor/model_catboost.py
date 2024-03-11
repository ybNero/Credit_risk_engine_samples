# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 09:39:29 2018
@author: zhi.li

2019/2/4
默认（一个月前）
MAE 15.363469865985705
score: 0.061111732914219646

1000棵树
MAE 15.421561457184445
score: 0.060895548977317214


2019/2/14
默认2
MAE 15.477598278622109
score: 0.06068845611422577

rf.fit(X_train, y_train,cat_features=[0,2,3,4,11,14,16,17,18,19,20])
MAE 15.4224028025341
score: 0.06089242920321577

readFileName="data1.xlsx"
耗时total: 2m 40s
MAE 15.414950464610982
score: 0.0609200741821245

最高分
readFileName="data1.xlsx",rf=cb.CatBoostRegressor(loss_function="RMSE"),rf.fit(X_train, y_train,cat_features=[0,2,3,4,11,14,16,17,18,19,20])
耗时total: 5m 26s 
MAE 15.209466193753132
score: 0.06169234619122646

loss_function="RMSE"
MAE 15.369204131711438
score: 0.06109032497571082

loss_function="MAE"
MAE 603.0985983304904
score: 0.0016553589145275912
"""
import catboost as cb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

#readFileName="data.xlsx"
readFileName="data1.xlsx"
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

#随机森林模型
rf=cb.CatBoostRegressor(loss_function="RMSE")
#rf=cb.CatBoostClassifier(cat_features=None)
#rf.fit(X_train, y_train)
rf.fit(X_train, y_train,cat_features=[0,2,3,4,11,14,16,17,18,19,20])

y_pred = rf.predict(X_test)
#rms = (np.mean((y - y_pred)**2))**0.5
MAE=sum(abs(y_test - y_pred))/len(y_test)
score=1/(1+MAE)
#print ("RF RMS", rms)
print("MAE",MAE)
print("score:",score)




'''
feature_importances=rf.feature_importances_
names=X.columns
list_feature_importances=list(zip(feature_importances,names))
df_feature_importances=pd.DataFrame(list_feature_importances)
df_feature_importances.to_excel("catboost变量信息增益.xlsx")

n_features=X.shape[1]
plt.barh(range(n_features),rf.feature_importances_,align='center')
plt.yticks(np.arange(n_features),X.columns)
plt.title("catboost")
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()

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

