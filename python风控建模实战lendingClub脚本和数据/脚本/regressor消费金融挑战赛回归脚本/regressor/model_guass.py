# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 15:53:57 2018

@author: zhi.li04

默认参数
GP RMS label    2.98575
GP r^2 score -1.26973055888

核参数改为kernel=1.0 * RBF(length_scale=1) + WhiteKernel(noise_level=1)
RMS    0.651688
dtype: float64
GP r^2 score 0.891869923330719

核参数修改，且正态化后
GP RMS label    0.597042
GP r^2 score 0.9092436176966117
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF

readFileName="data.xlsx"
#读取excel
data=pd.read_excel(readFileName)
x=data.ix[:,"用户实名制是否通过核实":"当月旅游资讯类应用使用次数"]
y=data["score"]

'''
st = StandardScaler()
X_transform = st.fit_transform(x)
 '''
 
#划分训练集和测试集
X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3, random_state=42)
#X_train, X_test, y_train, y_test = model_selection.train_test_split(X_transform, y, test_size=0.3, random_state=42)

#（工作电脑运行会死机）
kernel=1.0 * RBF(length_scale=1) + WhiteKernel(noise_level=1)
#gp = gaussian_process.GaussianProcessRegressor()
#gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
gp = gaussian_process.GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=0,normalize_y=True)
gp.fit(X_train, y_train)

y_pred, sigma = gp.predict(X_test, return_std=True)
#rms = (np.mean((y_test - y_pred)**2))**0.5
#s = np.std(y_test -y_pred)
#print ("GP RMS", rms)
print ("GP r^2 score",r2_score(y_test,y_pred))

MAE=sum(abs(y_test - y_pred))/len(y_test)
score=1/(1+MAE)
#print ("RF RMS", rms)
print("MAE",MAE)
print("score:",score)

'''
plt.scatter(y_train,gp.predict(X_train), label = 'Train', c='blue')
plt.title('GP Predictor')
plt.xlabel('Measured Solubility')
plt.ylabel('Predicted Solubility')
plt.scatter(y_test,gp.predict(X_test),c='lightgreen', label='Test', alpha = 0.8)
plt.legend(loc=4)
plt.savefig('GP Predictor.png', dpi=300)
plt.show()
'''

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