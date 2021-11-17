# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 09:24:11 2021

@author: 231469242@qq.com

smote算法(ratio 5：5)：
测试集
accuracy on subset:0.995
model accuracy is: 0.9954312648944714
model precision is: 0.9999734169812324
model sensitivity is: 0.990885862550378
f1_score: 0.9954088990619336
AUC: 0.9996416626973991
"""

from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
#过采样算法
from imblearn.combine import SMOTEENN
from imblearn.datasets import make_imbalance
from imblearn.over_sampling import SMOTE, ADASYN
import catboost as cb

from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
 
readFileName="data_Q5_filter.xlsx"
#读取excel
data=pd.read_excel(readFileName)
features=data.loc[:,"installment":"emp_length"]
x=features.values
y=data["target"]
print(y.value_counts())
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp.fit(x)

#缺失数据处理
#x1是处理缺失数据后的值
x1=imp.transform(x)
#过采样，好坏客户平分
sm = SMOTE(random_state=42)
#0.9:0.1比例
#ratio = {0: 54045, 1:5404}
 
#sm=ADASYN(ratio=ratio)
#sm = SMOTE(ratio ="auto")
#sm = SMOTEENN(ratio = ratio)
X_resampled, y_resampled = sm.fit_sample(x1, y)
print(y_resampled.value_counts())

train_x, test_x, y_train, y_test=train_test_split(X_resampled,y_resampled,test_size=0.3,random_state=0)
 
#建模和验证
#树的算法能自动处理缺失值
clf=cb.CatBoostClassifier()
clf.fit(X_resampled, y_resampled)
 
print("fit success")
#y_pred=clf.predict(x)
y_pred=clf.predict(test_x)
#测试数据集
print("accuracy on subset:{:.3f}".format(clf.score(test_x, y_test)))
 
df_y_pred=pd.DataFrame(y_pred)
df_y_pred.to_excel("predict.xlsx")
 
proba_bad=clf.predict_proba(test_x)
y_scores=pd.DataFrame(proba_bad)[1]
y_scores=np.array(y_scores)
 
y_true =y_test
 
accuracyScore = accuracy_score(y_true, y_pred)
print('model accuracy is:',accuracyScore)
 
#precision,TP/(TP+FP) （真阳性）/（真阳性+假阳性）
precision=precision_score(y_true, y_pred)
print('model precision is:',precision)
 
#recall（sensitive）敏感度，(TP)/（TP+FN）
sensitivity=recall_score(y_true, y_pred)
print('model sensitivity is:',sensitivity)
  
#F1 = 2 x (精确率 x 召回率) / (精确率 + 召回率)
#F1 分数会同时考虑精确率和召回率，以便计算新的分数。可将 F1 分数理解为精确率和召回率的加权平均值，其中 F1 分数的最佳值为 1、最差值为 0：
f1Score=f1_score(y_true, y_pred)
print("f1_score:",f1Score)
 
 
#auc分数
#auc分数有两种计算方式，第一种是根据目标变量y_true,预测分数/预测概率y_socres,通过roc_auc_score(y_true, y_scores)计算AUC
AUC=roc_auc_score(y_true, y_scores)
print("AUC:",AUC)
#auc第二种方法是通过fpr,tpr，通过auc(fpr,tpr)来计算AUC
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores, pos_label=1)
AUC1 = auc(fpr,tpr) ###计算auc的值
           
#绘制ROC曲线
#画对角线
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Diagonal line')
plt.plot(fpr,tpr,label='ROC curve (area = %0.2f)' % AUC)
plt.title('ROC curve') 
plt.legend(loc="lower right")   
