# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 20:15:17 2018
作者：231469242@qq.com
python风控建模实战lendingClub
https://ke.qq.com/course/3063950?tuin=dcbf0ba

accuracy on the training subset:0.992
accuracy on the test subset:0.990
test data:
model accuracy is: 0.989720693593604
model precision is: 0.9294871794871795
model sensitivity is: 0.27358490566037735
f1_score: 0.4227405247813411
AUC: 0.8633646722769536
good classifier
gini 0.7267293445539071
ks value:0.5602


stacking2
accuracy
Accuracy: 0.95 (+/- 0.06) [StackingClassifier]
scores
Out[17]: array([0.85772358, 0.98845902, 0.98887954])

auc
3-fold cross validation:
Accuracy: 0.83 (+/- 0.03) [LGBMClassifier]
Accuracy: 0.87 (+/- 0.02) [CatBoostClassifier]
Accuracy: 0.83 (+/- 0.05) [XGBClassifier]
Accuracy: 0.70 (+/- 0.12) [StackingClassifier]

accuracy
3-fold cross validation:
accuracy: 0.94 (+/- 0.07) [LGBMClassifier]
accuracy: 0.96 (+/- 0.05) [CatBoostClassifier]
accuracy: 0.95 (+/- 0.06) [XGBClassifier]
accuracy: 0.94 (+/- 0.07) [StackingClassifier]-LGBMClassifier

3-fold cross validation:
accuracy: 0.94 (+/- 0.07) [LGBMClassifier]
accuracy: 0.96 (+/- 0.05) [CatBoostClassifier]
accuracy: 0.95 (+/- 0.06) [XGBClassifier]
accuracy: 0.94 (+/- 0.07) [StackingClassifier]-CatBoostClassifier


3-fold cross validation:

accuracy: 0.94 (+/- 0.07) [LGBMClassifier]
accuracy: 0.96 (+/- 0.05) [CatBoostClassifier]
accuracy: 0.95 (+/- 0.06) [XGBClassifier]
accuracy: 0.95 (+/- 0.06) [StackingClassifier] --LogisticRegression

"""
from sklearn import model_selection 
from sklearn.linear_model import LogisticRegression 
import lightgbm as lgb
import catboost as cb
from xgboost import XGBClassifier
from mlxtend.classifier import StackingClassifier 
import pandas as pd


#字符串转换为数值型，删除空缺值100%变量
#readFileName="data1.xlsx"
#删除信息增益低或0的变量，单一占比高变量，入模型变量99个，类别变量处理
readFileName="data_Q5_filter.xlsx"
#读取excel
data=pd.read_excel(readFileName)
X=data.loc[:,"installment":"emp_length"]
y=data["target"]
clf1 = lgb.LGBMClassifier() 
clf2 = cb.CatBoostClassifier(verbose=False)
#eval_metric=['logloss','auc','error']
clf3 = XGBClassifier(missing=None,eval_metric='logloss',use_label_encoder=False) 
lr = LogisticRegression() 
classifier= StackingClassifier(classifiers=[clf1, clf2, clf3],  
                          meta_classifier=lr )  


print('3-fold cross validation:\n')

for clf, label in zip([clf1, clf2, clf3, classifier], 
                      ['LGBMClassifier', 
                       'CatBoostClassifier', 
                       'XGBClassifier',
                       'StackingClassifier']):
  
    #scoring='accuracy',scoring='roc_auc'
    scores1 = model_selection.cross_val_score(clf, X, y, 
                                              cv=3, scoring='roc_auc')
    print("AUC: %0.2f (+/- %0.2f) [%s]" 
          % (scores1.mean(), scores1.std(), label))

#保存模型
import pickle
save_classifier = open("stacking.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

#list_coef = list(classifier.coef_[0])

'''
#导入分类器
classifier_f = open("stacking.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()
'''