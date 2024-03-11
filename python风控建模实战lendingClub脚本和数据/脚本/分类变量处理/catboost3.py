# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 20:15:17 2018
@author: 231469242@qq.com
test
未做分类变量处理
accuracy on the training subset:0.991
accuracy on the test subset:0.990
test data:
model accuracy is: 0.9899543141937494
model precision is: 0.9386503067484663
model sensitivity is: 0.28867924528301886
f1_score: 0.44155844155844154
AUC: 0.8608825524586305
good classifier
gini 0.7217651049172611
ks value:0.5632
total: 58.6s
"""
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import catboost as cb
import pandas as pd
import numpy as np
#混淆矩阵计算
from sklearn import metrics
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

#字符串转换为数值型，删除空缺值100%变量
#readFileName="data1.xlsx"
#删除信息增益低或0的变量，单一占比高变量，入模型变量99个，类别变量处理
readFileName="data_类别变量.xlsx"
#读取excel
data=pd.read_excel(readFileName)

X=data.ix[:,"installment":"purpose"]
y=data["target"]
train_x, test_x, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=0)
 
cb = cb.CatBoostClassifier()
cb.fit(train_x, y_train)

#分类变量有空缺值，程序无法运行
#cb.fit(train_x, y_train,cat_features=[3,6,7,10,11,12,22,33,36,97])

 
print("accuracy on the training subset:{:.3f}".format(cb.score(train_x,y_train)))
print("accuracy on the test subset:{:.3f}".format(cb.score(test_x,y_test)))

y_true=y_test
y_pred=cb.predict(test_x)
accuracyScore = accuracy_score(y_true, y_pred)
print("test data:")
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

#获取所有x数据的预测概率,包括好客户和坏客户，0为好客户，1为坏客户
probablity_list=cb.predict_proba(test_x)
#获取所有x数据的坏客户预测概率
pos_probablity_list=[i[1] for i in probablity_list]


def AUC(y_true, y_scores):
    auc_value=0
    #auc第二种方法是通过fpr,tpr，通过auc(fpr,tpr)来计算AUC
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores, pos_label=1)
    auc_value= auc(fpr,tpr) ###计算auc的值 
    #print("fpr:",fpr)
    #print("tpr:",tpr)
    #print("thresholds:",thresholds)
    if auc_value<0.5:
        auc_value=1-auc_value
    return auc_value

def Draw_roc(auc_value):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pos_probablity_list, pos_label=1)
    #画对角线 
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Diagonal line') 
    plt.plot(fpr,tpr,label='ROC curve (area = %0.2f)' % auc_value) 
    plt.title('ROC curve')  
    plt.legend(loc="lower right")

#评价AUC表现
def AUC_performance(AUC):
    if AUC >=0.7:
        print("good classifier")
    if 0.7>AUC>0.6:
        print("not very good classifier")
    if 0.6>=AUC>0.5:
        print("useless classifier")
    if 0.5>=AUC:
        print("bad classifier,with sorting problems")
        
#Gini
def Gini(auc):
    gini=2*auc-1
    return gini

### 计算KS值
def KS(df, thescore, target):
    '''
    :param df: 包含目标变量与预测值的数据集
    :param score: 得分或者概率
    :param target: 目标变量
    :return: KS值
    '''
    #根据score分数和目标变量，对score分数计数
    total = df.groupby([thescore])[target].count()
    '''
    score
    0.00001    4
    0.00005    7
    0.00006    4
    0.00007    1
    0.00008    1
    '''
    bad = df.groupby([thescore])[target].sum()
    all = pd.DataFrame({'total':total, 'bad':bad})
    all['good'] = all['total'] - all['bad']
    all[thescore] = all.index
    all.index = range(len(all))
    all = all.sort_values(by=thescore,ascending=True)
    #坏客户总数
    num_bad=all['bad'].sum()
    #好客户总数
    num_good= all['good'].sum()
    #累计坏客户概率
    all['badCumRate'] = all['bad'].cumsum() / num_bad
    #累计好客户概率
    all['goodCumRate'] = all['good'].cumsum() /num_good
    #坏客户-好客户概率的序列
    ks_array = all.apply(lambda x: abs(x.badCumRate - x.goodCumRate), axis=1)
    #坏客户-好客户概率的序列的最大值就是ks值
    ks=max(ks_array)
    return ks
        
#Auc验证，数据采用测试集数据
auc_value=AUC(y_test, pos_probablity_list)
print("AUC:",auc_value)
#评价AUC表现
AUC_performance(auc_value)
#绘制ROC曲线
Draw_roc(auc_value)


df=pd.DataFrame({'score':pos_probablity_list, 'target':y_test})
#基尼系数
gini=Gini(auc_value)
print ("gini",gini)  
 
#计算KS
ks = KS(df,'score','target')
print("ks value:%.4f"%ks)



feature_importances=cb.feature_importances_
names=X.columns
list_feature_importances=list(zip(feature_importances,names))
df_feature_importances=pd.DataFrame(list_feature_importances)
#df_feature_importances.to_excel("catboost_110变量信息增益.xlsx")
df_feature_importances.to_excel("catboost_变量重要性.xlsx")

n_features=X.shape[1]
plt.barh(range(n_features),cb.feature_importances_,align='center')
plt.yticks(np.arange(n_features),X.columns)
plt.title("catboost")
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()
plt.savefig("featureImportance.png")


'''
#保存模型
import pickle
save_classifier = open("catboost5.pickle","wb")
pickle.dump(cb, save_classifier)
save_classifier.close()

#导入分类器
classifier_f = open("catboost5.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()
'''








