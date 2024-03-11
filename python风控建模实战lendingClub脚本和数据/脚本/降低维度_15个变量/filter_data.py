# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 21:12:52 2020

@author: Administrator
"""
import catboost as cb
import pandas as pd


#字符串转换为数值型，删除空缺值100%变量
#readFileName="data1.xlsx"
#删除信息增益低或0的变量，单一占比高变量，入模型变量99个，类别变量处理
file1="data_Q4.xlsx"
#file1="data_Q3.xlsx"
file2="变量.xlsx"
#读取excel
data=pd.read_excel(file2)
list_variables=list(data.columns)

df=pd.read_excel(file1)
df1=df.filter(items=list_variables)
#df1.to_excel("data_Q3_filter.xlsx")
#df1.to_excel("data_Q4_filter.xlsx")
df1.to_excel("data_Q5_filter.xlsx")
'''
X=data.ix[:,"loan_amnt":"disbursement_method"]
y=data["target"]

'''