# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 13:16:22 2021

@author: J Xin
"""
import os
import pandas as pd

os.chdir(r'D:\Xin\Program\Alpha_Trading\data\intermediate_data\factors_evaluation\spearman_ic\ic_report')
file_chdir = os.getcwd()


# compare ic ir
# global file_list
file_list = []
for root, dirs, files in os.walk(file_chdir):
    for file in files:
        if os.path.splitext(file)[1] == '.xlsx':
            file_list.append(file)

l = []
for file in file_list:
    os.chdir(r'D:\Xin\Program\Alpha_Trading\data\intermediate_data\factors_evaluation\spearman_ic\ic_report')
    df = pd.read_excel(file, index_col=0)['IC'].rename(file[:-15])
    l.append(df)

compare_ic = pd.concat(l, axis=1).T
os.chdir(r'D:\Xin\Program\Alpha_Trading\data\intermediate_data\factors_evaluation\compare_factors')
compare_ic.to_excel('compare_ic.xlsx')

#%%
# compare group return
os.chdir(r'D:\Xin\Program\Alpha_Trading\data\intermediate_data\factors_evaluation\group_return')
file_chdir = os.getcwd()

# global file_list
file_list = []
for root, dirs, files in os.walk(file_chdir):
    for file in files:
        if os.path.splitext(file)[1] == '.h5':
            file_list.append(file)

l = []
for file in file_list:
    os.chdir(r'D:\Xin\Program\Alpha_Trading\data\intermediate_data\factors_evaluation\spearman_ic\ic_report')
    df = pd.read_hdf(file, index_col=0)['IC'].rename(file[:-15])
    l.append(df)
