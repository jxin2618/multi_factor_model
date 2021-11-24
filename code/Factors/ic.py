# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 21:16:27 2021

@author: J Xin
"""
import pandas as pd
import os
import sys
sys.path.append(r'D:\Xin\Program\Alpha_Trading\code\Factors')
sys.path.append(r'D:\Xin\Program\Alpha_Trading\pypackage')
import matplotlib.pyplot as plt
from FactorStats import FactorStat
from datapro import statspro
import warnings
warnings.filterwarnings("ignore")


# get the return array
os.chdir(r'D:\Xin\Program\Alpha_Trading\data\raw_data\returns')
returns = pd.read_excel('rtn.xlsx', index_col=[0,1])

# get all the factors
os.chdir(r'D:\Xin\Program\Alpha_Trading\data\intermediate_data\factors')
file_chdir = os.getcwd()

# global file_list
file_list = []
for root, dirs, files in os.walk(file_chdir):
    for file in files:
        if os.path.splitext(file)[1] == '.xlsx':
            file_list.append(file)
            
for file in file_list[-80:]:
    print(file)
    os.chdir(r'D:\Xin\Program\Alpha_Trading\data\intermediate_data\factors')
    factors = pd.read_excel(file, index_col=[0,1])
    alpha_name = file.split('.')[0]
    factor_dic = FactorStat.get_ic_for_diff_hold_positions(factors, returns)
    # 输出不同持仓周期下，每日的RankIC值
    os.chdir(r'D:\Xin\Program\Alpha_Trading\data\intermediate_data\factors_evaluation\spearman_ic\ic_value')
    factor_df = pd.concat(list(factor_dic.values()), axis=1)
    factor_df.columns = list(factor_dic.keys())
    factor_df.to_excel('{}_ic_value.xlsx'.format(alpha_name))
    # 计算不同周期下的ICIR
    alpha_report = FactorStat.get_ic_ir(factor_dic)
    os.chdir(r'D:\Xin\Program\Alpha_Trading\data\intermediate_data\factors_evaluation\spearman_ic\ic_report')
    alpha_report.to_excel('{}_ic_report.xlsx'.format(alpha_name))
    # 画图比较不同周期的ICIR
    os.chdir(r'D:\Xin\Program\Alpha_Trading\figure\ic_report\compare_ic_for_hold_positions')
    FactorStat.plot_ic_for_hold_positions(alpha_report, alpha_name)
    # 画图比较不同周期的IC分布
    os.chdir(r'D:\Xin\Program\Alpha_Trading\figure\ic_report\ic_distribution')
    FactorStat.plot_ic_distribution(factor_dic, alpha_name)
    # 画图比较不同周期的IC累计
    os.chdir(r'D:\Xin\Program\Alpha_Trading\figure\ic_report\ic_accsum')
    FactorStat.plot_ic_accsum(factor_dic, alpha_name)
    # 画图每日IC走势
    os.chdir(r'D:\Xin\Program\Alpha_Trading\figure\ic_report\ic_value')
    FactorStat.plot_ic_value(factor_dic, alpha_name)
    # 研究因子衰减效应
    ic_decay = FactorStat.get_ic_decay(factors, returns, holding_period=[1, 5, 10, 20])
    os.chdir(r'D:\Xin\Program\Alpha_Trading\data\intermediate_data\factors_evaluation\spearman_ic\ic_decay')
    ic_decay.to_excel('{}_ic_decay.xlsx'.format(alpha_name))
    # 比较不同因子周期的衰减效应
    os.chdir(r'D:\Xin\Program\Alpha_Trading\figure\ic_report\ic_decay')
    FactorStat.plot_ic_decay(ic_decay, alpha_name)
                            
    
    
    
    
    
    
    
    
    