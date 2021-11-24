# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 23:34:02 2021

@author: J Xin
"""

import pandas as pd
import os
import sys
sys.path.append(r'D:\Xin\Program\Alpha_Trading\code\Factors')
sys.path.append(r'D:\Xin\Program\Alpha_Trading\pypackage')
import matplotlib.pyplot as plt
from GroupTest import GroupAgent, Main

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
            
for file in file_list[-6:]:
    print(file)
    os.chdir(r'D:\Xin\Program\Alpha_Trading\data\intermediate_data\factors')
    factors = pd.read_excel(file, index_col=[0,1])
    factors = factors.dropna()
    alpha_name = file.split('.')[0]
    Main(factors, returns, alpha_name).main()
    
  