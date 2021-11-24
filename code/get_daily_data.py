# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 20:52:49 2021

@author: J Xin
"""

import pandas as pd
import numpy as np
import os
import requests

os.chdir(r'D:\Xin\Program\Alpha_Trading\data\raw_data')
def get_daily_data(start_date, end_date, sec):
    url = 'http://192.168.7.209:6868/constituent'
    res = requests.post(url, json=dict(start=start_date, end=end_date, 
                                       columns=['open', 'high', 'low', 'close', 'settle', 'vwap', 'pre_close',
                                                'volume', 'amount', 'openinterest', 
                                                'sec_name', 'contract_code'], 
                                       sec_code=sec))
    res = pd.DataFrame(res.json())
    res['date'] = res['date'].apply(lambda x: pd.Timestamp(x[0:10]))
    # res = res.set_index('date')
    
    
    return res

security_list = ['CU', 'ZN', 'AL', 'NI', 'PB', 'SN', 'AU', 'AG', 'SS',
               'FG', 'ZC', 'RB', 'HC', 'I', 'J', 'JM', 'SF', 'SM',
               'C', 'CS', 'A', 'B', 'Y', 'M', 'RM', 'OI', 'P', 'CF', 'JD', 'SR', 
               'SP', 'L', 'PP', 'V', 'MA', 'RU', 'TA', 'BU', 'SC', 'FU', 'EG', 'UR', 'NR', 'EB'
               ]
l = []
for security in security_list:
    print(security)
    data = get_daily_data('2010-01-01', '2021-09-30', security+'.CCRI')
    data = data.set_index(['date', 'sec_code'])
    l.append(data)

res = pd.concat(l, axis=0)
res.to_excel('daily_data.xlsx')