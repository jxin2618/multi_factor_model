# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 14:16:31 2021

@author: J Xin
"""
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(r'D:\Xin\Program\Alpha_Trading\code\Factors')
sys.path.append(r'D:\Xin\Program\Alpha_Trading\pypackage')
from FactorStats import FactorStat
import matplotlib.pyplot as plt
from datapro import statspro
import json
import warnings
warnings.filterwarnings("ignore")
#%%
os.chdir(r'D:\Xin\Program\Alpha_Trading\data\raw_data\returns')
returns = pd.read_excel('rtn.xlsx', index_col=[0,1])
os.chdir(r'D:\Xin\Program\Alpha_Trading\data\intermediate_data\factors')
factors = pd.read_excel('alpha_001.xlsx', index_col=[0, 1])




class GroupAgent(object):

    
    @classmethod
    def quantile_grouping(cls, data, g=3, gtype=0, label='g',
                          duplicates='drop', **kwargs):
        """Grouping by quantile.

        params:
        -------
        data: series
        g: int
            nums of groups
        gtype: 0 or 1
            only used when g=3
            * 0: 0-0.333, 0.333-0.666, 0.666-1
            * 1: 0-0.3, 0.3-0.7, 0.7-1
        duplicates : {default 'raise', 'drop'}, optional
            If bin edges are not unique, raise ValueError or drop non-uniques.

        Returns:
        --------
        res: series(Categories)

        """
        glist = list(pd.np.arange(0, 1, 1/(g))) + [1]
        if gtype != 0 and g == 3:
            glist = [0, 0.3, 0.7, 1]
        glabel = ['{l}{i}'.format(l=label, i=i+1)
                  for i in pd.np.arange(len(glist)-1)]
        res = pd.cut(x=data, bins=g, labels=glabel, duplicates=duplicates)
        # res = pd.qcut(x=data, q=glist, labels=glabel, duplicates=duplicates)
        
        return res

    @classmethod
    def get_group(cls, factors, group_num):
        """get group by quantile.

        params:
        -------
        factors: DataFrame
            * index: ['date', 'sec_code']
            * column: factor value

        Returns:
        --------
        group:  DataFrame
            * index: ['date', 'sec_code']
            * column: 'group'

        """
        group = factors.groupby('date').transform(lambda x : cls.quantile_grouping(x.rank(), g=group_num))
        group.columns = ['group']
        return group
    
    @classmethod
    def get_return_for_holding_position(cls, returns, holding_days):
        '''
        get return for a fixed holding days

        Parameters
        ----------
        returns:  DataFrame
            * index: ['date', 'sec_code']
            * column: ['open', 'ret']
        holding_days : int

        Returns
        -------
        return_for_holding_days : DataFrame
            * index: ['date', 'sec_code']
            * column: ['open', 'ret_nd'], where n is the integer. 
        '''
        
        return_for_holding_days = returns.groupby(by='sec_code')['open'].transform(lambda x: x /  x.shift(periods=holding_days, axis=0)  - 1).rename('ret_{}d'.format(holding_days))
      
        return return_for_holding_days
    
    @classmethod
    def split_capital(cls, index_date, N):
        '''
        
        split the capital in N groups
        Parameters
        ----------
        index_date : TYPE
            DESCRIPTION.
        N : TYPE
            DESCRIPTION.

        Returns
        -------
        capital_root : dict
            * key: Timestamp, 'date'.
            * value: int, 'capital_root_group'

        '''
        
        
        
        capital_root = dict(zip(index_date, [i % N + 1 for i in range(len(index_date))]))
        return capital_root
    
    @classmethod
    def get_group_return(cls, group, returns, holding_days=5):
        """
        Get group return of specific holding days without dependence on the root
        by splitting the capital into N(=the value of holding days) equal parts 
        and making positions on each trading day with the 1/N * total capital

        Parameters
        ----------
        group:  DataFrame
            * index: ['date', 'sec_code']
            * column: 'group'
        returns:  DataFrame
            * index: ['date', 'sec_code']
            * column: ['open', 'ret']
        holding_days : int, optional
            The default is 5.

        Returns
        -------
        group_net_value_dict : dict
            * key: str, group name
            * value: Series
                - index='date'
                - value='new_value'

        """
        group2 = group.copy() 
        return_for_holding_days = cls.get_return_for_holding_position(returns, holding_days).dropna()
        group2['lagged_group'] = group2.groupby('sec_code')['group'].transform(lambda x: x.shift(periods=holding_days + 1, axis=0))
        df = pd.merge(group2, return_for_holding_days, on=['date', 'sec_code'], how='right').fillna(method='ffill').dropna()
        # 假如调仓周期为N，将资金N等分， 依次间隔N个交易日建仓，将每个通道的净值相加得到策略的总净值
        group_return = df.groupby(['lagged_group', 'date'])['ret_{}d'.format(holding_days)].mean()
        index_date = group_return.index.get_level_values('date').unique().to_list()
        capital_root = cls.split_capital(index_date, holding_days)
        # 获得每个交易日的分组路径
        tmp = group_return.reset_index()
        tmp['capital_root'] = tmp['date'].map(capital_root)
        tmp = tmp.set_index(['date', 'lagged_group', 'capital_root'])
        # 获得每个因子分组每个交易路径的净值
        capital_return = tmp.groupby(['lagged_group', 'capital_root'], as_index=1).transform(lambda x: (x + 1).cumprod()).rename(columns={'ret_{}d'.format(holding_days): 'net_value'})
        group_names = capital_return.index.get_level_values('lagged_group').unique().to_list()
        group_net_value_dict = {}
        
        for group_name in group_names:
            l = [capital_return.xs((group_name, i), level=[1,2]) for i in range(1, holding_days + 1)]
            net_value = pd.concat(l, axis=1).fillna(method='ffill').fillna(1)
            net_value['group_net_value'] = net_value.mean(axis=1)
            group_net_value_dict[group_name] = net_value['group_net_value']
        
        return group_net_value_dict
         
    
    @classmethod
    def compare_group_return_for_diff_holding_position(cls, group, returns, holding_periods=[1,2,3,5,10,20]):
        '''
        

        Parameters
        ----------
        group:  DataFrame
            * index: ['date', 'sec_code']
            * column: 'group'
        returns:  DataFrame
            * index: ['date', 'sec_code']
            * column: ['open', 'ret']
        holding_periods : list, optional
            holding days going to compare. The default is [1,2,3,5,10,20].

        Returns
        -------
        None.

        '''
        res = {}
        for day in holding_periods:
            key = 'holding_{}d'.format(day)
            group_return_for_current_holding_day = cls.get_group_return(group, returns, holding_days=day)
            res[key] = group_return_for_current_holding_day

        return res


    @classmethod
    def plot_annualized_return(cls, group_dict, alpha_name):
        """
        比较各组年化收益

        Parameters
        ----------
        group_dict : dict
            * key: holding day
            * value: dict, group return
                - key: group name
                - value: timeseries for net value

        Returns
        -------
        None.

        """
        fig = plt.figure(figsize=(12,8))
        graph_index = 0
        for holding_day, group_returns  in group_dict.items():
            dict_annual_ret = {}
            ax = fig.add_subplot(3, 2, 1+graph_index)
            graph_index += 1
            for group_name, returns in group_returns.items():
                dict_annual_ret[group_name] = returns.iloc[-1]**(252/len(returns)) - 1
            ax.bar(dict_annual_ret.keys(), dict_annual_ret.values())   
            ax.set_ylabel('annualized_return')
            ax.set_xlabel('group')
            ax.set_title('annualized_return for holding {}'.format(holding_day))
        fig.tight_layout()
        plt.savefig('{}_annualized_return.png'.format(alpha_name))
        plt.close('all')
    
    @classmethod
    def plot_net_value(cls, group_dict, alpha_name):
        """
        比较各组年化收益

        Parameters
        ----------
        group_dict : dict
            * key: holding day
            * value: dict, group return
                - key: group name
                - value: timeseries for net value

        Returns
        -------
        None.

        """
        fig = plt.figure(figsize=(12,8))
        graph_index = 0
        for holding_day, group_returns  in group_dict.items():
            ax = fig.add_subplot(3, 2, 1+graph_index)
            graph_index += 1
            for group_name, returns in group_returns.items():
                ax.plot(returns.index, returns.values, label=group_name)        
            ax.set_ylabel('net_value')
            ax.set_xlabel('group')
            ax.set_title('net value for holding {}'.format(holding_day))
            plt.legend()
            plt.grid('on')
        fig.tight_layout()
        plt.savefig('{}_net_value.png'.format(alpha_name))
        plt.close('all')
    
    
    
class Main(object):
    
    def __init__(self, factors, returns, alpha_name):
        self.factors = factors
        self.returns = returns
        self.alpha_name = alpha_name
   
        
    def main(self):
        group = GroupAgent.get_group(self.factors, 5)
        group_dict = GroupAgent.compare_group_return_for_diff_holding_position(group, self.returns)
        os.chdir(r'D:\Xin\Program\Alpha_Trading\data\intermediate_data\factors_evaluation\group_return')
        for holding_day, group_returns in group_dict.items():
            df = pd.DataFrame(group_returns)
            df.to_hdf('{}.h5'.format(self.alpha_name), key=holding_day)
        os.chdir(r'D:\Xin\Program\Alpha_Trading\figure\group_test\annualized_return_bar')
        GroupAgent.plot_annualized_return(group_dict, self.alpha_name)
        os.chdir(r'D:\Xin\Program\Alpha_Trading\figure\group_test\net_value_curve')
        GroupAgent.plot_net_value(group_dict, self.alpha_name)
            
            
         
         
         