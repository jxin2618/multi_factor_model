# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 16:16:41 2021

@author: J Xin
"""

import pandas as pd
import numpy as np
import datetime
import time
import statsmodels.api as sm
from scipy.stats import spearmanr 
import math
from statsmodels import regression
import matplotlib.pyplot as plt
import sys
sys.path.append(r'D:\Xin\Program\Alpha_Trading\pypackage')
from datapro import statspro


class FactorStat(object):
    
    @classmethod
    def _standardize_factor(cls, factors):
        """
        Standardize the factors by removing infinities

        Parameters
        ----------
        factors : DataFrame
            * index: ['date', 'sec_code']
            * columns: factor value

        Returns
        -------
        factors : DataFrame
            * index: ['date', 'sec_code']
            * columns: factor value

        """
        factors = factors.replace([np.inf, -np.inf], np.nan)
        factors = factors.dropna()
        return factors
    
    # @classmethod
    # def _cal_spearman_ic(cls, factors, returns, days=1):
    #     """
    #     calculate the cross sectional spearman IC in different lag period

    #     Parameters
    #     ----------
    #     factors : pd.DataFrame
    #         * columns = ['date', 'sec_code', factor].
    #     returns : pd.DataFrame
    #         * columns = ['date', 'sec_code', 'ret'].
    #     days : int
    #         lagged days for the factor. The default is 1.

    #     Returns
    #     -------
    #     ic : float
    #         cross sectional spearman IC.

    #     """
       
    #     date_index = list(factors.date.unique())
    #     date_index.sort()
    #     ic_df = pd.DataFrame(index=date_index, columns=['IC'])
    #     for i in range(len(date_index)-days-1):
    #         tmp_factor = factors.loc[factors.date == date_index[i], :]
    #         tmp_ret = returns.loc[returns.date == date_index[i + days + 1], :]
    #         factor = tmp_factor.drop(columns=['date'])
    #         ret = tmp_ret.drop(columns=['date'])
    #         df = pd.merge(factor, ret, on='sec_code', how='inner')
    #         ic, p_value = spearmanr(df.iloc[:, 1], df.iloc[:, 2])   # 计算秩相关系数RankIC
    #         ic_df.loc[date_index[i], 'IC']  = ic
    #     return ic_df
    
    @classmethod
    def get_spearman_ic(cls, factors, returns, holding_days=1, lag_days=1):
        """
        get the spearman IC fixed holding periods and lagging periods

        Parameters
        ----------
        factors : DataFrame
            * index: ['date', 'sec_code']
            * columns: factor value
        returns : DataFrame
            * index: ['date', 'sec_code']
            * columns: ['open', 'ret'] 
        holding_days : int, optional
            holding periods. The default is 1.
        lag_days : int, optional
            lagging periods. The default is 1.

        Returns
        -------
        ic_df : DataFrame
            * index: ['date']
            * columns: ['IC'] 

        """
        factors = cls._standardize_factor(factors)
        return_for_holding_days = returns.groupby(by='sec_code')['open'].transform(lambda x: x.shift(periods=-holding_days, axis=0) / x  - 1).rename('ret_{}d'.format(holding_days))
        lagged_returns = return_for_holding_days.groupby(by='sec_code').transform(lambda x: x.shift(periods=-lag_days, axis=0)).rename('ret_{}d_lagged'.format(holding_days))
        df = pd.merge(factors, lagged_returns, on=['date', 'sec_code'], how='inner')
        ic_df = df.groupby('date').apply(lambda x: spearmanr(x.iloc[:, 0], x.iloc[:, 1])[0]).rename('IC').to_frame()
        
        return ic_df
    
    
    @classmethod
    def get_ic_for_diff_hold_positions(cls, factors, returns, holding_period=[1, 2, 3, 5, 10, 20]):
        """
        get the spearman IC a series of different holding periods and fixed lagging period of 1 day

        Parameters
        ----------
        factors : DataFrame
            * index: ['date', 'sec_code']
            * columns: factor value
        returns : DataFrame
            * index: ['date', 'sec_code']
            * columns: ['open', 'ret'] 
        holding_period : list, optional
            holding periods to be compared. The default is [1, 2, 3, 5, 10, 20].

        Returns
        -------
        factor_ic : dictionary
            * key: holding period, like '1day'
            * value: Rank_IC for each day

        """
        factor_ic = {}
        #用于计算不同周期的信息系数
        for i in holding_period:
            print(i)
            ic_alpha_day = FactorStat.get_spearman_ic(factors, returns, holding_days=i)
            factor_ic[str(i)+'day'] = ic_alpha_day
        return factor_ic
    
    @classmethod
    def get_ic_decay(cls, factors, returns, holding_period=[1, 2, 3, 5, 10, 20]):
        '''
        

        Parameters
        ----------

        factors : TYPE
            DESCRIPTION.
        returns : TYPE
            DESCRIPTION.
        holding_period : TYPE, optional
            DESCRIPTION. The default is [1, 2, 3, 5, 10, 20].

        Returns
        -------
        ic_decay : TYPE
            DESCRIPTION.

        '''
        #用于计算不同持仓周期的IC均值的衰减效应
        ic_decay = pd.DataFrame(index=[str(i) + 'day' for i in range(1, 183)], columns=[str(i) + 'd' for i in holding_period])
        for day in holding_period:
            print(day)
            for i in range(1, 183):
                ic = FactorStat.get_spearman_ic(factors, returns, holding_days=day, lag_days=i)['IC'].dropna().mean()
                ic_decay.loc[str(i) + 'day', str(day) + 'd']  = ic
        
        return ic_decay
    
    @classmethod
    def plot_ic_decay(cls, ic_decay, alpha_name):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(ic_decay.shape[1]):
            plt.plot(range(ic_decay.shape[0]), ic_decay.iloc[:, i], label='holding_' + ic_decay.columns[i])
        plt.xlabel('lagging_period')
        plt.ylabel('Rank_IC_mean')
        plt.title('Explore the decay effect')
        plt.tight_layout()
        plt.legend()
        plt.savefig('{}_decay.png'.format(alpha_name))
        plt.close('all')
    
    @classmethod
    def get_ic_ir(cls, factor_ic):
        hold_pos = list(factor_ic.keys())
        alpha_report = pd.DataFrame(index=hold_pos, columns=['IC', 'IC_STD', 'IC_IR', 'IC>0'])
        for period in hold_pos:
            ic = factor_ic[period].dropna()
            l = [ic['IC'].mean(), ic['IC'].std(), ic['IC'].mean()/ic['IC'].std(),
                 (ic['IC'] > 0).value_counts()[True] / ic.shape[0]]
            alpha_report.loc[period,:] = l
    
        return alpha_report
    
    @classmethod
    def plot_ic_for_hold_positions(cls, alpha_report, alpha_name):
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        ax.bar(x=alpha_report.index,height=alpha_report['IC'])
        ax.set_xlabel('holding_period')
        ax.set_ylabel('Rank_IC_mean')
        ax.set_title('Compare Rank_IC for different holding period')
        plt.savefig('{}.png'.format(alpha_name))
        plt.close('all')
    
    @classmethod
    def plot_ic_distribution(cls, factor_ic, alpha_name):
        fig = plt.figure(figsize=(12,8))
        num = len(factor_ic)
        days = list(factor_ic.keys())
        for i in range(num):
            ax = fig.add_subplot(3, 2, 1+i)
            # plt.subplot(3, 2, 1+i)
            ax.hist(factor_ic[days[i]].dropna().values, 50, density=True, facecolor='orange')
            ax.set_xlabel(days[i])
            ax.set_ylabel('Rank_IC')
        plt.tight_layout()
        plt.savefig('{}.png'.format(alpha_name))
        plt.close('all')
    
    @classmethod
    def plot_ic_value(cls, factor_ic, alpha_name, days='1day'):
        data = factor_ic[days]
        plt.figure(dpi=120, figsize=(8, 5))
        fig, axs = plt.subplots(2, 1)
        data['ic_cumsum'] = data['IC'].cumsum()
        data['ic_rolling_21d'] = data['IC'].rolling(21).mean()
        axs[0].fill_between(data.index, data['IC'].fillna(0), label='ic_value')
        axs[1].fill_between(data.index, data['ic_rolling_21d'].fillna(0))
        ax2 = axs[0].twinx()
        ax3 = axs[1].twinx()
        ax2.plot(data.index, data['ic_cumsum'], color='red', label='ic_accsum')
        ax3.plot(data.index, data['ic_cumsum'], color='red', label='ic_accsum')
        plt.xlabel('date')
        axs[0].set_ylabel('ic_value')
        axs[1].set_ylabel('ic_value_rolling_21d')
        ax2.set_ylabel('ic_cumsum')
        ax3.set_ylabel('ic_cumsum')
        plt.title('ic_value_for_holding_{}'.format(days))
        plt.tight_layout()
        plt.savefig('{}_ic.png'.format(alpha_name), dpi=150)
        plt.close('all')
    
    @classmethod
    def plot_ic_accsum(cls, factor_ic, alpha_name):
        """
        plot the accumulate sum of rank_ic for different holding periods

        Parameters
        ----------
        factor_ic : dictionanry
            * key: holding period, like '1day'
            * value: Rank_IC for each day
        alpha_name : str

        Returns
        -------
        None.

        """
        plt.figure(dpi=120, figsize=(8, 5))
        fig, ax = plt.subplots()
        for days in factor_ic.keys():
            data = factor_ic[days]
            data['ic_cumsum'] = data['IC'].cumsum()
            plt.plot(data.index, data['ic_cumsum'], label=days)
        plt.xlabel('date')
        plt.ylabel('ic_cumsum')
        plt.title('ic_accsum_for_diff_holding_position')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig('{}_accsum_ic.png'.format(alpha_name), dpi=150)
        plt.close('all')