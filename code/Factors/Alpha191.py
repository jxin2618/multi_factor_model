# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 10:40:58 2021

@author: J Xin
"""

import numpy as np
import pandas as pd
import os
from scipy.stats import rankdata
import scipy as sp
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

os.chdir(r'D:\Xin\Program\Alpha_Trading\data\raw_data')

data = pd.read_excel('daily_data.xlsx')
data = data.loc[data.date >= pd.Timestamp('2014-01-01'), :]
benchmark = pd.read_excel('nh0100.xlsx').set_index('date')


class CalAlpha:

    def __init__(self, end_date):
        df = data.loc[data.date <= pd.Timestamp(end_date), :]
        # ####分别取开盘价，收盘价，最高价，最低价，最低价，均价，分钟成交量，当日总成交量#######
        self.open_price = df.pivot(index='date', columns='sec_code', values='open')
        self.close = df.pivot(index='date', columns='sec_code', values='close')
        self.low = df.pivot(index='date', columns='sec_code', values='low')
        self.high = df.pivot(index='date', columns='sec_code', values='high')
        self.avg_price  = df.pivot(index='date', columns='sec_code', values='vwap')
        self.prev_close = df.pivot(index='date', columns='sec_code', values='pre_close')
        self.volume = df.pivot(index='date', columns='sec_code', values='volume')
        self.amount = df.pivot(index='date', columns='sec_code', values='amount')
        self.benchmark_open_price = benchmark.loc[:, 'open']
        self.benchmark_close_price = benchmark.loc[:, 'close']
        
      #########################################################################
    
    def func_rank(self, na):
        return rankdata(na)[-1]/rankdata(na).max()
    
    def func_decaylinear(self, na):
        n = len(na)
        decay_weights = np.arange(1,n+1,1) 
        decay_weights = decay_weights / decay_weights.sum()

        return (na * decay_weights).sum()
    
    def func_highday(self, na):
        return len(na) - na.values.argmax()

    def func_lowday(self, na):
        return len(na) - na.values.argmin()

    def func_regbeta(self, na):
        n = len(na)
        x = np.arange(1, n + 1, 1)  # 等差S
        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(x, na)
        if p_value > 0.05 or p_value == np.nan :
            slope = 0

        return slope    
    #############################################################################
    
    def alpha_001(self):
        data1 = self.volume.diff(periods=1).rank(axis=1,pct=True)
        data2 = ((self.close - self.open_price)/self.open_price).rank(axis=1,pct=True)
        alpha = -data1.iloc[-6:,:].corrwith(data2.iloc[-6:,:]).dropna()
        alpha=alpha.dropna()
        return alpha
    
    def alpha_002(self):
        ##### -1 * delta((((close-low)-(high-close))/((high-low)),1))####
        result=((self.close-self.low)-(self.high-self.close))/((self.high-self.low)).diff()
        m=result.iloc[-1,:].dropna() 
        alpha=m[(m<np.inf)&(m>-np.inf)]      
        return alpha.dropna() 


    ################################################################
    def alpha_003(self):  
        delay1 = self.close.shift()   
        condtion1 = (self.close == delay1)
        condition2 = (self.close > delay1)
        condition3 = (self.close < delay1)

        part2 = (self.close-np.minimum(delay1[condition2],self.low[condition2])).iloc[-6:,:] #取最近的6位数据
        part3 = (self.close-np.maximum(delay1[condition3],self.low[condition3])).iloc[-6:,:] 

        result=part2.fillna(0)+part3.fillna(0)
        alpha=result.sum()
        return alpha.dropna()

    ########################################################################
    def alpha_004(self):
        condition1=(self.close.rolling(8).std()<self.close.rolling(2).sum()/2)
        condition2=(self.close.rolling(2).sum()/2<(self.close.rolling(8).sum()/8-self.close.rolling(8).std()))
        condition3=(1<=self.volume/self.volume.rolling(20).mean()) 
        condition3

        indicator1=pd.DataFrame(np.ones(self.close.shape),index=self.close.index,columns=self.close.columns)#[condition2]
        indicator2=-pd.DataFrame(np.ones(self.close.shape),index=self.close.index,columns=self.close.columns)#[condition3]

        part0=self.close.rolling(8).sum()/8
        part1=indicator2[condition1].fillna(0)
        part2=(indicator1[~condition1][condition2]).fillna(0)
        part3=(indicator1[~condition1][~condition2][condition3]).fillna(0)
        part4=(indicator2[~condition1][~condition2][~condition3]).fillna(0)

        result=part0+part1+part2+part3+part4
        alpha=result.iloc[-1,:]
        return alpha.dropna()

    ################################################################
    def alpha_005(self):
        ts_volume=(self.volume.iloc[-7:,:]).rank(axis=0,pct=True)
        ts_high=(self.high.iloc[-7:,:]).rank(axis=0,pct=True)
        corr_ts=ts_high.rolling(5).corr(ts_volume)
        alpha=corr_ts.max().dropna()
        alpha=alpha[(alpha<np.inf)&(alpha>-np.inf)] 
        return alpha 

    ###############################################################
    def alpha_006(self):
        condition1=((self.open_price*0.85+self.high*0.15).diff(4)>1)
        condition2=((self.open_price*0.85+self.high*0.15).diff(4)==1)
        condition3=((self.open_price*0.85+self.high*0.15).diff(4)<1)
        indicator1=pd.DataFrame(np.ones(self.close.shape),index=self.close.index,columns=self.close.columns) 
        indicator2=pd.DataFrame(np.zeros(self.close.shape),index=self.close.index,columns=self.close.columns)
        indicator3=-pd.DataFrame(np.ones(self.close.shape),index=self.close.index,columns=self.close.columns) 
        part1=indicator1[condition1].fillna(0)
        part2=indicator2[condition2].fillna(0)
        part3=indicator3[condition3].fillna(0)
        result=part1+part2+part3
        alpha=(result.rank(axis=1,pct=True)).iloc[-1,:]    #cross section rank
        return alpha.dropna()

    ##################################################################
    def alpha_007(self):
        part1=(np.maximum(self.avg_price-self.close,3)).rank(axis=1,pct=True)
        part2=(np.minimum(self.avg_price-self.close,3)).rank(axis=1,pct=True)
        part3=(self.volume.diff(3)).rank(axis=1,pct=True)
        result=part1+part2*part3
        alpha=result.iloc[-1,:]
        return alpha.dropna()

    ##################################################################
    def alpha_008(self):
        temp=(self.high+self.low)*0.2/2+self.avg_price*0.8
        result=-temp.diff(4)
        alpha=result.rank(axis=1,pct=True)
        alpha=alpha.iloc[-1,:]
        return alpha.dropna()

    ##################################################################
    def alpha_009(self):
        temp=(self.high+self.low)*0.5-(self.high.shift()+self.low.shift())*0.5*(self.high-self.low)/self.volume #计算close_{i-1}
        result=temp.ewm(alpha=2/7).mean()
        alpha=result.iloc[-1,:]
        return alpha.dropna()

    ##################################################################
    def alpha_010(self):
        ret=self.close.pct_change()
        condtion=(ret<0)
        part1=(ret.rolling(20).std()[condtion]).fillna(0)
        part2=(self.close[~condtion]).fillna(0)
        result=np.maximum((part1+part2)**2,5)
        alpha=result.rank(axis=1,pct=True)
        alpha=alpha.iloc[-1,:]
        return alpha.dropna()

    ##################################################################
    def alpha_011(self):
        temp=((self.close-self.low)-(self.high-self.close))/(self.high-self.low)
        result=temp*self.volume
        alpha=result.iloc[-6:,:].sum()
        return alpha.dropna()


    ##################################################################
    def alpha_012(self):
        vwap10=self.avg_price.rolling(10).sum()/10
        temp1=self.open_price-vwap10
        part1=temp1.rank(axis=1,pct=True)
        temp2=(self.close-self.avg_price).abs()
        part2=-temp2.rank(axis=1,pct=True)
        result=part1*part2
        alpha=result.iloc[-1,:]
        return alpha.dropna()


    ##################################################################
    def alpha_013(self):
        result=((self.high-self.low)**0.5)-self.avg_price
        alpha=result.iloc[-1,:]
        return alpha.dropna()


    ##################################################################
    def alpha_014(self):
        result=self.close-self.close.shift(5)
        alpha=result.iloc[-1,:]
        return alpha.dropna()


    ################################################################## 
    def alpha_015(self):
        result=self.open_price/self.close.shift()-1
        alpha=result.iloc[-1,:]
        return alpha.dropna()


    ##################################################################  
    def alpha_016(self):
        temp1=self.volume.rank(axis=1,pct=True)
        temp2=self.avg_price.rank(axis=1,pct=True) 
        part=temp1.rolling(5).corr(temp2)#  
        part=part[(part<np.inf)&(part>-np.inf)]
        result=part.iloc[-5:,:]
        result=result.dropna(axis=1)
        alpha=-result.max()  
        return alpha.dropna()


    ################################################################## 
    def alpha_018(self):
        delay5=self.close.shift(5)
        alpha=self.close/delay5
        alpha=alpha.iloc[-1,:]
        return alpha.dropna()


    ##################################################################  
    def alpha_019(self):
        delay5=self.close.shift(5)
        condition1=(self.close<delay5)
        condition3=(self.close>delay5)
        part1=(self.close[condition1]-delay5[condition1])/delay5[condition1]
        part1=part1.fillna(0)
        part2=(self.close[condition3]-delay5[condition3])/self.close[condition3]
        part2=part2.fillna(0)
        result=part1+part2
        alpha=result.iloc[-1,:]
        return alpha.dropna()


    ##################################################################  
    def alpha_020(self):
        delay6=self.close.shift(6)
        result=(self.close-delay6)*100/delay6
        alpha=result.iloc[-1,:]
        return alpha.dropna()


    ##################################################################   
    def alpha_021(self):
        A=self.close.rolling(6).mean().iloc[-6:,:]
        alpha=A.apply(self.func_regbeta,axis=0)  
        return alpha.dropna()


    ##################################################################    
    def alpha_022(self):
        part1=(self.close-self.close.rolling(6).mean())/self.close.rolling(6).mean()
        temp=(self.close-self.close.rolling(6).mean())/self.close.rolling(6).mean()
        part2=temp.shift(3)
        result=part1-part2
        result=result.ewm(alpha=1.0/12).mean()
        alpha=result.iloc[-1,:]
        return alpha.dropna()    


    ##################################################################  
    def alpha_023(self):
        condition1=(self.close>self.close.shift())
        temp1=self.close.rolling(20).std()[condition1]
        temp1=temp1.fillna(0)
        temp2=self.close.rolling(20).std()[~condition1]
        temp2=temp2.fillna(0)
        part1=temp1.ewm(alpha=1.0/20).mean()
        part2=temp2.ewm(alpha=1.0/20).mean()
        result=part1*100/(part1+part2)
        alpha=result.iloc[-1,:]
        return alpha.dropna()


    ################################################################## 
    def alpha_024(self):
        delay5=self.close.shift(5)
        result=self.close-delay5
        result=result.ewm(alpha=1.0/5).mean()
        alpha=result.iloc[-1,:]
        return alpha.dropna()


    ##################################################################     
    def alpha_025(self):
        n=9
        part1=(self.close.diff(7)).rank(axis=1,pct=True)
        part1=part1.iloc[-1,:]
        temp=self.volume/self.volume.rolling(20).mean()
        temp1=temp.iloc[-9:,:]
        seq=[2*i/(n*(n+1)) for i in range(1,n+1)]   
        weight=np.array(seq)
        temp1=temp1.apply(lambda x: x*weight)   
        ret=self.close.pct_change()   
        rank_sum_ret=(ret.sum()).rank(pct=True)
        part2=1-temp1.sum() 
        part3=1+rank_sum_ret
        alpha=-part1*part2*part3
        return alpha.dropna()


    ##################################################################        
    def alpha_026(self):
        part1=self.close.rolling(7).sum()/7-self.close
        part1=part1.iloc[-1,:]
        delay5=self.close.shift(5)
        part2=self.avg_price.rolling(230).corr(delay5)
        part2=part2.iloc[-1,:]
        alpha=part1+part2
        return alpha.dropna()

    ##################################################################     
    def alpha_028(self):
        temp1=self.close-self.low.rolling(9).min()
        temp2=self.high.rolling(9).max()-self.low.rolling(9).min()
        part1=3*(temp1*100/temp2).ewm(alpha=1.0/3).mean()
        temp3=(temp1*100/temp2).ewm(alpha=1.0/3).mean()
        part2=2*temp3.ewm(alpha=1.0/3).mean()
        result=part1-part2
        alpha=result.iloc[-1,:]
        return alpha.dropna()


    ##################################################################     
    def alpha_029(self):
        delay6=self.close.shift(6)
        result=(self.close-delay6)*self.volume/delay6
        alpha=result.iloc[-1,:]
        return alpha.dropna()



    ##################################################################     
    def alpha_031(self):
        result=(self.close-self.close.rolling(12).mean())*100/self.close.rolling(12).mean()
        alpha=result.iloc[-1,:]
        return alpha.dropna()


    ##################################################################     
    def alpha_032(self):
        temp1=self.high.rank(axis=1,pct=True)
        temp2=self.volume.rank(axis=1,pct=True)
        temp3=temp1.rolling(3).corr(temp2)
        temp3=temp3[(temp3<np.inf)&(temp3>-np.inf)].fillna(0) 
        result=(temp3.rank(axis=1,pct=True)).iloc[-3:,:]
        alpha=-result.sum()
        return alpha.dropna()


    ##################################################################     
    def alpha_033(self):
        ret=self.close.pct_change()
        temp1=self.low.rolling(5).min()  #TS_MIN
        part1=temp1.shift(5)-temp1
        part1=part1.iloc[-1,:]
        temp2=(ret.rolling(240).sum()-ret.rolling(20).sum())/220
        part2=temp2.rank(axis=1,pct=True)
        part2=part2.iloc[-1,:]
        temp3=self.volume.iloc[-5:,:]
        part3=temp3.rank(axis=0,pct=True)   #TS_RANK
        part3=part3.iloc[-1,:]
        alpha=part1+part2+part3
        return alpha.dropna()


    ##################################################################     
    def alpha_034(self):
        result=self.close.rolling(12).mean()/self.close
        alpha=result.iloc[-1,:]
        return alpha.dropna()


    ##################################################################     
    def alpha_035(self):
        n=15
        m=7
        temp1=self.open_price.diff()
        temp1=temp1.iloc[-n:,:]
        seq1=[2*i/(n*(n+1)) for i in range(1,n+1)]   
        seq2=[2*i/(m*(m+1)) for i in range(1,m+1)]   
        weight1=np.array(seq1)
        weight2=np.array(seq2)
        part1=temp1.apply(lambda x: x*weight1)       
        part1=part1.rank(axis=1,pct=True)

        temp2=0.65*self.open_price+0.35*self.open_price
        temp2=temp2.rolling(17).corr(self.volume)
        temp2=temp2.iloc[-m:,:]
        part2=temp2.apply(lambda x: x*weight2)
        alpha=np.minimum(part1.iloc[-1,:],-part2.iloc[-1,:])
        alpha=alpha[(alpha<np.inf)&(alpha>-np.inf)] 
        alpha=alpha.dropna()    
        return alpha

    ##################################################################     
    def alpha_036(self):
        temp1=self.volume.rank(axis=1,pct=True)
        temp2=self.avg_price.rank(axis=1,pct=True)
        part1=temp1.rolling(6).corr(temp2)
        result=part1.rolling(2).sum()
        result=result.rank(axis=1,pct=True)
        alpha=result.iloc[-1,:]
        return alpha.dropna()


    ##################################################################     
    def alpha_037(self):
        ret=self.close.pct_change()
        temp=self.open_price.rolling(5).sum()*ret.rolling(5).sum()
        part1=temp.rank(axis=1,pct=True)
        part2=temp.shift(10)
        result=-part1-part2
        alpha=result.iloc[-1,:].dropna()
        return alpha


    ##################################################################     
    def alpha_038(self):
        sum_20=self.high.rolling(20).sum()/20
        delta2=self.high.diff(2)
        condition=(sum_20<self.high)
        result=-delta2[condition].fillna(0)
        alpha=result.iloc[-1,:]
        return alpha


    ##################################################################     
    def alpha_039(self):
        n=8
        m=12
        temp1=self.close.diff(2)
        temp1=temp1.iloc[-n:,:]
        seq1=[2*i/(n*(n+1)) for i in range(1,n+1)]          
        seq2=[2*i/(m*(m+1)) for i in range(1,m+1)]          
            
        weight1=np.array(seq1)
        weight2=np.array(seq2)
        part1=temp1.apply(lambda x: x*weight1)       
        part1=part1.rank(axis=1,pct=True)

        temp2=0.3*self.avg_price+0.7*self.open_price
        volume_180=self.volume.rolling(180).mean()
        sum_vol=volume_180.rolling(37).sum()
        temp3=temp2.rolling(14).corr(sum_vol)
        temp3=temp3.iloc[-m:,:]
        part2=-temp3.apply(lambda x: x*weight2)
        part2.rank(axis=1,pct=True)
        result=part1.iloc[-1,:]-part2.iloc[-1,:]
        alpha=result
        alpha=alpha[(alpha<np.inf)&(alpha>-np.inf)]
        alpha=alpha.dropna()
        return alpha


    ##################################################################     
    def alpha_040(self):
        delay1=self.close.shift()
        condition=(self.close>delay1)
        vol=self.volume[condition].fillna(0)
        vol_sum=vol.rolling(26).sum()
        vol1=self.volume[~condition].fillna(0)
        vol1_sum=vol1.rolling(26).sum()
        result=100*vol_sum/vol1_sum
        result=result.iloc[-1,:]
        alpha=result
        alpha=alpha[(alpha<np.inf)&(alpha>-np.inf)]
        alpha=alpha.dropna()
        return alpha


    ##################################################################     
    def alpha_041(self):
        delta_avg=self.avg_price.diff(3)
        part=np.maximum(delta_avg,5)
        result=-part.rank(axis=1,pct=True)
        alpha=result.iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    ##################################################################     
    def alpha_042(self):
        part1=self.high.rolling(10).corr(self.volume)
        part2=self.high.rolling(10).std()
        part2=part2.rank(axis=1,pct=True)
        result=-part1*part2
        alpha=result.iloc[-1,:]
        alpha=alpha[(alpha<np.inf)&(alpha>-np.inf)]
        alpha=alpha.dropna()
        return alpha


    ##################################################################     
    def alpha_043(self):
        delay1=self.close.shift()
        condition1=(self.close>delay1)
        condition2=(self.close<delay1)
        temp1=self.volume[condition1].fillna(0)
        temp2=-self.volume[condition2].fillna(0)
        result=temp1+temp2
        result=result.rolling(6).sum()
        alpha=result.iloc[-1,:].dropna()
        return alpha


    ##################################################################     
    def alpha_044(self):
        part1=self.open_price*0.4+self.close*0.6
        n=6
        m=10
        temp1=self.low.rolling(7).corr(self.volume.rolling(10).mean())
        temp1=temp1.iloc[-n:,:]
        seq1=[2*i/(n*(n+1)) for i in range(1,n+1)]          
        seq2=[2*i/(m*(m+1)) for i in range(1,m+1)]          
        weight1=np.array(seq1)
        weight2=np.array(seq2)
        part1=temp1.apply(lambda x: x*weight1)   
        part1=part1.iloc[-4:,].rank(axis=0,pct=True)

        temp2=self.avg_price.diff(3)
        temp2=temp2.iloc[-m:,:]
        part2=temp2.apply(lambda x: x*weight2)
        part2=part1.iloc[-5:,].rank(axis=0,pct=True)
        alpha=part1.iloc[-1,:]+part2.iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    ##################################################################     
    def alpha_045(self):
        temp1=self.close*0.6+self.open_price*0.4
        part1=temp1.diff()
        part1=part1.rank(axis=1,pct=True)
        temp2=self.volume.rolling(150).mean()
        part2=self.avg_price.rolling(15).corr(temp2)
        part2=part2.rank(axis=1,pct=True)
        result=part1*part2
        alpha=result.iloc[-1,:].dropna()
        return alpha


    ##################################################################     
    def alpha_046(self):
        part1=self.close.rolling(3).mean()
        part2=self.close.rolling(6).mean()
        part3=self.close.rolling(12).mean()
        part4=self.close.rolling(24).mean()
        result=(part1+part2+part3+part4)*0.25/self.close
        alpha=result.iloc[-1,:].dropna()
        return alpha


    ##################################################################     
    def alpha_047(self):
        part1=self.high.rolling(6).max()-self.close
        part2=self.high.rolling(6).max()- self.low.rolling(6).min()
        result=100*part1/part2.ewm(alpha=1.0/9).mean()
        alpha=result.iloc[-1,:].dropna()
        return alpha   


    ##################################################################     
    def alpha_048(self):
        condition1=(self.close>self.close.shift())
        condition2=(self.close.shift()>self.close.shift(2))
        condition3=(self.close.shift(2)>self.close.shift(3))

        indicator1=pd.DataFrame(np.ones(self.close.shape),index=self.close.index,columns=self.close.columns)[condition1].fillna(0)
        indicator2=pd.DataFrame(np.ones(self.close.shape),index=self.close.index,columns=self.close.columns)[condition2].fillna(0)
        indicator3=pd.DataFrame(np.ones(self.close.shape),index=self.close.index,columns=self.close.columns)[condition3].fillna(0)

        indicator11=-pd.DataFrame(np.ones(self.close.shape),index=self.close.index,columns=self.close.columns)[(~condition1)&(self.close!=self.close.shift())].fillna(0)
        indicator22=-pd.DataFrame(np.ones(self.close.shape),index=self.close.index,columns=self.close.columns)[(~condition2)&(self.close.shift()!=self.close.shift(2))].fillna(0)
        indicator33=-pd.DataFrame(np.ones(self.close.shape),index=self.close.index,columns=self.close.columns)[(~condition3)&(self.close.shift(2)!=self.close.shift(3))].fillna(0)

        summ=indicator1+indicator2+indicator3+indicator11+indicator22+indicator33  
        result=-summ*self.volume.rolling(5).sum()/self.volume.rolling(20).sum()
        alpha=result.iloc[-1,:].dropna()
        return alpha


    ##################################################################     
    def alpha_049(self):
        delay_high=self.high.shift()
        delay_low=self.low.shift()
        condition1=(self.high+self.low>=delay_high+delay_low)
        condition2=(self.high+self.low<=delay_high+delay_low)
        part1=np.maximum(np.abs(self.high-delay_high),np.abs(self.low-delay_low))
        part1=part1[~condition1]
        part1=part1.iloc[-12:,:].sum()

        part2=np.maximum(np.abs(self.high-delay_high),np.abs(self.low-delay_low))
        part2=part2[~condition2]
        part2=part2.iloc[-12:,:].sum()
        result=part1/(part1+part2)
        alpha=result.dropna()
        return alpha


 
    ##################################################################    
    def alpha_052(self):
        delay=((self.high+self.low+self.close)/3).shift()
        part1=(np.maximum(self.high-delay,0)).iloc[-26:,:]

        part2=(np.maximum(delay-self.low,0)).iloc[-26:,:]
        alpha=part1.sum()+part2.sum()
        return alpha
   

    ##################################################################    
    def alpha_053(self):
        delay=self.close.shift() 
        condition=self.close>delay
        result=self.close[condition].iloc[-12:,:] 
        alpha=result.count()*100/12
        return alpha.dropna()


    ##################################################################    
    def alpha_054(self):
        part1=(self.close-self.open_price).abs()
        part1=part1.std()
        part2=(self.close-self.open_price).iloc[-1,:]
        part3=self.close.iloc[-10:,:].corrwith(self.open_price.iloc[-10:,:])
        result=(part1+part2+part3).dropna()
        alpha=result.rank(pct=True)
        return alpha.dropna()
    


    ##################################################################    
    def alpha_056(self):
        part1=self.open_price.iloc[-1,:]-self.open_price.iloc[-12:,:].min()
        part1=part1.rank(pct=1)
        temp1=(self.high+self.low)/2
        temp1=temp1.rolling(19).sum()
        temp2=self.volume.rolling(40).mean().rolling(19).sum()
        part2=temp1.iloc[-13:,:].corrwith(temp2.iloc[-13:,:])
        part2=(part2.rank(pct=1))**5
        part2=part2.rank(pct=1)

        part1[part1<part2]=1                        
        part1=part1.apply(lambda x: 0 if x <1 else None)
        alpha=part1.fillna(1)
        return alpha.dropna()


    ##################################################################    
    def alpha_057(self):
        part1=self.close-self.low.rolling(9).min()
        part2=self.high.rolling(9).max()-self.low.rolling(9).min()
        result=100*part1/part2.ewm(alpha=1.0/3).mean()
        alpha=result.iloc[-1,:]
        return alpha.dropna()


    ##################################################################    
    def alpha_058(self):
        delay=self.close.shift() 
        condition=self.close>delay
        result=self.close[condition].iloc[-20:,:] 
        alpha=result.count()*100/20
        return alpha.dropna()   


    ##################################################################    
    def alpha_059(self):
        delay=self.close.shift()
        condition1=(self.close>delay)
        condition2=(self.close<delay)
        part1=np.minimum(self.low[condition1],delay[condition1]).fillna(0)
        part2=np.maximum(self.high[condition2],delay[condition2]).fillna(0)
        part1=part1.iloc[-20:,:]
        part2=part2.iloc[-20:,:]
        result=self.close-part1-part2
        alpha=result.sum()
        return alpha


    ##################################################################    
    def alpha_060(self):
        part1=(self.close.iloc[-20:,:]-self.low.iloc[-20:,:])-(self.high.iloc[-20:,:]-self.close.iloc[-20:,:])
        part2=self.high.iloc[-20:,:]-self.low.iloc[-20:,:]
        result=self.volume.iloc[-20:,:]*part1/part2
        alpha=result.sum()
        return alpha


    ##################################################################    
    def alpha_061(self):
        n=12
        m=17
        temp1=self.avg_price.diff()
        temp1=temp1.iloc[-n:,:]
        seq1=[2*i/(n*(n+1)) for i in range(1,n+1)]          
        seq2=[2*i/(m*(m+1)) for i in range(1,m+1)]          
            
            
        weight1=np.array(seq1)
        weight2=np.array(seq2)
        part1=temp1.apply(lambda x: x*weight1)       
        part1=part1.rank(axis=1,pct=True)

        temp2=self.low
        temp2=self.volume.rolling(80).mean().rolling(8).corr(temp2)
        temp2=temp2.rank(axis=1,pct=1)
        temp2=temp2.iloc[-m:,:]
        part2=temp2.apply(lambda x: x*weight2)
        part2=-part2.rank(axis=1,pct=1)
        alpha=np.maximum(part1.iloc[-1,:],part2.iloc[-1,:])
        alpha=alpha[(alpha<np.inf)&(alpha>-np.inf)]    
        alpha=alpha.dropna()    
        return alpha
    

    ##################################################################    
    def alpha_062(self):
        volume_rank=self.volume.rank(axis=1,pct=1)
        result=self.high.iloc[-5:,:].corrwith(volume_rank.iloc[-5:,:])
        alpha=-result
        return alpha.dropna()  


    ##################################################################    
    def alpha_063(self):
        part1=np.maximum(self.close-self.close.shift(),0)
        part1=part1.ewm(alpha=1.0/6).mean()
        part2=(self.close-self.close.shift()).abs()
        part2=part2.ewm(alpha=1.0/6).mean()
        result=part1*100/part2
        alpha=result.iloc[-1,:]
        return alpha.dropna()


    ##################################################################    
    def alpha_064(self):
        n=4
        m=14
        temp1=self.avg_price.rank(axis=1,pct=1).rolling(4).corr(self.volume.rank(axis=1,pct=1))
        temp1=temp1.iloc[-n:,:]
        seq1=[2*i/(n*(n+1)) for i in range(1,n+1)]       
        seq2=[2*i/(m*(m+1)) for i in range(1,m+1)]        
        weight1=np.array(seq1)
        weight2=np.array(seq2)
        part1=temp1.apply(lambda x: x*weight1)       
        part1=part1.rank(axis=1,pct=True)

        temp2=self.close.rank(axis=1,pct=1)
        temp2=self.volume.rolling(60).mean().rolling(4).corr(temp2)
        temp2=np.maximum(temp2,13)
        temp2=temp2.iloc[-m:,:]
        part2=temp2.apply(lambda x: x*weight2)
        part2=-part2.rank(axis=1,pct=1)
        alpha=np.maximum(part1.iloc[-1,:],part2.iloc[-1,:])
        alpha=alpha[(alpha<np.inf)&(alpha>-np.inf)]    
        alpha=alpha.dropna()    
        return alpha


    ##################################################################    
    def alpha_065(self):
        part1=self.close.iloc[-6:,:]
        alpha=part1.mean()/self.close.iloc[-1,:]
        return alpha.dropna()


    ##################################################################    
    def alpha_066(self):
        part1=self.close.iloc[-6:,:]
        alpha=(self.close.iloc[-1,:]-part1.mean())/part1.mean()
        return alpha


    ##################################################################    
    def alpha_067(self):
        temp1=self.close-self.close.shift()
        part1=np.maximum(temp1,0)
        part1=part1.ewm(alpha=1.0/24).mean()
        temp2=temp1.abs()
        part2=temp2.ewm(alpha=1.0/24).mean()
        result=part1*100/part2
        alpha=result.iloc[-1,:].dropna()
        return alpha


    ##################################################################    
    def alpha_068(self):
        part1=(self.high+self.low)/2-self.high.shift()
        part2=0.5*self.low.shift()*(self.high-self.low)/self.volume
        result=(part1+part2)*100
        result=result.ewm(alpha=2.0/15).mean()
        alpha=result.iloc[-1,:].dropna()
        return alpha
    
    ##################################################################
    def alpha_070(self):
    #### STD(AMOUNT, 6)
    ##       
        alpha = self.amount.iloc[-6:,:].std().dropna()
        return alpha
    
    
    #############################################################################
    def alpha_071(self):
    # (CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100 
    #     
        data = self.close - self.close.rolling(24).mean() / self.close.rolling(24).mean()
        alpha = data.iloc[-1].dropna()
        return alpha
    
    
    #############################################################################
    def alpha_072(self):
    #SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,15,1)
    #     
        data1 = self.high.rolling(6).max() - self.close
        data2 = self.high.rolling(6).max() - self.low.rolling(6).min()
        alpha = (data1 / data2 * 100).ewm(alpha=1/15).mean().iloc[-1].dropna()
        return alpha
        
        
    #############################################################################    
    def alpha_074(self):
    #(RANK(CORR(SUM(((LOW * 0.35) + (VWAP * 0.65)), 20), SUM(MEAN(VOLUME,40), 20), 7)) + RANK(CORR(RANK(VWAP), RANK(VOLUME), 6))) 
    #     
        data1 = (self.low * 0.35 + self.avg_price * 0.65).rolling(window=20).sum()
        data2 = self.volume.rolling(window=40).mean()
        rank1 = data1.rolling(window=7).corr(data2).rank(axis=1, pct=True)
        data3 = self.avg_price.rank(axis=1, pct=True)
        data4 = self.volume.rank(axis=1, pct=True)
        rank2 = data3.rolling(window=6).corr(data4).rank(axis=1, pct=True)
        alpha = (rank1 + rank2).iloc[-1].dropna()
        return alpha
    
    
    #############################################################################
    def alpha_075(self):
    #COUNT(CLOSE>OPEN & BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50)/COUNT(BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50) 
    #     
        condition = self.benchmark_close_price < self.benchmark_open_price
        data1 = benchmark[condition]
        numbench = len(data1)
        data2 = pd.merge(self.close, data1, left_index=True, right_index=True).drop(['close', 'open'], axis=1)
        data3 = pd.merge(self.open_price, data1, left_index=True, right_index=True).drop(['close', 'open'], axis=1)
        data4 = data2[data2 > data3]
        alpha = 1 - data4.isnull().sum(axis=0) / numbench
        alpha = alpha.reset_index().rename(columns={'index': 'sec_code'}).set_index('sec_code')
        return alpha.iloc[:, 0]
    
    
    #############################################################################
    def alpha_076(self):
    #STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)/MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20) 
    #     
    
        data1 = abs((self.close / ((self.prev_close - 1) / self.volume).shift(20))).std()
        data2 = abs((self.close / ((self.prev_close - 1) / self.volume).shift(20))).mean()
        alpha = (data1 / data2).dropna()
        return alpha
    
    
    #############################################################################
    def alpha_077(self):
    #MIN(RANK(DECAYLINEAR(((((HIGH + LOW) / 2) + HIGH)  -  (VWAP + HIGH)), 20)), RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 3), 6)))     
    #     
    
        data1 = ((self.high + self.low) / 2 + self.high - (self.avg_price + self.high)).iloc[-20:,:]
        decay_weights = np.arange(1,20+1,1)[::-1]
        decay_weights = decay_weights / decay_weights.sum()
        rank1 = data1.apply(lambda x : x * decay_weights).rank(axis=1, pct=True)
        data2 = self.volume.rolling(window=40).mean().rolling(window=3).corr((self.high + self.low)/2).iloc[-6:,:]
        decay_weights2 = np.arange(1,6+1,1)[::-1]
        decay_weights2 = decay_weights2 / decay_weights2.sum()
        rank2 = data2.apply(lambda x : x * decay_weights2).rank(axis=1, pct=True)
        alpha = np.minimum(rank1.iloc[-1], rank2.iloc[-1])
        return alpha
     
    
    #############################################################################
    def alpha_078(self):
    #((HIGH+LOW+CLOSE)/3-MA((HIGH+LOW+CLOSE)/3,12))/(0.015*MEAN(ABS(CLOSE-MEAN((HIGH+LOW+CLOSE)/3,12)),12)) 
    #     
        data1 = (self.high + self.low + self.close) / 3 - ((self.high + self.low + self.close) / 3).rolling(window=12).mean()
        data2 = abs(self.close - ((self.high + self.low + self.close) / 3).rolling(window=12).mean())
        data3 = data2.rolling(window=12).mean() * 0.015
        alpha = (data1 / data3).iloc[-1].dropna()   
        return alpha
    
    
    #############################################################################
    def alpha_079(self):
    #SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100
    #     
        data1 = (np.maximum((self.close - self.prev_close), 0)).ewm(alpha=1/12).mean()
        data2 = (abs(self.close - self.prev_close)).ewm(alpha=1/12).mean()
        alpha = (data1 / data2 * 100).iloc[-1].dropna()
        return alpha
    
    
    #############################################################################
    def alpha_080(self):
    #(VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100
    #     
        alpha =  ((self.volume - self.volume.shift(5))/self.volume.shift(5) * 100).iloc[-1].dropna()
        return alpha
    
    
    #############################################################################
    def alpha_081(self):
        result=self.volume.ewm(alpha=2.0/21).mean()
        alpha=result.iloc[-1,:].dropna()
        return alpha

    
    #############################################################################
    def alpha_082(self):
        part1=self.high.rolling(6).max()-self.close
        part2=self.high.rolling(6).max()-self.low.rolling(6).min()
        result=(100*part1/part2).ewm(alpha=1.0/20).mean()
        alpha=result.iloc[-1,:].dropna()
        return alpha
  

    #############################################################################
    def alpha_083(self):
        part1=self.high.rank(axis=0,pct=True) 
        part1=part1.iloc[-5:,:]
        part2=self.volume.rank(axis=0,pct=True) 
        part2=part2.iloc[-5:,:]
        result=part1.corrwith(part2)
        alpha=-result
        return alpha.dropna()


    #############################################################################
    def alpha_084(self):
        condition1=(self.close>self.close.shift())
        condition2=(self.close<self.close.shift())
        part1=self.volume[condition1].fillna(0)
        part2=-self.volume[condition2].fillna(0)
        result=part1.iloc[-20:,:]+part2.iloc[-20:,:]
        alpha=result.sum().dropna()
        return alpha
    
    
    #############################################################################
    def alpha_085(self):
        temp1=self.volume.iloc[-20:,:]/self.volume.iloc[-20:,:].mean() 
        temp1=temp1 
        part1=temp1.rank(axis=0,pct=True)
        part1=part1.iloc[-1,:] 

        delta=self.close.diff(7)
        temp2=-delta.iloc[-8:,:]
        part2=temp2.rank(axis=0,pct=True).iloc[-1,:]
        part2=part2 
        alpha=part1*part2
        return alpha.dropna()
    
    
    #############################################################################
    def alpha_086(self):

        delay10=self.close.shift(10)
        delay20=self.close.shift(20)
        indicator1=pd.DataFrame(-np.ones(self.close.shape),index=self.close.index,columns=self.close.columns) 
        indicator2=pd.DataFrame(np.ones(self.close.shape),index=self.close.index,columns=self.close.columns) 

        temp=(delay20-delay10)/10-(delay10-self.close)/10
        condition1=(temp>0.25)
        condition2=(temp<0)
        temp2=(self.close-self.close.shift())*indicator1

        part1=indicator1[condition1].fillna(0)
        part2=indicator2[~condition1][condition2].fillna(0)
        part3=temp2[~condition1][~condition2].fillna(0)
        result=part1+part2+part3
        alpha=result.iloc[-1,:].dropna()

        return alpha
    
    
    #############################################################################
    def alpha_087(self):
        n=7
        m=11
        temp1=self.avg_price.diff(4)
        temp1=temp1.iloc[-n:,:]
        seq1=[2*i/(n*(n+1)) for i in range(1,n+1)]       
        seq2=[2*i/(m*(m+1)) for i in range(1,m+1)]        
            
        weight1=np.array(seq1)
        weight2=np.array(seq2)
        part1=temp1.apply(lambda x: x*weight1)       
        part1=part1.rank(axis=1,pct=True)

        temp2=self.low-self.avg_price
        temp3=self.open_price-0.5*(self.high+self.low)
        temp2=temp2/temp3
        temp2=temp2.iloc[-m:,:]
        part2=-temp2.apply(lambda x: x*weight2)

        part2=part2.rank(axis=0,pct=1)
        alpha=part1.iloc[-1,:]+part2.iloc[-1,:]
        alpha=alpha[(alpha<np.inf)&(alpha>-np.inf)]    
        alpha=alpha.dropna()    
        return alpha

    '''
    ########################################################################
    '''
    def alpha_088(self):
        #(close-delay(close,20))/delay(close,20)*100
        ####################      
        data1=self.close.iloc[-21,:]
        alpha=((self.close.iloc[-1,:]-data1)/data1)*100
        alpha=alpha.dropna()
        return alpha
    
    
    def alpha_089(self):
        #2*(sma(close,13,2)-sma(close,27,2)-sma(sma(close,13,2)-sma(close,27,2),10,2))
        ######################      
        data1=self.close.ewm(span=12,adjust=False).mean()
        data2=self.close.ewm(span=26,adjust=False).mean()
        data3=(data1-data2).ewm(span=9,adjust=False).mean()
        alpha=((data1-data2-data3)*2).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    
    def alpha_090(self):
        #(rank(corr(rank(vwap),rank(volume),5))*-1)
        #######################      
        data1=self.avg_price.rank(axis=1,pct=True)
        data2=self.volume.rank(axis=1,pct=True)
        corr=data1.iloc[-5:,:].corrwith(data2.iloc[-5:,:])
        rank1=corr.rank(pct=True)
        alpha=rank1*-1
        alpha=alpha.dropna()
        return alpha
    
    
    def alpha_091(self):
        #((rank((close-max(close,5)))*rank(corr((mean(volume,40)),low,5)))*-1)
        #################      
        data1=self.close
        cond=data1>5
        data1[~cond]=5
        rank1=((self.close-data1).rank(axis=1,pct=True)).iloc[-1,:]
        mean=self.volume.rolling(window=40).mean()
        corr=mean.iloc[-5:,:].corrwith(self.low.iloc[-5:,:])
        rank2=corr.rank(pct=True) 
        alpha=rank1*rank2*(-1)
        alpha=alpha.dropna()
        return alpha
    
    
    #       
    def alpha_092(self):
        # (MAX(RANK(DECAYLINEAR(DELTA(((CLOSE*0.35)+(VWAP*0.65)),2),3)),TSRANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME,180)),CLOSE,13)),5),15))*-1) #
        delta = (self.close*0.35+self.avg_price*0.65)-(self.close*0.35+self.avg_price*0.65).shift(2)
        rank1 = (delta.rolling(3).apply(self.func_decaylinear)).rank(axis=1, pct=True)
        rank2 = ((self.volume.rolling(180).mean().rolling(13).corr(self.close).abs()).rolling(5).apply(self.func_decaylinear)).rolling(15).apply(self.func_rank)
        cond_max = rank1>rank2
        rank2[cond_max] = rank1[cond_max]
        alpha = (-rank2).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    
    #       
    def alpha_093(self):
        # SUM((OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))),20) #
        cond = self.open_price>=self.open_price.shift()
        data1 = self.open_price-self.low
        data2 = self.open_price-self.open_price.shift()
        cond_max = data1>data2
        data2[cond_max] = data1[cond_max]
        data2[cond] = 0
        alpha = data2.iloc[-20:,:].sum()
        alpha=alpha.dropna()
        return alpha
    
    
    #       
    def alpha_094(self):
        # SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),30) #
        cond1 = self.close>self.prev_close
        cond2 = self.close<self.prev_close
        value = -self.volume
        value[~cond2] = 0
        value[cond1] = self.volume[cond1]
        alpha = value.iloc[-30:,:].sum()
        alpha=alpha.dropna()
        return alpha
    
    
    #       
    def alpha_095(self):
        # STD(AMOUNT,20) #
        alpha = self.amount.iloc[-20:,:].std()
        alpha=alpha.dropna()
        return alpha
    
    
    #       
    def alpha_096(self):
        # SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1) #
        sma1 = (100*(self.close-self.low.rolling(9).min())/(self.high.rolling(9).max()-self.low.rolling(9).min())).ewm(span=5, adjust=False).mean()
        alpha = sma1.ewm(span=5, adjust=False).mean().iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    
    #       
    def alpha_097(self):
        # STD(VOLUME,10) #
        alpha = self.volume.iloc[-10:,:].std()
        alpha=alpha.dropna()
        return alpha
    
    
    #       
    def alpha_098(self):
        # ((((DELTA((SUM(CLOSE,100)/100),100)/DELAY(CLOSE,100))<0.05)||((DELTA((SUM(CLOSE,100)/100),100)/DELAY(CLOSE,100))==0.05))?(-1*(CLOSE-TSMIN(CLOSE,100))):(-1*DELTA(CLOSE,3))) #
        sum_close = self.close.rolling(100).sum()
        cond = (sum_close/100-(sum_close/100).shift(100))/self.close.shift(100) <= 0.05
        left_value = -(self.close-self.close.rolling(100).min())
        right_value = -(self.close-self.close.shift(3))
        right_value[cond] = left_value[cond]
        alpha = right_value.iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    
    #       
    def alpha_099(self):
        # (-1 * RANK(COVIANCE(RANK(CLOSE), RANK(VOLUME), 5))) #
        alpha = (-self.close.rank(axis=1, pct=True)).rolling(window=5).cov(self.volume.rank(axis=1, pct=True).rank(axis=1, pct=True)).iloc[-1,:]
        alpha = alpha.dropna()
        return alpha
    
    
    def alpha_100(self):
        # STD(VOLUME,20) #
        alpha = self.volume.iloc[-20:,:].std()
        alpha=alpha.dropna()
        return alpha
    
    
    #       
    def alpha_101(self):
        # ((RANK(CORR(CLOSE,SUM(MEAN(VOLUME,30),37),15))<RANK(CORR(RANK(((HIGH*0.1)+(VWAP*0.9))),RANK(VOLUME),11)))*-1) #
        rank1 = (self.close.rolling(window=15).corr((self.volume.rolling(window=30).mean()).rolling(window=37).sum())).rank(axis=1, pct=True)
        rank2 = (self.high*0.1+self.avg_price*0.9).rank(axis=1, pct=True)
        rank3 = self.volume.rank(axis=1, pct=True)
        rank4 = (rank2.rolling(window=11).corr(rank3)).rank(axis=1, pct=True)
        alpha = -(rank1<rank4)
        alpha=alpha.iloc[-1,:].dropna()
        return alpha
    
    
    #       
    def alpha_102(self):
        # SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100 #
        max_cond = (self.volume-self.volume.shift())>0
        max_data = self.volume-self.volume.shift()
        max_data[~max_cond] = 0
        sma1 = max_data.ewm(span=11, adjust=False).mean()
        sma2 = ((self.volume-self.volume.shift()).abs()).ewm(span=11, adjust=False).mean()
        alpha = (sma1/sma2*100).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    
    def alpha_103(self):
        ##### ((20-LOWDAY(LOW,20))/20)*100 
        ##       
        alpha = (20 - self.low.iloc[-20:,:].apply(self.func_lowday))/20*100
        alpha=alpha.dropna()
        return alpha
    
    
    #       
    def alpha_104(self):
        # (-1*(DELTA(CORR(HIGH,VOLUME,5),5)*RANK(STD(CLOSE,20)))) #
        corr = self.high.rolling(window=5).corr(self.volume)
        alpha = (-(corr-corr.shift(5))*((self.close.rolling(window=20).std()).rank(axis=1, pct=True))).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    
    #       
    def alpha_105(self):
        # (-1*CORR(RANK(OPEN),RANK(VOLUME),10)) #
        alpha = -((self.open_price.rank(axis=1, pct=True)).iloc[-10:,:]).corrwith(self.volume.iloc[-10:,:].rank(axis=1, pct=True))
        alpha=alpha.dropna()
        return alpha
    
    
    #       
    def alpha_106(self):
        # CLOSE-DELAY(CLOSE,20) #
        alpha = (self.close-self.close.shift(20)).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    
    #       
    def alpha_107(self):
        # (((-1*RANK((OPEN-DELAY(HIGH,1))))*RANK((OPEN-DELAY(CLOSE,1))))*RANK((OPEN-DELAY(LOW,1)))) #
        rank1 = -(self.open_price-self.high.shift()).rank(axis=1, pct=True)
        rank2 = (self.open_price-self.close.shift()).rank(axis=1, pct=True)
        rank3 = (self.open_price-self.low.shift()).rank(axis=1, pct=True)
        alpha = (rank1*rank2*rank3).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    
    #       
    def alpha_108(self):
        # ((RANK((HIGH-MIN(HIGH,2)))^RANK(CORR((VWAP),(MEAN(VOLUME,120)),6)))*-1) #
        min_cond = self.high>2
        data = self.high
        data[min_cond] = 2
        rank1 = (self.high-data).rank(axis=1, pct=True)
        rank2 = (self.avg_price.rolling(window=6).corr(self.volume.rolling(window=120).mean())).rank(axis=1, pct=True)
        alpha = (-rank1**rank2).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    
    #       
    def alpha_109(self):
        # SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)#
        data = self.high-self.low
        sma1 = data.ewm(span=9, adjust=False).mean()
        sma2 = sma1.ewm(span=9, adjust=False).mean()
        alpha = (sma1/sma2).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    
    #       
    def alpha_110(self):
        #输出格式有问题
        # SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100 #
        data1 = self.high-self.prev_close
        data2 = self.prev_close-self.low
        max_cond1 = data1<0
        max_cond2 = data2<0
        data1[max_cond1] = 0
        data2[max_cond2] = 0
        sum1 = data1.rolling(window=20).sum()
        sum2 = data2.rolling(window=20).sum()
        alpha = sum1/sum2*100
        alpha=alpha.dropna()
        if alpha.shape[0]>0:
            alpha = alpha.iloc[-1,:]
        else: 
            alpha = pd.Series()
        return alpha

    
    
    def alpha_111(self):
        #sma(vol*((close-low)-(high-close))/(high-low),11,2)-sma(vol*((close-low)-(high-close))/(high-low),4,2)
        ######################      
        data1=self.volume*((self.close-self.low)-(self.high-self.close))/(self.high-self.low)
        x=data1.ewm(span=10).mean()
        y=data1.ewm(span=3).mean()
        alpha=(x-y).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    
    #       
    def alpha_112(self):
        # (SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)-SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))/(SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)+SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))*100 #
        cond1 = self.close>self.prev_close
        cond2 = self.close<self.prev_close
        data1 = self.close-self.prev_close
        data2 = self.close-self.prev_close
        data1[~cond1] = 0
        data2[~cond2] = 0
        data2 = data2.abs()
        sum1 = data1.rolling(window=12).sum()
        sum2 = data2.rolling(window=12).sum()
        alpha = ((sum1-sum2)/(sum1+sum2)*100).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    
    def alpha_113(self):
        #(-1*((rank((sum(delay(close,5),20)/20))*corr(close,volume,2))*rank(corr(sum(close,5),sum(close,20),2))))
        #####################      
        data1=self.close.iloc[:-5,:]
        rank1=((data1.rolling(window=20).sum())/20).rank(axis=1,pct=True)
        corr1=self.close.iloc[-2:,:].corrwith(self.volume.iloc[-2:,:])
        data2=self.close.rolling(window=5).sum()
        data3=self.close.rolling(window=20).sum()
        corr2=data2.iloc[-2:,:].corrwith(data3.iloc[-2:,:])
        rank2=corr2.rank(axis=0,pct=True)
        alpha=(-1*rank1*corr1*rank2)
        alpha=alpha.iloc[-1,:].dropna()

        return alpha
    
    
    def alpha_114(self):
        #((rank(delay(((high-low)/(sum(close,5)/5)),2))*rank(rank(volume)))/(((high-low)/(sum(close,5)/5))/(vwap-close)))
        #####################      
        data1=(self.high-self.low)/(self.close.rolling(window=5).sum()/5)

        rank1=(data1.iloc[-2,:]).rank(axis=0,pct=True)
        rank2=((self.volume.rank(axis=1,pct=True)).rank(axis=1,pct=True)).iloc[-1,:]
        data2=(((self.high-self.low)/(self.close.rolling(window=5).sum()/5))/(self.avg_price-self.close)).iloc[-1,:]
        alpha=(rank1*rank2)/data2
        alpha=alpha.dropna()
        return alpha 
    
    
    #       
    # def alpha_115(self):
    #     # RANK(CORR(((HIGH*0.9)+(CLOSE*0.1)),MEAN(VOLUME,30),10))^RANK(CORR(TSRANK(((HIGH+LOW)/2),4),TSRANK(VOLUME,10),7)) #
    #     data1 = (self.high*0.9+self.close*0.1)
    #     data2 = self.volume.rolling(window=30).mean()
    #     rank1 = (data1.iloc[-10:,:].corrwith(data2.iloc[-10:,:])).rank(pct=True)
    #     tsrank1 = ((self.high+self.low)/2).rolling(4).apply(self.func_rank)
    #     tsrank2 = self.volume.rolling(10).apply(self.func_rank)
    #     rank2 = tsrank1.iloc[-7:,:].corrwith(tsrank2.iloc[-7:,:]).rank(pct=True)
    #     alpha = rank1**rank2
    #     alpha=alpha.dropna()
    #     return alpha
    
    
    #       
    def alpha_116(self):
        # REGBETA(CLOSE,SEQUENCE,20) #
        if self.close.shape[0] > 20:
            sequence = pd.Series(range(1,21), index=self.close.iloc[-20:,].index)   # 1~20
            corr = self.close.iloc[-20:,:].corrwith(sequence)
            alpha = corr
            alpha=alpha.dropna()
        else: 
            alpha =pd.Series()
        return alpha

    
    def alpha_117(self):
        #######((tsrank(volume,32)*(1-tsrank(((close+high)-low),16)))*(1-tsrank(ret,32)))
        ####################      
        data1=(self.close+self.high-self.low).iloc[-16:,:]
        data2=1-data1.rank(axis=0,pct=True)
        data3=(self.volume.iloc[-32:,:]).rank(axis=0,pct=True)
        ret=(self.close/self.close.shift()-1).iloc[-32:,:]
        data4=1-ret.rank(axis=0,pct=True)
        alpha=(data2.iloc[-1,:])*(data3.iloc[-1,:])*(data4.iloc[-1,:])
        alpha=alpha.dropna()
        return alpha 
    
    
    def alpha_118(self):
        ######sum(high-open,20)/sum((open-low),20)*100
        ###################      
        data1=self.high-self.open_price
        data2=self.open_price-self.low
        data3=data1.rolling(window=20).sum()
        data4=data2.rolling(window=20).sum()
        alpha=((data3/data4)*100).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    
    #        
    def alpha_119(self):
        # (RANK(DECAYLINEAR(CORR(VWAP,SUM(MEAN(VOLUME,5),26),5),7))-RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN),RANK(MEAN(VOLUME,15)),21),9),7),8)))
        sum1 = (self.volume.rolling(window=5).mean()).rolling(window=26).sum()
        corr1 = self.avg_price.rolling(window=5).corr(sum1)
        rank1 = corr1.rolling(7).apply(self.func_decaylinear).rank(axis=1, pct=True)
        rank2 = self.open_price.rank(axis=1, pct=True)
        rank3 = (self.volume.rolling(window=15).mean()).rank(axis=1, pct=True)
        rank4 = rank2.rolling(window=21).corr(rank3).rolling(window=9).min().rolling(7).apply(self.func_rank)
        rank5 = rank4.rolling(8).apply(self.func_decaylinear).rank(axis=1, pct=True)
        alpha = (rank1 - rank5).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    
    def alpha_120(self):
        ###############(rank(vwap-close))/(rank(vwap+close))
        ###################      
        data1=(self.avg_price-self.close).rank(axis=1,pct=True)
        data2=(self.avg_price+self.close).rank(axis=1,pct=True)
        alpha=(data1/data2).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    
    
    def alpha_122(self):
        ##### (SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2) - DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)) 
        ##### / DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)
        ##     
        log_close = np.log(self.close)
        data = ((log_close.ewm(span=12, adjust=False).mean()).ewm(span=12, adjust=False).mean()).ewm(span=12, adjust=False).mean()
        alpha = (data.iloc[-1,:] / data.iloc[-2,:]) -1 
        alpha=alpha.dropna()
        return alpha
        
        
    def alpha_123(self):
        #####((RANK(CORR(SUM(((HIGH+LOW)/2), 20), SUM(MEAN(VOLUME, 60), 20), 9)) < RANK(CORR(LOW, VOLUME, 6))) * -1)
        ##       
        data1 = ((self.high + self.low)/2).rolling(20).sum()
        data2 = self.volume.rolling(60).mean().rolling(20).sum()
        rank1 = data1.iloc[-9:,:].corrwith(data2.iloc[-9:,:]).dropna().rank(axis=0, pct=True)
        rank2 = self.low.iloc[-6:,:].corrwith(self.volume.iloc[-6:,:]).dropna().rank(axis=0, pct=True)
        rank1 = rank1[rank1.index.isin(rank2.index)]
        rank2 = rank2[rank2.index.isin(rank1.index)]
        alpha = (rank1 < rank2) * (-1)
        alpha=alpha.dropna()
        return alpha
    
    
    def alpha_124(self):
        ##### (CLOSE - VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE, 30)),2)
        ##       
        data1 = self.close.rolling(30).max().rank(axis=1, pct=True)
        alpha = (self.close.iloc[-1,:] - self.avg_price.iloc[-1,:]) / (2./3*data1.iloc[-2,:] + 1./3*data1.iloc[-1,:])
        alpha=alpha.dropna()
        return alpha
    
    
    def alpha_125(self):
        ##### (RANK(DECAYLINEAR(CORR((VWAP), MEAN(VOLUME, 80), 17), 20)) / RANK(DECAYLINEAR(DELTA((CLOSE * 0.5 + VWAP * 0.5), 3), 16)))
        ##       
        data1 = self.avg_price.rolling(window = 17).corr(self.volume.rolling(80).mean()) 
        decay_weights = np.arange(1,21,1)[::-1]    # 倒序数组
        decay_weights = decay_weights / decay_weights.sum()
        rank1 = data1.iloc[-20:,:].mul(decay_weights, axis=0).sum().rank(axis=0, pct=True)

        data2 = (self.close * 0.5 + self.avg_price * 0.5).diff(3)
        decay_weights = np.arange(1,17,1)[::-1]    # 倒序数组
        decay_weights = decay_weights / decay_weights.sum()
        rank2 = data2.iloc[-16:,:].mul(decay_weights, axis=0).sum().rank(axis=0, pct=True)

        alpha = rank1 / rank2
        alpha=alpha.dropna()
        return alpha
  
        
    
    def alpha_126(self):
        #### (CLOSE + HIGH + LOW) / 3
        ##       
        alpha = (self.close.iloc[-1,:] + self.high.iloc[-1,:] + self.low.iloc[-1,:]) / 3
        alpha=alpha.dropna()
        return alpha
    
    
    
    def alpha_129(self):
        #### SUM((CLOSE - DELAY(CLOSE, 1) < 0 ? ABS(CLOSE - DELAY(CLOSE, 1)):0), 12)
        ##       
        data = self.close.diff(1)
        data[data >= 0] = 0
        data = abs(data)
        alpha = data.iloc[-12:,:].sum()
        alpha=alpha.dropna()
        return alpha

    
    def alpha_130(self):
        #### alpha_130
        #### (RANK(DELCAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME, 40), 9), 10)) / RANK(DELCAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 7), 3)))
        ##       
        data1 = (self.high + self.low) / 2
        data2 = self.volume.rolling(40).mean()
        data3 = data1.rolling(window=9).corr(data2)
        decay_weights = np.arange(1,11,1)[::-1]    # 倒序数组
        decay_weights = decay_weights / decay_weights.sum()
        rank1 = data3.iloc[-10:,:].mul(decay_weights, axis=0).sum().rank(axis=0, pct=True)

        data1 = self.avg_price.rank(axis=1, pct=True)
        data2 = self.volume.rank(axis=1, pct=True)
        data3 = data1.rolling(window=7).corr(data2)
        decay_weights = np.arange(1,4,1)[::-1]    # 倒序数组
        decay_weights = decay_weights / decay_weights.sum()
        rank2 = data3.iloc[-3:,:].mul(decay_weights, axis=0).sum().rank(axis=0, pct=True)

        alpha = (rank1 / rank2).dropna()
        return alpha
    
    
    def alpha_132(self):
        #### MEAN(AMOUNT, 20)
        ##        
        alpha = self.amount.iloc[-20:,:].mean()
        alpha=alpha.dropna()
        return alpha
    
    
    def alpha_133(self):
        #### alpha_133
        #### ((20 - HIGHDAY(HIGH, 20)) / 20)*100 - ((20 - LOWDAY(LOW, 20)) / 20)*100
        ##        

        alpha = (20 - self.high.iloc[-20:,:].apply(self.func_highday))/20*100 \
                 - (20 - self.low.iloc[-20:,:].apply(self.func_lowday))/20*100
        alpha=alpha.dropna()
        return alpha
    
    
    def alpha_134(self):
        #### (CLOSE - DELAY(CLOSE, 12)) / DELAY(CLOSE, 12) * VOLUME
        ##        
        alpha = ((self.close.iloc[-1,:] / self.close.iloc[-13,:] - 1) * self.volume.iloc[-1,:])
        alpha=alpha.dropna()
        return alpha
    
    
    def alpha_135(self):
        #### SMA(DELAY(CLOSE / DELAY(CLOSE, 20), 1), 20, 1)
        ##       
        def rolling_div(na):
            return na[-1]/na[-21]

        data1 = self.close.rolling(21).apply(rolling_div).shift(periods=1)
        alpha = data1.ewm(com=19, adjust=False).mean().iloc[-1,:]
        alpha=alpha.dropna()
        return alpha 
    
    def alpha_136(self):
        #### ((-1 * RANK(DELTA(RET, 3))) * CORR(OPEN, VOLUME, 10))
        ##        
        data1 = -(self.close / self.prev_close - 1).diff(3).rank(axis=1, pct=True)
        data2 = self.open_price.iloc[-10:,:].corrwith(self.volume.iloc[-10:,:])
        alpha = (data1.iloc[-1,:] * data2).dropna()
        
        return alpha
    
    
    def alpha_138(self):
        #### ((RANK(DECAYLINEAR(DELTA((((LOW * 0.7) + (VWAP * 0.3))), 3), 20)) - TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW, 8), TSRANK(MEAN(VOLUME, 60), 17), 5), 19), 16), 7)) * -1)
        ##        
        data1 = (self.low * 0.7 + self.avg_price * 0.3).diff(3)
        decay_weights = np.arange(1,21,1)[::-1]    # 倒序数组
        decay_weights = decay_weights / decay_weights.sum()
        rank1 = data1.iloc[-20:,:].mul(decay_weights, axis=0).sum().rank(axis=0, pct=True)

        data1 = self.low.rolling(8).apply(self.func_rank)
        data2 = self.volume.rolling(60).mean().rolling(17).apply(self.func_rank)
        data3 = data1.rolling(window=5).corr(data2).rolling(19).apply(self.func_rank)
        rank2 = data3.rolling(16).apply(self.func_decaylinear).iloc[-7:,:].rank(axis=0, pct=True).iloc[-1,:]
        
        alpha = (rank2 - rank1).dropna()
        return alpha
    
    def alpha_139(self):
        #### (-1 * CORR(OPEN, VOLUME, 10))
        ##        
        alpha = - self.open_price.iloc[-10:,:].corrwith(self.volume.iloc[-10:,:]).dropna()
        return alpha
    
    
    def alpha_141(self):
        #### (RANK(CORR(RANK(HIGH), RANK(MEAN(VOLUME, 15)), 9))* -1)
        ##       
        df1 = self.high.rank(axis=1, pct=True)
        df2 = self.volume.rolling(15).mean().rank(axis=1, pct=True)
        alpha = -df1.iloc[-9:,:].corrwith(df2.iloc[-9:,:]).rank(pct=True)
        alpha=alpha.dropna()
        return alpha
    
    def alpha_142(self):
        #### (((-1 * RANK(TSRANK(CLOSE, 10))) * RANK(DELTA(DELTA(CLOSE, 1), 1))) * RANK(TSRANK((VOLUME/MEAN(VOLUME, 20)), 5)))
        ##        

        rank1 = self.close.iloc[-10:,:].rank(axis=0, pct=True).iloc[-1,:].rank(pct=True)
        rank2 = self.close.diff(1).diff(1).iloc[-1,:].rank(pct=True)
        rank3 = (self.volume / self.volume.rolling(20).mean()).iloc[-5:,:].rank(axis=0, pct=True).iloc[-1,:].rank(pct=True)

        alpha = -(rank1 * rank2 * rank3).dropna()
        alpha=alpha.dropna()
        return alpha
    
    
    def alpha_144(self):
        #### SUMIF(ABS(CLOSE/DELAY(CLOSE, 1) - 1)/AMOUNT, 20, CLOSE < DELAY(CLOSE, 1))/COUNT(CLOSE < DELAY(CLOSE, 1), 20)
        ##       
        df1 = self.close < self.prev_close
        sumif = ((abs(self.close / self.prev_close - 1)/self.amount) * df1).iloc[-20:,:].sum()
        count = df1.iloc[-20:,:].sum()

        alpha = (sumif / count).dropna()
        alpha=alpha.dropna()
        return alpha
    
    
    def alpha_145(self):
        #### (MEAN(VOLUME, 9) - MEAN(VOLUME, 26)) / MEAN(VOLUME, 12) * 100
        ##       

        alpha = (self.volume.iloc[-9:,:].mean() - self.volume.iloc[-26:,:].mean()) / self.volume.iloc[-12:,:].mean() * 100
        alpha=alpha.dropna()
        return alpha
    
    
    def alpha_148(self):
        #### ((RANK(CORR((OPEN), SUM(MEAN(VOLUME, 60), 9), 6)) < RANK((OPEN - TSMIN(OPEN, 14)))) * -1)
        ##        
        df1 = self.volume.rolling(60).mean().rolling(9).sum()
        rank1 = self.open_price.iloc[-6:,:].corrwith(df1.iloc[-6:,:]).rank(pct=True)
        rank2 = (self.open_price - self.open_price.rolling(14).min()).iloc[-1,:].rank(pct=True)

        alpha = -1 * (rank1 < rank2)
        alpha=alpha.dropna()
        return alpha
    

    
    def alpha_150(self):
        #### (CLOSE + HIGH + LOW)/3 * VOLUME
        ##       

        alpha = ((self.close + self.high + self.low)/3 * self.volume).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    

    ######################## alpha_152 #######################
    #       
    def alpha_152(self):
        # SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12)-MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),26),9,1) #
        #       
        data1 = (((((self.close/self.close.shift(9)).shift())).ewm(span=17, adjust=False).mean()).shift()).rolling(12).mean()
        data2 = (((((self.close/self.close.shift(9)).shift())).ewm(span=17, adjust=False).mean()).shift()).rolling(26).mean()
        alpha = ((data1-data2).ewm(span=17, adjust=False).mean()).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    
    ######################## alpha_153 #######################
    #       
    def alpha_153(self):
        # (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4 #
        alpha = ((self.close.rolling(3).mean() + self.close.rolling(6).mean() + self.close.rolling(12).mean()+self.close.rolling(24).mean())/4).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    
    ######################## alpha_154 #######################
    #       
    def alpha_154(self):
        # (((VWAP-MIN(VWAP,16)))<(CORR(VWAP,MEAN(VOLUME,180),18))) #
        alpha = (self.avg_price-self.avg_price.rolling(16).min()).iloc[-1,:]<self.avg_price.iloc[-18:,:].corrwith((self.volume.rolling(180).mean()).iloc[-18:,:])
        alpha=alpha.dropna()
        return alpha
    
    
    ######################## alpha_155 #######################
    #       
    def alpha_155(self):
        # SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2) #
        sma1 = self.volume.ewm(span=12, adjust=False).mean()
        sma2 = self.volume.ewm(span=26, adjust=False).mean()
        sma = (sma1-sma2).ewm(span=9, adjust=False).mean()
        alpha = sma.iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    
    ######################## alpha_156 #######################
    def alpha_156(self):
        # (MAX(RANK(DECAYLINEAR(DELTA(VWAP,5),3)),RANK(DECAYLINEAR(((DELTA(((OPEN*0.15)+(LOW*0.85)),2)/((OPEN*0.15)+(LOW*0.85)))*-1),3)))*-1 #
        rank1 = ((self.avg_price-self.avg_price.shift(5)).rolling(3).apply(self.func_decaylinear)).rank(axis=1, pct=True)
        rank2 = (-((self.open_price*0.15+self.low*0.85)-(self.open_price*0.15+self.low*0.85).shift(2))/(self.open_price*0.15+self.low*0.85)).rolling(3).apply(self.func_decaylinear).rank(axis=1, pct=True)
        max_cond = rank1 > rank2
        result = rank2
        result[max_cond] = rank1[max_cond]
        alpha = (-result).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    
    ######################## alpha_157 #######################
    #       
    def alpha_157(self):
        # (MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK((-1*RANK(DELTA((CLOSE-1),5))))),2),1)))),1),5)+TSRANK(DELAY((-1*RET),6),5)) #
        rank1 = (-((self.close-1)-(self.close-1).shift(5)).rank(axis=1, pct=True)).rank(axis=1, pct=True).rank(axis=1, pct=True)
        min1 = rank1.rolling(2).min()
        log1 = np.log(min1)
        rank2 = log1.rank(axis=1, pct=True).rank(axis=1, pct=True)
        cond_min = rank2 > 5
        rank2[cond_min] = 5
        tsrank1 = ((-((self.close/self.prev_close)-1)).shift(6)).rolling(5).apply(self.func_rank)
        alpha = (rank2+tsrank1).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    
    ######################## alpha_158 #######################
    #       
    def alpha_158(self):
        # ((HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2)))/CLOSE #
        alpha = (((self.high-self.close.ewm(span=14, adjust=False).mean())-(self.low-self.close.ewm(span=14, adjust=False).mean()))/self.close).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    
    def alpha_159(self):
        #########((close-sum(min(low,delay(close,1)),6))/sum(max(high,delay(close,1))-min(low,delay(close,1)),6)*12*24+(close-sum(min(low,delay(close,1)),12))/sum(max(high,delay(close,1))-min(low,delay(close,1)),12)*6*24+(close-sum(min(low,delay(close,1)),24))/sum(max(high,delay(close,1))-min(low,delay(close,1)),24)*6*24)*100/(6*12+6*24+12*24)
        ###################      
        data1=self.low
        data2=self.close.shift()
        cond=data1>data2
        data1[cond]=data2
        data3=self.high
        data4=self.close.shift()
        cond=data3>data4
        data3[~cond]=data4
        #计算出公式核心部分x
        x=((self.close-data1.rolling(6).sum())/(data2-data1).rolling(6).sum())*12*24
        #计算出公式核心部分y
        y=((self.close-data1.rolling(12).sum())/(data2-data1).rolling(12).sum())*6*24
        #计算出公式核心部分z
        z=((self.close-data1.rolling(24).sum())/(data2-data1).rolling(24).sum())*6*24
        data5=(x+y+z)*(100/(6*12+12*24+6*24))
        alpha=data5.iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    
    def alpha_160(self):
        ################      
        ############sma((close<=delay(close,1)?std(close,20):0),20,1)
        data1=self.close.rolling(20).std()
        cond=self.close<=self.close.shift(0)
        data1[~cond]=0
        data2=data1.ewm(span=39).mean()
        alpha=data2.iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    
    def alpha_161(self):
        ###########mean((max(max(high-low),abs(delay(close,1)-high)),abs(delay(close,1)-low)),12)
        ################      
        data1=(self.high-self.low)
        data2=pd.Series.abs(self.close.shift()-self.high)
        cond=data1>data2
        data1[~cond]=data2
        data3=pd.Series.abs(self.close.shift()-self.low)
        cond=data1>data3
        data1[~cond]=data3
        alpha=(data1.rolling(12).mean()).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha 
    
    
    def alpha_162(self):
        ###############(sma(max(close-delay(close,1),0),12,1)/sma(abs(close-delay(close,1)),12,1)*100-min(sma(max(close-delay(close,1),0),12,1)/sma(abs(close-delay(close,1)),12,1)*100,12))/(max(sma(max(close-delay(close,1),0),12,1)/sma(abs(close-delay(close,1)),12,1)*100),12)-min(sma(max(close-delay(close,1),0),12,1)/sma(abs(close-delay(close,1)),12,1)*100),12))
        #################      
        #算出公式核心部分X
        data1=self.close-self.close.shift()
        cond=data1>0
        data1[~cond]=0
        x=data1.ewm(span=23).mean()
        data2=pd.Series.abs(self.close-self.close.shift())
        y=data2.ewm(span=23).mean()
        z=(x/y)*100
        cond=z>12
        z[cond]=12
        c=(x/y)*100
        cond=c>12
        c[~cond]=12
        data3=(x/y)*100-(z/c)-c
        alpha=data3.iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    
    def alpha_163(self):
        ################      
        #######rank(((((-1*ret)*,ean(volume,20))*vwap)*(high-close)))
        data1=(-1)*(self.close/self.close.shift()-1)*self.volume.rolling(20).mean()*self.avg_price*(self.high-self.close)
        data2=(data1.rank(axis=1,pct=True)).iloc[-1,:]
        alpha=data2
        alpha=alpha.dropna()
        return alpha
    
    
    def alpha_164(self):
        ################      
        ############sma((((close>delay(close,1))?1/(close-delay(close,1)):1)-min(((close>delay(close,1))?1/(close/delay(close,1)):1),12))/(high-low)*100,13,2)
        cond=self.close>self.close.shift()
        data1=1/(self.close-self.close.shift())
        data1[~cond]=1
        data2=1/(self.close-self.close.shift())
        cond=data2>12
        data2[cond]=12
        data3=data1-data2/((self.high-self.low)*100)
        alpha=(data3.ewm(span=12).mean()).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    
    def alpha_167(self):
        ##      
        ####sum(((close-delay(close,1)>0)?(close-delay(close,1)):0),12)####
        data1=self.close-self.close.shift()
        cond=(data1<0)
        data1[cond]=0
        data2=(data1.rolling(12).sum()).iloc[-1,:]
        alpha=data2
        alpha=alpha.dropna()
        return alpha
    
    
    def alpha_168(self):
        ##      
        #####-1*volume/mean(volume,20)####
        data1=(-1*self.volume)/self.volume.rolling(20).mean()
        alpha=data1.iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    
    def alpha_169(self):
        ##      
        ###sma(mean(delay(sma(close-delay(close,1),9,1),1),12)-mean(delay(sma(close-delay(close,1),1,1),1),26),10,1)#####
        data1=self.close-self.close.shift()
        data2=(data1.ewm(span=17).mean()).shift()
        data3=data2.rolling(12).mean() - data2.rolling(26).mean()
        alpha=(data3.ewm(span=19).mean()).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha  
    
    
    def alpha_170(self):
        ##      
        #####((((rank((1/close))*volume)/mean(volume,20))*((high*rank((high-close)))/(sum(high,5)/5)))-rank((vwap-delay(vwap,5))))####
        data1=(1/self.close).rank(axis=0,pct=True)
        data2=self.volume.rolling(20).mean()
        x=(data1*self.volume)/data2
        data3=(self.high-self.close).rank(axis=0,pct=True)
        data4=self.high.rolling(5).mean()
        y=(data3*self.high)/data4
        z=(self.avg_price.iloc[-1,:]-self.avg_price.iloc[-5,:]).rank(axis=0,pct=True)
        alpha=(x*y-z).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    
    def alpha_171(self):
        ##      
        ####(((low-close)*open^5)*-1)/((close-high)*close^5)#####
        data1=-1*(self.low-self.close)*(self.open_price**5)
        data2=(self.close-self.high)*(self.close**5)
        alpha = (data1/data2).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    
    #       
    def alpha_172(self):
        # MEAN(ABS(SUM((LD>0&LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0&HD>LD)?HD:0,14)*100/(SUM((LD>0&LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM(TR,14)+SUM((HD>0&HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6) #
        hd = self.high-self.high.shift()
        ld = self.low.shift()-self.low
        temp1 = self.high-self.low
        temp2 = (self.high-self.close.shift()).abs()
        cond1 = temp1>temp2
        temp2[cond1] = temp1[cond1]
        temp3 = (self.low-self.close.shift()).abs()
        cond2 = temp2>temp3
        temp3[cond2] = temp2[cond2]
        tr = temp3   # MAX(MAX(HIGH-LOW,ABS(HIGH-DELAY(CLOSE,1))),ABS(LOW-DELAY(CLOSE,1)))
        sum_tr14 = tr.rolling(14).sum()
        cond3 = ld>0
        cond4 = ld>hd
        cond3[~cond4] = False
        data1 = ld
        data1[~cond3] = 0
        sum1 = data1.rolling(14).sum()*100/sum_tr14
        cond5 = hd>0
        cond6 = hd>ld
        cond5[~cond6] = False
        data2 = hd
        data2[~cond5] = 0
        sum2 = data2.rolling(14).sum() *100/sum_tr14
        alpha = ((sum1-sum2).abs()/(sum1+sum2)*100).rolling(6).mean().iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    
    def alpha_173(self):
        ##      
        ####3*sma(close,13,2)-2*sma(sma(close,13,2),13,2)+sma(sma(sma(log(close),13,2),13,2),13,2)#####
        data1=self.close.ewm(span=12).mean()
        data2=data1.ewm(span=12).mean()
        close_log=np.log(self.close)
        data3=close_log.ewm(span=12).mean()
        data4=data3.ewm(span=12).mean()
        data5=data4.ewm(span=12).mean()
        alpha=(3*data1-2*data2+data5).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    
    def alpha_174(self):
        ##      
        ####sma((close>delay(close,1)?std(close,20):0),20,1)#####
        cond=self.close>self.prev_close
        data2=self.close.rolling(20).std()
        data2[~cond] = 0
        alpha=(data2.ewm(span=39,adjust=False).mean()).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    
    def alpha_175(self):
        ##      
        #####mean(max(max(high-low),abs(delay(close,1)-high)),abs(delay(close,1)-low)),6)####
        data1=self.high-self.low
        data2=pd.Series.abs(self.close.shift()-self.high)
        cond=(data1>data2)
        data2[cond] = data1[cond]
        data3=pd.Series.abs(self.close.shift()-self.low)
        cond=(data2>data3)
        data3[cond] = data2[cond]
        data4=(data3.rolling(window=6).mean()).iloc[-1,:]
        alpha=data4
        alpha=alpha.dropna()
        return alpha
    
    
    def alpha_176(self):
        ##      
        ######### #########corr(rank((close-tsmin(low,12))/(tsmax(high,12)-tsmin(low,12))),rank(volume),6)#############
        data1=(self.close-self.low.rolling(window=12).min())/(self.high.rolling(window=12).max()-self.low.rolling(window=12).min())
        data2=data1.rank(axis=0,pct=True)
        #获取数据求出rank2
        data3=self.volume.rank(axis=0,pct=True)
        corr=data2.iloc[-6:,:].corrwith(data3.iloc[-6:,:])
        alpha=corr
        alpha=alpha.dropna()
        return alpha
    
    
    ################## alpha_177 ####################
    #       
    def alpha_177(self):
        ##### ((20-HIGHDAY(HIGH,20))/20)*100 #####
        alpha = (20 - self.high.iloc[-20:,:].apply(self.func_highday))/20*100
        alpha=alpha.dropna()
        return alpha
    
    
    def alpha_178(self):
        ##### (close-delay(close,1))/delay(close,1)*volume ####
        ##       
        alpha=((self.close-self.close.shift())/self.close.shift()*self.volume).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    
    def alpha_179(self):
        #####（rank(corr(vwap,volume,4))*rank(corr(rank(low),rank(mean(volume,50)),12))####
        ##       
        rank1=(self.avg_price.iloc[-4:,:].corrwith(self.volume.iloc[-4:,:])).rank(axis=0,pct=True)
        data2=self.low.rank(axis=0,pct=True)
        data3=(self.volume.rolling(window=50).mean()).rank(axis=0,pct=True)
        rank2=(data2.iloc[-12:,:].corrwith(data3.iloc[-12:,:])).rank(axis=0,pct=True)
        alpha=rank1*rank2
        alpha=alpha.dropna()
        return alpha 
    
    
    ##################### alpha_180 #######################
    #       
    def alpha_180(self):
        ##### ((MEAN(VOLUME,20)<VOLUME)?((-1*TSRANK(ABS(DELTA(CLOSE,7)),60))*SIGN(DELTA(CLOSE,7)):(-1*VOLUME))) #####
        ma = self.volume.rolling(window=20).mean()
        cond = (ma < self.volume).iloc[-20:,:]
        sign = delta_close_7 = self.close.diff(7)
        sign[sign.iloc[:,:]<0] = -1
        sign[sign.iloc[:,:]>0] = 1
        sign[sign.iloc[:,:]==0] = 0
        left = (((self.close.diff(7).abs()).iloc[-60:,:].rank(axis=0, pct=True)*(-1)).iloc[-20:,:] * sign.iloc[-20:,:]).iloc[-20:,:]
        right = self.volume.iloc[-20:,:]*(-1)
        right[cond] = left[cond]
        alpha = right.iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    
    def alpha_184(self):
        #####(rank(corr(delay((open-close),1),close,200))+rank((open-close))) ####
        ##       
        data1=self.open_price.shift()-self.close.shift()
        data2=self.open_price.iloc[-1,:] - self.close.iloc[-1,:]
        corr=data1.iloc[-200:,:].corrwith(self.close.iloc[-200:,:])
        alpha=data2.rank(axis=0,pct=True)+corr.rank(axis=0,pct=True)
        alpha=alpha.dropna()
        return alpha

    
    def alpha_185(self):
        ##### RANK((-1 * ((1 - (OPEN / CLOSE))^2))) ####
        alpha = (-(1-self.open_price/self.close)**2).rank(axis=1, pct=True).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    
    #       
    def alpha_186(self):
        # (MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)+DELAY(MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6),6))/2 #
        hd = self.high-self.high.shift()
        ld = self.low.shift()-self.low
        temp1 = self.high-self.low
        temp2 = (self.high-self.close.shift()).abs()
        cond1 = temp1>temp2
        temp2[cond1] = temp1[cond1]
        temp3 = (self.low-self.close.shift()).abs()
        cond2 = temp2>temp3
        temp3[cond2] = temp2[cond2]
        tr = temp3   # MAX(MAX(HIGH-LOW,ABS(HIGH-DELAY(CLOSE,1))),ABS(LOW-DELAY(CLOSE,1)))
        sum_tr14 = tr.rolling(14).sum()
        cond3 = ld>0
        cond4 = ld>hd
        cond3[~cond4] = False
        data1 = ld
        data1[~cond3] = 0
        sum1 = data1.rolling(14).sum()*100/sum_tr14
        cond5 = hd>0
        cond6 = hd>ld
        cond5[~cond6] = False
        data2 = hd
        data2[~cond5] = 0
        sum2 = data2.rolling(14).sum()*100/sum_tr14
        mean1 = ((sum1-sum2).abs()/(sum1+sum2)*100).rolling(6).mean()
        alpha = ((mean1 + mean1.shift(6))/2).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    
    def alpha_187(self):
        ##### SUM((OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))),20) ####
              
        cond = (self.open_price <= self.open_price.shift())
        data1 = self.high - self.low                        # HIGH-LOW
        data2 = self.open_price - self.open_price.shift()   # OPEN-DELAY(OPEN,1)
        cond_max = data2 > data1
        data1[cond_max] = data2[cond_max]
        data1[cond] = 0
        alpha = data1.iloc[-20:,:].sum()
        alpha=alpha.dropna()
        return alpha

    
    def alpha_188(self):
        ##### ((HIGH-LOW–SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2))*100 #####
              
        sma = (self.high - self.low).ewm(span=10, adjust=False).mean()
        alpha = ((self.high - self.low - sma)/sma*100).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    
    def alpha_189(self):
        ##### mean(abs(close-mean(close,6),6)) ####
        ma6 = self.close.rolling(window=6).mean()
        alpha = (self.close - ma6).abs().rolling(window=6).mean().iloc[-1,:]
        alpha=alpha.dropna()
        return alpha
    
    
    def alpha_191(self):
        ##### (CORR(MEAN(VOLUME,20), LOW, 5) + ((HIGH + LOW) / 2)) - CLOSE ####
        volume_avg = self.volume.rolling(window=20).mean()
        corr = volume_avg.iloc[-5:,:].corrwith(self.low.iloc[-5:,:])    
        alpha = corr + (self.high.iloc[-1,:] + self.low.iloc[-1,:])/2 - self.close.iloc[-1,:]
        alpha=alpha.dropna()
        return alpha



columns_security = data.sec_code.unique()
index_date = [pd.Timestamp(x) for x in data.date.unique()]
index_date.sort()
cal_alpha = CalAlpha(index_date[-1])
alpha_names = [method for method in dir(cal_alpha) if callable(getattr(cal_alpha, method)) if method[:5] == 'alpha']
alpha_names = [method for method in alpha_names if (int(method[-3:]) >= 140)]
alpha_names.sort()


os.chdir(r'D:\Xin\Program\Alpha_Trading\data\intermediate_data\factors')

for alpha_name in alpha_names:
    print(alpha_name)
    l = []
    for i in index_date[252: ]:
        cal_alpha = CalAlpha(i)
        alpha_value =  getattr(cal_alpha, alpha_name)().dropna().rename(alpha_name).to_frame().reset_index()
        alpha_value['date'] = i
        l.append(alpha_value)
    alpha_df = pd.concat(l, axis=0)
    if alpha_df.shape[0] != 0:
        alpha_df = alpha_df.set_index(['date', 'sec_code'])
    alpha_df.to_excel('{}.xlsx'.format(alpha_name))






