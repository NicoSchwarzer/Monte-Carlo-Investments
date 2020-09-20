# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 22:41:51 2020

@author: Nico
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 16 14:09:36 2020

@author: Nico
"""

## Optimierungsalgorithmus


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quandl

start = pd.to_datetime('2011-01-01')
end = pd.to_datetime('2018-01-01')


quandl.ApiConfig.api_key = "..."
# can't show it here :D


# amazon
amzn = quandl.get('WIKI/AMZN.11',start_date=start,end_date=end)

## evonik
evo = quandl.get('FSE/EVK_X',start_date=start,end_date=end)

## deutsche wohnen
dw = quandl.get('FSE/DWNI_X',start_date=start,end_date=end)

## Siemens
sie = quandl.get('FSE/SIE_X',start_date=start,end_date=end)

## SAP
sap = quandl.get('FSE/SAP_X',start_date=start,end_date=end)

# USA AA -rated bonds, price!!
us_aa = quandl.get('ML/AATRI', start_date=start,end_date=end)

# emerging markets bonds price
em = quandl.get('ML/EMCTRI', start_date=start,end_date=end)

# US-corporate BBB
us_bbb = quandl.get('ML/BBBTRI', start_date=start,end_date=end)


amzn = pd.DataFrame(amzn)

evo = pd.DataFrame(evo["Close"])
dw = pd.DataFrame(dw["Close"])
sie = pd.DataFrame(sie["Close"])
sap = pd.DataFrame(sap["Close"])
us_aa = pd.DataFrame(us_aa)
em = pd.DataFrame(em)
us_bbb = pd.DataFrame(us_bbb)

em.head()
sap.head()
us_bbb.head()
us_aa.head()
sie.head()
dw.head()
evo.head()
amzn.head()


df1 = pd.concat([amzn, dw, sie, sap, us_aa, em, us_bbb], axis = 1)

df1.head()
df1.tail()

df1.columns = ["amzn", "dw", "sie", "sap", "us_aa", "em", "us_bbb"]


df2 = np.log(df1/df1.shift(1))
df2.head()

# alle NAN löschen
df3 = df2.iloc[1: , :]
df3.dropna()
df3.head()

df3.to_csv('C:\\Users\\Nico\Documents\Machine Learning\Projekte\Monte Carlo Anlagen\df3.csv', sep='\t')


############################
## Einmal mit Monte-Carlo ##
############################

df3 = pd.read_csv('C:\\Users\\Nico\Documents\Machine Learning\Projekte\Monte Carlo Anlagen\df3.csv', sep='\t')

df3.columns = ["date", "amzn", "dw", "sie", "sap", "us_aa", "em", "us_bbb"]

df3 = df3.drop('date', 1)

## One possible allocation
# mean return without risk -> return of US AA
mean_us_aa = np.mean(df3["us_aa"])
print(mean_us_aa)



## weights for the 7 investments

w1 = np.random.random(7)
w2 = w1/np.sum(w1)

# expected DAILY return
# multiply df3 (matrix) with w2 (vector)
df4 = df3*w2

df4.head()

# return
ret = df4["amzn"] + df4["dw"] + df4["sie"] +  df4["sap"] + df4["us_aa"] + df4["em"] + df4["us_bbb"]

print(ret)

# return
mean_ret = np.mean(ret)
mean_ret

# volatility
vol = np.std(ret)
vol

# (daily) sharpe
sharpe = (mean_ret - mean_us_aa) / vol
sharpe

##################################
### looping -> real Monte Carlo ##
##################################

ra = 2000

# creating empty matrices and vectors to be filled
cols = len(df3.columns)


weights = np.ones((num, cols))

def to_ones(x):
    x1 = np.ones(x)
    return x1


mean_return = to_ones(num)
vola = to_ones(num)
sharpe_ratios = to_ones(num)

for i in range(ra):
    
    ## weights for the 7 investments
    w1 = np.random.random(7)
    w2 = w1/np.sum(w1)

    weights[i, :] = w2
    
    # expected DAILY return
    # multiply df3 (matrix) with w2 (vector)
    df4 = df3*w2
    ret = df4["amzn"] + df4["dw"] + df4["sie"] +  df4["sap"] + df4["us_aa"] + df4["em"] + df4["us_bbb"]

    mean_ret = np.mean(ret)
    mean_return[i] = mean_ret
    
    vol = np.std(ret)

    sharpe = (mean_ret - mean_us_aa) / vol
    sharpe_ratios[i] = sharpe


dfw1 = pd.DataFrame(weights)
dfw1.head()
dfw1.columns = ["amzn", "dw", "sie", "sap", "us_aa", "em", "us_bbb"]
dfw1["ret"] = mean_return
dfw1["SR"] = sharpe_ratios


dfw1["SR"].max()
dfw1["SR"].argmax()

dfw1.iloc[718, :]


# daily reuturn of 0.000377 (logarithmic)

# If one invested 1000 € ... 
num_days = df3.shape[0]

total_ret = 1000 * (1 + dfw1["ret"].iloc[718]) ** num_days
total_ret


