# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 01:34:11 2019

@author: abdav
"""

import numpy as np
import pandas as pd
import events as eve
import datetime as dt
import pandas_datareader as web
from sklearn.ensemble import RandomForestRegressor
import FeatureGeneration as fg
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#def main():
ticklist = ['ADS.DE', 'ALV.DE', 'BAS.DE', 'BEI.DE', 'BMW.DE', 'CON.DE', 'DAI.DE', 'DBK.DE', 'DTE.DE', 'EOAN.DE', 'FME.DE',
 'FRE.DE', 'HEI.DE', 'HEN3.DE', 'LHA.DE', 'LIN.DE', 'MRK.DE', 'MUV2.DE', 'RWE.DE', 'SAP.DE', 'SIE.DE', 'TKA.DE', 'VOW3.DE']
#parameters : 
delta=0.94
d1 = '1999-12-20'
d2 = '2019-01-05'

tickdict = dict()
Ydict = dict()

dtes = pd.read_csv('trading_days.csv', index_col=0)
Y = []
X = pd.DataFrame(columns=['Date', 'EMA10', 'EMA16', 'EMA22', 'SMA10', 'SMA16', 'SMA22','ValueAtRisk', 
                       'Bollu20', 'Bollu26', 'Bollu32', 'Bolld20', 'Bolld26', 'Bolld32',
                       'Mom12', 'Mom18', 'Mom24', 'ACC12', 'ACC18', 'ACC24', 'ROC10', 'ROC16',
                       'ROC22', 'MACD1812', 'MACD2412','MACD3012', 'MACDS18129', 'MACDS24129', 'MACDS30129', 
                       'RSI8', 'RSI14', 'RSI20', 'OBV', 'CHV1010', 'CHV1016', 'CHV1022',
                       'FastK12', 'FastD12', 'FastK18', 'SlowK12', 'FastD18', 'SlowD12',
                       'FastK24', 'SlowK18', 'FastD24', 'SlowD18', 'SlowK24', 'SlowD24',
                       'High', 'Low', 'Open', 'Close', 'Volume', 'AdjClose', 'Ticker',
                       'Month', 'DAX', 'ADL'])# Add CHO and ADL
dax = web.get_data_yahoo('^GDAXI', start=d1, end=d2)

for tick in ticklist[:4]:
    temp = pd.read_csv('tickData/'+ str(delta).replace('.', '') +'/' + tick.replace('.', '') +'.csv', index_col=0)
    #temp = fg.tick_data(tick, '2000-01-01','2005-01-01', delta=0.94, tocsv=True)
    tickdict[tick] = temp
    temp.set_index('Date', inplace=True, drop=True)
    temp.index = pd.to_datetime(temp.index)
    B = temp.loc[pd.to_datetime(dtes.Buy), 'AdjClose']
    S = temp.loc[pd.to_datetime(dtes.Sell), 'Open']
    mask = np.logical_not((np.isnan(B.values) | np.isnan(S.values)))
    B2 = B.values[mask]
    S2 = S.values[mask]
    y = S2-B2
    y = (y-y.mean())/y.std()
    Ydict[tick] = y
    Y = Y+list(y)
    ds = B[mask].index
    temp = temp.loc[pd.to_datetime(dtes.Buy), :]
    temp['Month'] = temp.index.month
    temp['Date'] = temp.index
    temp = temp.loc[mask, :]
    temp['DAX'] = dax.loc[ds, 'Adj Close']
    temp = temp.loc[:,['Date', 'EMA10', 'EMA16', 'EMA22', 'SMA10', 'SMA16', 'SMA22','ValueAtRisk', 
                       'Bollu20', 'Bollu26', 'Bollu32', 'Bolld20', 'Bolld26', 'Bolld32',
                       'Mom12', 'Mom18', 'Mom24', 'ACC12', 'ACC18', 'ACC24', 'ROC10', 'ROC16',
                       'ROC22', 'MACD1812', 'MACD2412','MACD3012', 'MACDS18129', 'MACDS24129', 'MACDS30129', 
                       'RSI8', 'RSI14', 'RSI20', 'OBV', 'CHV1010', 'CHV1016', 'CHV1022',
                       'FastK12', 'FastD12', 'FastK18', 'SlowK12', 'FastD18', 'SlowD12',
                       'FastK24', 'SlowK18', 'FastD24', 'SlowD18', 'SlowK24', 'SlowD24',
                       'High', 'Low', 'Open', 'Close', 'Volume', 'AdjClose', 'Ticker',
                       'Month', 'DAX', 'ADL']] ## Add ADL ad CHO
    for col in temp.columns:
        try:
            temp.loc[:,col] = (temp.loc[:,col].values-temp.loc[:,col].mean()) /temp.loc[:,col].std()
        except:
            pass
    X = pd.concat([X, temp], axis=0, ignore_index=True, sort=False) 
    

#X.dropna(axis=0, how=any, inplace=True)    
T = X.Ticker.unique()
tickdict = dict(zip(T, range(len(T))))
for tick in T:
    X.loc[X.Ticker==tick,'Ticker']= tickdict[tick]
    
X['Y'] = Y
X = X.loc[np.logical_not(X.DAX.isnull()),:]
X.sort_values(by='Date', inplace=True)
X.set_index('Date', drop=True, inplace=True)
C = ['EMA10', 'EMA16', 'EMA22', 'SMA10', 'SMA16', 'SMA22', 'ValueAtRisk',
       'Bollu20', 'Bollu26', 'Bollu32', 'Bolld20', 'Bolld26', 'Bolld32',
       'Mom12', 'Mom18', 'Mom24', 'ACC12', 'ACC18', 'ACC24', 'ROC10', 'ROC16',
       'ROC22', 'MACD1812', 'MACD2412', 'MACD3012', 'MACDS18129', 'MACDS24129',
       'MACDS30129', 'RSI8', 'RSI14', 'RSI20', 'OBV', 'CHV1010', 'CHV1016',
       'CHV1022', 'FastK12', 'FastD12', 'FastK18', 'SlowK12', 'FastD18',
       'SlowD12', 'FastK24', 'SlowK18', 'FastD24', 'SlowD18', 'SlowK24',
       'SlowD24', 'High', 'Low', 'Open', 'Close', 'Volume', 'AdjClose',
       'Ticker', 'Month', 'DAX', 'ADL']
Xtrain1 = X.loc[((X.index>pd.to_datetime('2001-01-01')) & (X.index<=pd.to_datetime('2004-01-01'))), C]
Xcv1 = X.loc[((X.index>pd.to_datetime('2004-01-01')) & (X.index<=pd.to_datetime('2005-01-01'))), C]
#Xtest1 = X.loc[((X.index>pd.to_datetime('2010-01-01')) & (X.index<=pd.to_datetime('2013-01-01'))), C]
Ytrain1 = X.loc[((X.index>pd.to_datetime('2001-01-01')) & (X.index<=pd.to_datetime('2004-01-01'))), 'Y']
Ycv1 = X.loc[((X.index>pd.to_datetime('2004-01-01')) & (X.index<=pd.to_datetime('2005-01-01'))), 'Y']
#Ytest1 = X.loc[((X.index>pd.to_datetime('2010-01-01')) & (X.index<=pd.to_datetime('2013-01-01'))), 'Y']
#Xtrain2
#Xcv2
#Xtest2
#Ytrain2
#Ycv2
#Ytest2

### Cross Validation for features:
#order features:
nfeatures=len(Xtrain1.columns)
rmse = []
rf = RandomForestRegressor(n_estimators=200).fit(Xtrain1, Ytrain1)
df2 = pd.DataFrame(index=Xtrain1.columns, columns=['rankval'])
df2.rankval = rf.feature_importances_
df2.sort_values(by='rankval', inplace=True, ascending=False)
yhat = rf.predict(Xcv1)
fig = plt.figure(figsize=(10,15))
plt.title('Variables relative importance')
plt.barh(np.arange(len(df2.index)),
         df2.rankval, align='center')
plt.yticks(range(len(df2.index)), df2.index.values.tolist())
plt.xlabel('Relative Importance')
plt.show()
rmse.append(mean_squared_error(yhat, Ycv1))
cols = df2.index.values[df2.index!=df2.index[-1]]
colused = df2.index.values
for i in range(1,len(df2.index)-11):
    Xtemp = Xtrain1.loc[:,cols]
    rf = RandomForestRegressor(n_estimators=200).fit(Xtemp, Ytrain1)
    df2 = pd.DataFrame(index=Xtemp.columns, columns=['rankval'])
    df2.rankval = rf.feature_importances_
    df2.sort_values(by='rankval', inplace=True, ascending=False)
    yhat = rf.predict(Xcv1.loc[:,cols])
    mse = mean_squared_error(yhat, Ycv1)
    rmse.append(mse)
    if mse == min(rmse): colused = cols    
    cols = df2.index.values[df2.index!=df2.index[-1]]
    
plt.scatter(range(10,30), rmse[::-1][:20])
plt.show()
print(colused)
    



'''




rf50=[]
rf100=[]
rf200=[]

first_day= Xtrain1.index.sort_values()[0]+dt.timedelta(200)
final_day= Xtrain1.index.sort_values()[-1]
d = first_day+dt.timedelta(51)
blast = True


while d <= final_day+dt.timedelta(1):
    X50_train = Xtrain1.loc[d+dt.timedelta(-51):d,:]
    Y50_train = Ytrain1[d+dt.timedelta(-51):d]
    rf = RandomForestRegressor(n_estimators=200).fit(X50_train, Y50_train)
    rf50.append((d, rf))
    if d > first_day + dt.timedelta(100):
        X100_train = Xtrain1.loc[d+dt.timedelta(-101):d,:]
        Y100_train = Ytrain1[d+dt.timedelta(-101):d]
        rf = RandomForestRegressor(n_estimators=200).fit(X100_train, Y100_train)
        rf100.append((d,rf))
    if d > first_day + dt.timedelta(200):
        X200_train = Xtrain1.loc[d+dt.timedelta(-201):d,:]
        Y200_train = Ytrain1[d+dt.timedelta(-201):d]
        rf = RandomForestRegressor(n_estimators=200).fit(X200_train, Y200_train)
        rf200.append((d,rf))
    dtemp = d+ dt.timedelta(50)
    if dtemp > final_day+dt.timedelta(1) and blast:
        d = dtemp
        blast = False
    else:
        d = d+ dt.timedelta(50)
        print(d)
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
''' 
    
    
    
#if __name__ == '__main__':
#    main()
