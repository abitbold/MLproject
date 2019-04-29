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
ticklist =  ['BMW.DE', 'CON.DE', 'DAI.DE', 'DBK.DE', 'DTE.DE', 'EOAN.DE', 'FME.DE','FRE.DE', 'HEI.DE']

#parameters : 
delta=0.94
d0 = '2001-01-01'
d1 = '2008-01-01'
d2 = '2010-01-01'
d3 = '2013-01-01'

norm=True
lbd = 0.85
alpha = 0.15

dtes = pd.read_csv('trading_days.csv', index_col=0)
tempdt = dtes.copy()
tempdt.set_index('Buy', drop=True, inplace=True)
tempdt.index = pd.to_datetime(tempdt.index)

Y = []
X = pd.DataFrame(columns=['Date', 'EMA10', 'EMA16', 'EMA22', 'SMA10', 'SMA16', 'SMA22','ValueAtRisk', 
                       'Bollu20', 'Bollu26', 'Bollu32', 'Bolld20', 'Bolld26', 'Bolld32',
                       'Mom12', 'Mom18', 'Mom24', 'ACC12', 'ACC18', 'ACC24', 'ROC10', 'ROC16',
                       'ROC22', 'MACD1812', 'MACD2412','MACD3012', 'MACDS18129', 'MACDS24129', 'MACDS30129', 
                       'RSI8', 'RSI14', 'RSI20', 'OBV', 'CHV1010', 'CHV1016', 'CHV1022',
                       'FastK12', 'FastD12', 'FastK18', 'SlowK12', 'FastD18', 'SlowD12',
                       'FastK24', 'SlowK18', 'FastD24', 'SlowD18', 'SlowK24', 'SlowD24',
                       'High', 'Low', 'Open', 'Close', 'Volume', 'AdjClose', 'Ticker',
                       'Month', 'DAX', 'ADL', 'Type', 'Y'])
dax = web.get_data_yahoo('^GDAXI', start=d0, end=d2)
prices = pd.DataFrame(columns=['Ticker', 'High', 'Low', 'Open', 'Close', 'Volume', 'AdjClose', 'Date'])

for tick in ticklist:
    if norm:
        temp = pd.read_csv('tickDataNorm/'+ str(delta).replace('.', '') +'/' + tick.replace('.', '') +'.csv', index_col=0)
    else:
        temp = pd.read_csv('tickData/'+ str(delta).replace('.', '') +'/' + tick.replace('.', '') +'.csv', index_col=0)
    prices = pd.concat([prices, temp.loc[:, ['Ticker', 'High', 'Low', 'Open', 'Close', 'Volume', 'AdjClose', 'Date']]],axis=0, ignore_index=True, sort=False,copy=True)
    #temp = fg.tick_data(tick, '2000-01-01','2005-01-01', delta=0.94)#, tocsv=True)
    temp.set_index('Date', inplace=True, drop=True)
    temp.index = pd.to_datetime(temp.index)
    B = temp.loc[pd.to_datetime(dtes.Buy), 'Close']
    S = temp.loc[pd.to_datetime(dtes.Sell), 'Close']
    temp = temp.loc[tempdt.index, :]
    temp['Y'] = 100*(S.values-B.values)/B.values
    
    mask = np.logical_not(np.isnan(temp['Y'].values))
    temp = temp.loc[mask, :]
    if norm:
        temp.loc[:,'High'] = temp.loc[:,'Norm_High']
        temp.loc[:,'Low'] = temp.loc[:,'Norm_Low']
        temp.loc[:,'Open'] = temp.loc[:,'Norm_Open']
        temp.loc[:,'Close'] = temp.loc[:,'Norm_Close']
        temp.loc[:,'AdjClose'] = temp.loc[:,'Norm_AdjClose']
    #temp = temp.loc[pd.to_datetime(dtes.Buy), :]
    temp['Month'] = temp.index.month
    temp['Date'] = temp.index
    temp['DAX'] = dax.loc[temp.index, 'Adj Close']
    temp.loc[:,'Type'] = tempdt.loc[mask, 'Type'].values
    temp = temp.loc[:,['Date', 'EMA10', 'EMA16', 'EMA22', 'SMA10', 'SMA16', 'SMA22','ValueAtRisk', 
                       'Bollu20', 'Bollu26', 'Bollu32', 'Bolld20', 'Bolld26', 'Bolld32',
                       'Mom12', 'Mom18', 'Mom24', 'ACC12', 'ACC18', 'ACC24', 'ROC10', 'ROC16',
                       'ROC22', 'MACD1812', 'MACD2412','MACD3012', 'MACDS18129', 'MACDS24129', 'MACDS30129', 
                       'RSI8', 'RSI14', 'RSI20', 'OBV', 'CHV1010', 'CHV1016', 'CHV1022',
                       'FastK12', 'FastD12', 'FastK18', 'SlowK12', 'FastD18', 'SlowD12',
                       'FastK24', 'SlowK18', 'FastD24', 'SlowD18', 'SlowK24', 'SlowD24',
                       'High', 'Low', 'Open', 'Close', 'Volume', 'AdjClose', 'Ticker',
                       'Month', 'DAX', 'ADL','Type', 'Y']]
    
    
    X = pd.concat([X, temp], axis=0, ignore_index=True, copy=True, sort=False)

'''
for col in ['EMA10', 'EMA16', 'EMA22', 'SMA10', 'SMA16', 'SMA22','ValueAtRisk', 
                       'Bollu20', 'Bollu26', 'Bollu32', 'Bolld20', 'Bolld26', 'Bolld32',
                       'Mom12', 'Mom18', 'Mom24', 'ACC12', 'ACC18', 'ACC24', 'ROC10', 'ROC16',
                       'ROC22', 'MACD1812', 'MACD2412','MACD3012', 'MACDS18129', 'MACDS24129', 'MACDS30129', 
                       'RSI8', 'RSI14', 'RSI20', 'OBV', 'CHV1010', 'CHV1016', 'CHV1022',
                       'FastK12', 'FastD12', 'FastK18', 'SlowK12', 'FastD18', 'SlowD12',
                       'FastK24', 'SlowK18', 'FastD24', 'SlowD18', 'SlowK24', 'SlowD24',
                       'High', 'Low', 'Open', 'Close', 'Volume', 'AdjClose', 'DAX', 'ADL']:
    try:
        X.loc[:,col] = (X.loc[:,col].values-X.loc[:,col].mean()) /X.loc[:,col].std()
    except:
        pass
'''

T = X.Ticker.unique()
tickdict = dict(zip(T, range(len(T))))
for tick in T:
    X.loc[X.Ticker==tick,'Ticker']= tickdict[tick]

#X['Y'] = Y
#X = X.loc[np.logical_not(X.DAX.isnull()),:]

X.sort_values(by=['Date', 'Ticker'], inplace=True)
X.set_index('Date', drop=True, inplace=True)
X = X.loc[((X.index>=pd.to_datetime(d0)) & (X.index<=pd.to_datetime(d2))), :]
C = ['EMA10', 'EMA16', 'EMA22', 'SMA10', 'SMA16', 'SMA22', 'ValueAtRisk',
       'Bollu20', 'Bollu26', 'Bollu32', 'Bolld20', 'Bolld26', 'Bolld32',
       'Mom12', 'Mom18', 'Mom24', 'ACC12', 'ACC18', 'ACC24', 'ROC10', 'ROC16',
       'ROC22', 'MACD1812', 'MACD2412', 'MACD3012', 'MACDS18129', 'MACDS24129',
       'MACDS30129', 'RSI8', 'RSI14', 'RSI20', 'OBV', 'CHV1010', 'CHV1016',
       'CHV1022', 'FastK12', 'FastD12', 'FastK18', 'SlowK12', 'FastD18',
       'SlowD12', 'FastK24', 'SlowK18', 'FastD24', 'SlowD18', 'SlowK24',
       'SlowD24', 'High', 'Low', 'Open', 'Close', 'Volume', 'AdjClose',
       'Month', 'DAX', 'ADL','Type', 'Ticker']
Xtrain1 = X.loc[((X.index>pd.to_datetime(d0)) & (X.index<=pd.to_datetime(d1))), C]
Xcv1 = X.loc[((X.index>pd.to_datetime(d1)) & (X.index<=pd.to_datetime(d2))), C]
#Xtest1 = X.loc[((X.index>pd.to_datetime('2010-01-01')) & (X.index<=pd.to_datetime('2013-01-01'))), C]
Ytrain1 = X.loc[((X.index>pd.to_datetime(d0)) & (X.index<=pd.to_datetime(d1))), 'Y']
Ycv1 = X.loc[((X.index>pd.to_datetime(d1)) & (X.index<=pd.to_datetime(d2))), 'Y']

'''
#Code for number of trees
oob = []

xa = []

for i in range(1, 300):
    print(i)
    rf = RandomForestRegressor(n_estimators=i, oob_score=True).fit(Xtrain1, Ytrain1)
    oob.append(1 - rf.oob_score_)
    xa.append(i)
plt.scatter(xa, oob)
--> 100 tress seem enough
'''

    
### Cross Validation for features:
#order features:
nfeatures=len(Xtrain1.columns)
rmse = []
colsrmse =[]
rf = RandomForestRegressor(n_estimators=200, max_features=None).fit(Xtrain1, Ytrain1)
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
rmse.append(np.sqrt(mean_squared_error(yhat, Ycv1)))
colsrmse.append(df2.index)
CL = []
CL.append(df2.index.values)
cols = df2.index.values[df2.index!=df2.index[-1]]

for i in range(1,len(df2.index)):
    Xtemp = Xtrain1.loc[:,cols]
    CL.append(cols)
    rf = RandomForestRegressor(n_estimators=200, max_features=None).fit(Xtemp, Ytrain1)
    df2 = pd.DataFrame(index=Xtemp.columns, columns=['rankval'])
    df2.rankval = rf.feature_importances_
    df2.sort_values(by='rankval', inplace=True, ascending=False)
    yhat = rf.predict(Xcv1.loc[:,cols])
    mse = mean_squared_error(yhat, Ycv1)
    rmse.append(np.sqrt(mse))
    colsrmse.append(df2.index) 
    cols = df2.index.values[df2.index!=df2.index[-1]]
    
plt.scatter(range(len(rmse)), np.array(rmse[::-1]))

plt.show()
#Gives features we use
colused = CL[::-1][np.argmin(rmse[::-1][10:30])+10]


## Train
#X.sort_values(by=['Date','Ticker'], inplace=True)


rfL = []
dtlist = X.index.unique().sort_values()
first_day= dtlist[0]
d = first_day+dt.timedelta(50)


while d < pd.to_datetime(d2):
    X50_train = X.loc[((X.index>=d+dt.timedelta(-50)) & (X.index<d)),colused]
    Y50_train = X.loc[((X.index>=d+dt.timedelta(-50)) & (X.index<d)),'Y']
    rf = RandomForestRegressor(n_estimators=200, max_features=None).fit(X50_train, Y50_train)
    rfL.append((d, rf))
    if d >= first_day + dt.timedelta(100):
        X100_train = X.loc[((X.index>=d+dt.timedelta(-100)) & (X.index<d)),colused]
        Y100_train = X.loc[((X.index>=d+dt.timedelta(-100)) & (X.index<d)),'Y']
        rf = RandomForestRegressor(n_estimators=200, max_features=None).fit(X100_train, Y100_train)
        rfL.append((d,rf))
    if d >= first_day + dt.timedelta(200):
        X200_train = X.loc[((X.index>=d+dt.timedelta(-200)) & (X.index<d)),colused]
        Y200_train = X.loc[((X.index>=d+dt.timedelta(-200)) & (X.index<d)),'Y']
        rf = RandomForestRegressor(n_estimators=200, max_features=None).fit(X200_train, Y200_train)
        rfL.append((d,rf))
    d = d+ dt.timedelta(50)

'''
#Weights1
lbd=0.85
def calc_ki (lbd, ki1, ri):
    return lbd*ri*100 + (1-lbd)*ki1


    
w = np.array([0.0]*len(rfL))
k = np.array([0.0]*len(rfL))
r = np.array([0.0]*len(rfL))
c = np.array([0.0]*len(rfL))
rmse = np.ones(len(rfL))
mn = np.array([0]*len(rfL))
mn[0]=1
w0= []
w1= []
w2= []
w3= []
w4= []

nrf=0
feats = colused
d = dtlist[dtlist>=rfL[0][0]][0] 
prevT=1000
nrfprev = 0
while d < pd.to_datetime(d1):
    nrfprev = nrf
    T = d - first_day
    T50 = (T // 50).days
    if T50==1:
        nrf=1
    elif T50 == 2:
        nrf=3
    elif T50 == 3:
        nrf = 5
        prevT=3
    elif T50 > prevT:
        nrf +=3
        prevT = T50
    #if nrf==1:
    T1 = d - rfL[0][0]
    try:
        l = len(X.loc[d,feats].columns)
        k[0] = calc_ki(lbd, k[0], r[0])
        if w[0] != 0 : w[0] = np.exp(k[0]/np.sqrt(T1.days)) * w[0] #(1/rmse[0])
        else: w[0] = 1
        rp = rfL[0][1].predict(X.loc[d,feats])
        BS =(rp / np.abs(rp).sum())
        r[0] = ((BS * X.loc[d,'Y'].values/100).sum() - (1/len(X.loc[d,'Y'].values)) * (X.loc[d,'Y'].values/100).sum())
        rmse[0] = 0.7*rmse[0] + np.sqrt(mean_squared_error(rp, X.loc[d,'Y']))
    except:
        d=pd.to_datetime(d)
        k[0] = calc_ki(lbd, k[0], r[0])
        if w[0] != 0 : w[0] = np.exp(k[0]/np.sqrt(T1.days)) * w[0] #(1/rmse[0])
        else: w[0] = 1
        rp = rfL[0][1].predict(pd.DataFrame(X.loc[d,feats].values.reshape(1,-1)))
        r[0] = ((np.sign(rp) * X.loc[d,'Y']/100) - X.loc[d,'Y']/100)
        rmse[0] = 0.7*rmse[0] + np.sqrt(mean_squared_error(rp, [X.loc[d,'Y']]))
    #else:
    for i in range(1, nrf):
        T1 = d - rfL[i][0]
        if i<nrfprev:
            try:
                l = len(X.loc[d,feats].columns)
                d=pd.to_datetime(d)
                k[i] = calc_ki(lbd, k[i], r[i])
                w[i] = np.exp(k[i]/np.sqrt(T1.days))  * w[i] #(1/rmse[i])  # w[:mn[i]].mean()# *min((d-rfL[i][0]).days/50, 1)# * w[:nrfprev].mean()  
                rp = rfL[i][1].predict(X.loc[d,feats])
                BS =(rp / np.abs(rp).sum())
                r[i] = ((BS * X.loc[d,'Y'].values/100).sum() - (1/len(X.loc[d,'Y'].values)) * (X.loc[d,'Y'].values/100).sum())
                rmse[i] = 0.7*rmse[i] + np.sqrt(mean_squared_error(rp, X.loc[d,'Y']))
            except:
                d=pd.to_datetime(d)
                k[i] = calc_ki(lbd, k[i], r[i])
                w[i] = np.exp(k[i]/np.sqrt(T1.days)) * w[i] #(1/rmse[i]) # w[:mn[i]].mean() #*min((d-rfL[i][0]).days/50, 1) #* w[:nrfprev].mean()
                rp = rfL[i][1].predict(pd.DataFrame(X.loc[d,feats].values.reshape(1,-1)))
                r[i] = ((np.sign(rp) * X.loc[d,'Y']/100) - X.loc[d,'Y']/100)
                rmse[i] = 0.7*rmse[i] + np.sqrt(mean_squared_error(rp, [X.loc[d,'Y']]))
        else:
            try:
                l = len(X.loc[d,feats].columns)
                d=pd.to_datetime(d)
                mn[i] = int(nrfprev)
                w[i] = 1 #w[:nrfprev].mean()
                rp = rfL[i][1].predict(X.loc[d,feats])
                BS =(rp / np.abs(rp).sum())
                r[i] = ((BS * X.loc[d,'Y'].values/100).sum() -(1/len(X.loc[d,'Y'].values/100)) * (X.loc[d,'Y'].values/100).sum()) 
                k[i] = r[i]
                rmse[i] = 0.7*np.sqrt(mean_squared_error(rp, X.loc[d,'Y']))
            except:
                d=pd.to_datetime(d)
                mn[i] = int(nrfprev)
                w[i] = 1 #w[:nrfprev].mean()
                rp = rfL[i][1].predict(pd.DataFrame(X.loc[d,feats].values.reshape(1,-1)))
                r[i] = ((np.sign(rp) * X.loc[d,'Y']/100) - X.loc[d,'Y']/100)
                k[i] = r[i]
                rmse[i] = 0.7*np.sqrt(mean_squared_error(rp, [X.loc[d,'Y']]))
            

    w0.append(w[0])
    w1.append(w[1])
    w2.append(w[5])
    w3.append(w[10])
    w4.append(w[15])
    #print(w[0], w[1], w[5], w[10], w[15])
    d = dtlist[dtlist>d][0]
print(w)           
            
xa=range(len(w0))
plt.figure(figsize=(15,15))
plt.plot(xa, w0, label='1')
plt.plot(xa, w1, label='2')
plt.plot(xa, w2, label='3')
plt.plot(xa, w3, label='4')
plt.plot(xa, w4, label='5')
#plt.ylim(0.99, 1.05)
plt.legend()
plt.show()

'''


#Weights2
lbd = 0.85
lbd2 = 1
def calc_ki (lbd, ki1, ri):
    return lbd*ri*100 + (1-lbd)*ki1

w = np.zeros(len(rfL))
k = np.array([0.0]*len(rfL))
r = np.array([0.0]*len(rfL))
c = np.array([0.0]*len(rfL))
mn = np.array([0]*len(rfL))
rmse = np.ones(len(rfL))
mn[0]=1
w0= []
w1= []
w2= []
w3= []
w4= []
w5 = []

nrf=0
feats = colused
d = dtlist[dtlist>=rfL[0][0]][0] 
prevT=1000
nrfprev = 0
P = []
ntick = len(X.loc[:, 'Ticker'].unique())
B = np.zeros(ntick)
DL = pd.DataFrame(columns = range(ntick))
predictions = pd.DataFrame(columns = range(ntick))
portfolio = pd.DataFrame(columns=['P'], index=X.loc[((X.index>=pd.to_datetime(d1)) & (X.index<=pd.to_datetime(d2))),:].index.unique().sort_values())
j=0
cl = np.zeros(len(rfL))
while d < pd.to_datetime(d2):
    nrfprev = nrf
    T = d - first_day
    T50 = (T // 50).days
    if T50==1:
        nrf=1
    elif T50 == 2:
        nrf=3
    elif T50 == 3:
        nrf = 5
        prevT=3
    elif T50 > prevT:
        nrf +=3
        prevT = T50
    #if nrf==1:
    T1 = d - rfL[0][0]
    try:
        l = len(X.loc[d,feats].columns)
        k[0] = calc_ki(lbd, k[0], r[0])
        if w[0] != 0 : w[0] = np.exp(k[0]/np.sqrt(T1.days)) * (w[0] * lbd2 + (1-lbd2) * w[:nrfprev].mean()) #(1/rmse[0])
        else: w[0] = 1
        rp = rfL[0][1].predict(X.loc[d,feats])
        BS =(rp / np.abs(rp).sum())
        r[0] = ((BS * X.loc[d,'Y'].values/100).sum() - (1/len(X.loc[d,'Y'].values)) * (X.loc[d,'Y'].values/100).sum())
        #rmse[0] = 0.7*rmse[0] + np.sqrt(mean_squared_error(rp, X.loc[d,'Y']))
    except:
        d=pd.to_datetime(d)
        k[0] = calc_ki(lbd, k[0], r[0])
        if w[0] != 0 : w[0] = np.exp(k[0]/np.sqrt(T1.days)) * (w[0]* lbd2 + (1-lbd2) * w[:nrfprev].mean())  #(1/rmse[0])
        else: w[0] = 1
        rp = rfL[0][1].predict(pd.DataFrame(X.loc[d,feats].values.reshape(1,-1)))
        r[0] = ((np.sign(rp) * X.loc[d,'Y']/100) - X.loc[d,'Y']/100)
        #rmse[0] = 0.7*rmse[0] + np.sqrt(mean_squared_error(rp, [X.loc[d,'Y']]))
    try:
        temp2 = rfL[i][1].predict(X.loc[d,feats])
        temp = np.zeros(ntick)
        for tick in range(ntick):
            temp[tick] = temp2[X.loc[d,'Ticker'].values == tick].sum()
    except:
        temp2 = rfL[i][1].predict(X.loc[d,feats].values.reshape(1, -1))
        for tick in range(ntick):
            temp[tick] = temp2[X.loc[d,'Ticker'] == tick].sum()
    P.append(temp)
    #else:
    for i in range(1, nrf):
        T1 = d - rfL[i][0]
        if i<nrfprev:
            try:
                l = len(X.loc[d,feats].columns)
                d=pd.to_datetime(d)
                k[i] = calc_ki(lbd, k[i], r[i])
                w[i] = np.exp(k[i]/np.sqrt(T1.days))  * (w[i] * lbd2 + (1-lbd2) * w[:nrfprev].mean())  #(1/rmse[i])  # w[:mn[i]].mean()# *min((d-rfL[i][0]).days/50, 1)# * w[:nrfprev].mean()  
                rp = rfL[i][1].predict(X.loc[d,feats])
                BS =(rp / np.abs(rp).sum())
                r[i] = ((BS * X.loc[d,'Y'].values/100).sum() - (1/len(X.loc[d,'Y'].values)) * (X.loc[d,'Y'].values/100).sum())
                #rmse[i] = 0.7*rmse[i] + np.sqrt(mean_squared_error(rp, X.loc[d,'Y']))
            except:
                d=pd.to_datetime(d)
                k[i] = calc_ki(lbd, k[i], r[i])
                w[i] = np.exp(k[i]/np.sqrt(T1.days)) * (w[i] * lbd2 + (1-lbd2) * w[:nrfprev].mean() ) #(1/rmse[i]) # w[:mn[i]].mean() #*min((d-rfL[i][0]).days/50, 1) #* w[:nrfprev].mean()
                rp = rfL[i][1].predict(pd.DataFrame(X.loc[d,feats].values.reshape(1,-1)))
                r[i] = ((np.sign(rp) * X.loc[d,'Y']/100) - X.loc[d,'Y']/100)
                #rmse[i] = 0.7*rmse[i] + np.sqrt(mean_squared_error(rp, [X.loc[d,'Y']]))
        else:
            try:
                l = len(X.loc[d,feats].columns)
                d=pd.to_datetime(d)
                mn[i] = int(nrfprev)
                w[i] = 1 #w[:nrfprev].mean()
                rp = rfL[i][1].predict(X.loc[d,feats])
                BS =(rp / np.abs(rp).sum())
                r[i] = ((BS * X.loc[d,'Y'].values/100).sum() -(1/len(X.loc[d,'Y'].values/100)) * (X.loc[d,'Y'].values/100).sum()) 
                k[i] = r[i]
                #rmse[i] = 0.7*np.sqrt(mean_squared_error(rp, X.loc[d,'Y']))
            except:
                d=pd.to_datetime(d)
                mn[i] = int(nrfprev)
                w[i] = 1 #w[:nrfprev].mean()
                rp = rfL[i][1].predict(pd.DataFrame(X.loc[d,feats].values.reshape(1,-1)))
                r[i] = ((np.sign(rp) * X.loc[d,'Y']/100) - X.loc[d,'Y']/100)
                k[i] = r[i]
                #rmse[i] = 0.7*np.sqrt(mean_squared_error(rp, [X.loc[d,'Y']]))
        try:
            temp2 = rfL[i][1].predict(X.loc[d,feats])
            temp = np.zeros(ntick)
            for tick in range(ntick):
                temp[tick] = temp2[X.loc[d,'Ticker'].values == tick].sum()
        except:
            temp2 = rfL[i][1].predict(X.loc[d,feats].values.reshape(1, -1))
            temp = np.zeros(ntick)
            for tick in range(ntick):
                temp[tick] = temp2[X.loc[d,'Ticker'] == tick].sum()
        P.append(temp)
        
    if d >= pd.to_datetime(d1):   
        P = pd.DataFrame(P)
        predictions.loc[d, :] = (P.values * np.array([w[:nrf].tolist()]*ntick).transpose()).sum(axis=0)/w.sum()
        for tick in  range(ntick):
            temp = P.loc[:, tick]
            if (temp.values>0).sum()>0 : B[tick] = w[:nrf][temp.values>0].sum()/w.sum()
            else : B[tick]=0
        S = 1-B
        D = B-S
        D[np.abs(D)<alpha] = 0
        DL.loc[d, :] = D
        if j == 0 : c =1
        else : c = portfolio.iloc[j-1,0]
        wd = predictions.loc[d, :].values/predictions.loc[d, :].abs().sum()
        ret = np.zeros(ntick)
        if len(X.loc[d,'Y']!=ntick):
            for tick in range(ntick):
                ret[tick] = X.loc[((X.index==d) & (X.Ticker==tick)),'Y'].values[0]/100
        else:
            ret = X.loc[d,'Y']/100
        profit = (wd * c * ret).sum()
        portfolio.iloc[j, 0] = c+profit
        j+=1
    w0.append(w[0])
    w1.append(w[10])
    w2.append(w[20])
    w3.append(w[70])
    w4.append(w[79])
    w5.append(w[21])
    #print(w[0], w[1], w[5], w[10], w[15])
    P=[]
    try:
        d = dtlist[dtlist>d][0]
    except:
        break
              
            
xa=range(len(w0))
plt.figure(figsize=(15,15))
plt.plot(xa, w0, label='1')
plt.plot(xa, w1, label='2')
plt.plot(xa, w2, label='3')
plt.plot(xa, w3, label='4')
plt.plot(xa, w4, label='5')
plt.plot(xa, w5, LABEL='6')
#plt.ylim(0.99, 1.05)
plt.legend()
plt.show()

'''
# Code for weights 1
P = []
ntick = len(X.loc[:, 'Ticker'].unique())
B = np.zeros(ntick)
DL = pd.DataFrame(columns = range(ntick))
d = X.loc[X.index>=pd.to_datetime(d1),:].index.sort_values()[0] 
predictions = pd.DataFrame(columns = range(ntick))

while d < pd.to_datetime(d2):
    for i in range(nrf):
        temp2 = rfL[i][1].predict(X.loc[d,feats])
        temp = np.zeros(ntick)
        for tick in range(ntick):
            temp[tick] = temp2[X.loc[d,'Ticker'].values == tick].sum()
        P.append(temp)
    P = pd.DataFrame(P)
    predictions.loc[d, :] = (P.values * np.array([w.tolist()]*ntick).transpose()).sum(axis=0)/w.sum()
    for tick in  range(ntick):
        temp = P.loc[:, tick]
        B[tick] = w[temp.values>0].sum()/w.sum()
    S = 1-B
    D = B-S
    DL.loc[d, :] = D
    if len(dtlist[dtlist>d])>0:
        d = dtlist[dtlist>d][0]
        P = []
    else:
        break


dl2=DL.abs()
alpha = dl2.mean().mean() - dl2.values.std()
val = predictions.values
val[dl2.values<alpha] = 0
predictions = pd.DataFrame(val, index = predictions.index, columns=predictions.columns)
d = X.loc[X.index>=pd.to_datetime(d1),:].index.sort_values()[0] 
portfolio = pd.DataFrame(columns=['P'], index=X.loc[((X.index>=pd.to_datetime(d1)) & (X.index<=pd.to_datetime(d2))),:].index.unique().sort_values())

j=0
while d < pd.to_datetime(d2):
    if j == 0 : c =1
    else : c = portfolio.iloc[j-1,0]
    wd = predictions.loc[d, :].values/predictions.loc[d, :].abs().sum()
    ret = np.zeros(ntick)
    if len(X.loc[d,'Y']!=ntick):
        for tick in range(ntick):
            ret[tick] = X.loc[((X.index==d) & (X.Ticker==tick)),'Y'].values[0]/100
    else:
        ret = X.loc[d,'Y']/100
    profit = (wd * c * ret).sum()
    portfolio.iloc[j, 0] = c+profit
    j+=1
    if len(dtlist[dtlist>d])>0:
        d = dtlist[dtlist>d][0]
    else:
        break


    
    
 '''
    
rethold = np.zeros(ntick)
for tick in tickdict.keys():
    prices.Date = pd.to_datetime(prices.Date)
    prices.loc[prices.Ticker==tick,'Ticker'] = tickdict[tick]
    t2 = tickdict[tick]
    p1 = prices.loc[((prices.Date>=pd.to_datetime(d1)) & (prices.Date<pd.to_datetime(d2)) & (prices.Ticker==t2)), 'Close'].values[0]
    p2 = prices.loc[((prices.Date>=pd.to_datetime(d1)) & (prices.Date<pd.to_datetime(d2)) & (prices.Ticker==t2)), 'Open'].values[-1]
    rethold[t2] = p2/p1-1
    
hold = rethold.mean()    
    
    
    
    
    
    
    
    
    
    

    
    
    
#if __name__ == '__main__':
#    main()
