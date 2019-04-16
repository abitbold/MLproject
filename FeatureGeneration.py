import numpy as np
import pandas as pd
import pandas_datareader as web

a = web.get_data_yahoo('ADS', start='2019-01-01', end='today')


df = pd.DataFrame(columns = ['EMA10','EMA16', 'EMA22', 'SMA10', 'SMA16', 'SMA22', 'Return', 'Variance', 'ValueAtRisk', 'VarScalar',
                            'SMA20', 'SMA26', 'SMA32', 'Bollu20','Bollu26', 'Bollu32', 'Bolld20', 'Bolld26', 'Bolld32', 'Mom12', 
                             'Mom18', 'Mom24', 'ACC12', 'ACC18', 'ACC24', 'ROC10', 'ROC16', 'ROC22', 'EMA12', 'EMA18', 'EMA24',
                            'EMA30', 'MACD1812', 'MACD2412', 'MACD3012'], 
                  index = range(len(a.index)))

df['Date'] = a.index
### parameters ###

delta = 0.94

### EMA ###
df.loc[0,'EMA10'] = a.iloc[0,5] #regular EMA 10
df.loc[0,'EMA16'] = a.iloc[0,5] #regular EMA 16
df.loc[0,'EMA22'] = a.iloc[0,5] #regular EMA 22
df.loc[0,'EMA12'] = a.iloc[0,5] #regular EMA 12
df.loc[0,'EMA18'] = a.iloc[0,5] #regular EMA 18
df.loc[0,'EMA24'] = a.iloc[0,5] #regular EMA 24
df.loc[0,'EMA30'] = a.iloc[0,5] #regular EMA 30


### SMA ###
df.loc[0,'SMA10'] = a.iloc[0,5]
df.loc[0,'SMA16'] = a.iloc[0,5]
df.loc[0,'SMA22'] = a.iloc[0,5]
df.loc[0,'SMA20'] = a.iloc[0,5]
df.loc[0,'SMA26'] = a.iloc[0,5]
df.loc[0,'SMA32'] = a.iloc[0,5]

### Returns ###
df.iloc[1:,6] = (a.iloc[1:,5].values-a.iloc[0:-1,5].values)/a.iloc[0:-1,5].values

### Variance ###

df.iloc[0,7] = np.power(df.iloc[1:32,6].std(),2)

### Value at Risk ###

df.iloc[0,8] = 1.96*np.sqrt(df.iloc[0,7])
df.iloc[0,9] = 1

### ADL ###
df.loc[0,'ADL'] = a.iloc[0,4]*\
                ((a.iloc[0,3] - a.iloc[0,1])-(a.iloc[0,0]-a.iloc[0,3]))/\
                (a.iloc[0,0]-a.iloc[0,1])

### OBV ###
df.loc[0,'OBV'] = a.iloc[0,4]

### EMADL ###
df.loc[0,'EMADL3'] = df.loc[0,'ADL']
df.loc[0,'EMADL10'] = df.loc[0,'ADL']

### EMAHL ### 
df.loc[0,'EMAHL10'] = a.iloc[0,0]-a.iloc[0,1]
df.loc[0, 'EMAHL16'] = a.iloc[0,0]-a.iloc[0,1]
df.loc[0, 'EMAHL22'] = a.iloc[0,0]-a.iloc[0,1]

for i in range(1, len(a.index)):
    ### EMA ###
    df.loc[i,'EMA10'] = EMA(a.iloc[:,5].values,10,i,df.loc[i-1,'EMA10'])
    df.loc[i,'EMA16'] = EMA(a.iloc[:,5].values,16,i,df.loc[i-1,'EMA16'])
    df.loc[i,'EMA22'] = EMA(a.iloc[:,5].values,22,i,df.loc[i-1,'EMA22'])
    df.loc[i,'EMA12'] = EMA(a.iloc[:,5].values,12,i,df.loc[i-1,'EMA12'])
    df.loc[i,'EMA18'] = EMA(a.iloc[:,5].values,18,i,df.loc[i-1,'EMA18'])
    df.loc[i,'EMA24'] = EMA(a.iloc[:,5].values,24,i,df.loc[i-1,'EMA24'])
    df.loc[i,'EMA30'] = EMA(a.iloc[:,5].values,30,i,df.loc[i-1,'EMA30'])

    
    ### SMA ###
    df.loc[i,'SMA10'] = SMA(a.iloc[:,5].values,10,i) #SMA n=10
    df.loc[i,'SMA16'] = SMA(a.iloc[:,5].values,16,i) #SMA n=16
    df.loc[i,'SMA22'] = SMA(a.iloc[:,5].values,22,i) #SMA n=22
    df.loc[i,'SMA20'] = SMA(a.iloc[:,5].values,20,i) #SMA n=20
    df.loc[i,'SMA26'] = SMA(a.iloc[:,5].values,26,i) #SMA n=26
    df.loc[i,'SMA32'] = SMA(a.iloc[:,5].values,32,i) #SMA n=32

    ### Variance ###
    df.iloc[i,7] = delta*df.iloc[i-1,7]+(1-delta)*np.power(df.iloc[i,6],2)
    
    ### Value at Risk ###
    if (df.iloc[i,6] < -df.iloc[i-1,8]):
        df.iloc[i,9] = df.iloc[i-1,9] + 1
    else:
        df.iloc[i,9] = df.iloc[i-1,9]*delta
    df.iloc[i,8] = 1.96*df.iloc[i,9]*np.sqrt(df.iloc[i,7]) 
    
    ### Bollu/Bolld ###
    if i>=20:
        df.loc[i,'Bollu20'] = df.loc[i,'SMA20']+2*a.iloc[i-20:i+1,5].std()
        df.loc[i,'Bolld20'] = df.loc[i,'SMA20']-2*a.iloc[i-20:i+1,5].std()
    else:
        df.loc[i,'Bollu20'] = df.loc[i,'SMA20']+2*a.iloc[0:i+1,5].std()
        df.loc[i,'Bolld20'] = df.loc[i,'SMA20']-2*a.iloc[0:i+1,5].std()
    if i>=26:
        df.loc[i,'Bollu26'] = df.loc[i,'SMA26']+2*a.iloc[i-26:i+1,5].std()
        df.loc[i,'Bolld26'] = df.loc[i,'SMA26']-2*a.iloc[i-26:i+1,5].std()
    else:
        df.loc[i,'Bollu26'] = df.loc[i,'SMA26']+2*a.iloc[0:i+1,5].std()
        df.loc[i,'Bolld26'] = df.loc[i,'SMA26']-2*a.iloc[0:i+1,5].std()
    if i>=32:
        df.loc[i,'Bollu32'] = df.loc[i,'SMA32']+2*a.iloc[i-32:i+1,5].std()
        df.loc[i,'Bolld32'] = df.loc[i,'SMA32']-2*a.iloc[i-32:i+1,5].std()
    else:
        df.loc[i,'Bollu32'] = df.loc[i,'SMA32']+2*a.iloc[0:i+1,5].std()
        df.loc[i,'Bolld32'] = df.loc[i,'SMA32']-2*a.iloc[0:i+1,5].std()
        
    ### ADL ###    
    df.loc[i, 'ADL'] = df.loc[i-1, 'ADL'] + a.iloc[i,4]*\
                    ((a.iloc[i,3] - a.iloc[i,1])-(a.iloc[i,0]-a.iloc[i,3]))/\
                    (a.iloc[i,0]-a.iloc[i,1])
    
    ### OBV ###
    if a.iloc[i,5]-a.iloc[i-1,5]>0:
        df.loc[i, 'OBV'] = df.loc[i-1, 'OBV'] + a.iloc[i,4]
    else:
        df.loc[i, 'OBV'] = df.loc[i-1, 'OBV'] - a.iloc[i,4]
        
    ### EMA ADL ###
    df.loc[i, 'EMADL3'] = EMA(df.ADL.values,3,i,df.loc[i-1,'EMADL3'])
    df.loc[i, 'EMADL10'] = EMA(df.ADL.values,10,i,df.loc[i-1,'EMADL10'])
    
    ### EMAHL / CHV ### 
    df.loc[i,'EMAHL10'] = EMA(a.iloc[:,0]-a.iloc[:,1], 10, i, df.loc[i-1,'EMAHL10'])
    df.loc[i, 'EMAHL16'] = EMA(a.iloc[:,0]-a.iloc[:,1], 16, i, df.loc[i-1,'EMAHL16'])
    df.loc[i, 'EMAHL22'] = EMA(a.iloc[:,0]-a.iloc[:,1], 22, i, df.loc[i-1,'EMAHL22'])
    if i>=10:
        df.loc[i, 'CHV1010'] = df.loc[i,'EMAHL10']/df.loc[i-10,'EMAHL10']
        df.loc[i, 'CHV1016'] = df.loc[i,'EMAHL16']/df.loc[i-10,'EMAHL16']
        df.loc[i, 'CHV1022'] = df.loc[i,'EMAHL22']/df.loc[i-10,'EMAHL22']
        
    ### fast % k ###
    if i >= 12 : df.loc[i,'FastK12'] = 100*(a.iloc[i,5]-a.iloc[i-12:i+1,1].min())/\
                                        (a.iloc[i-12:i+1,0].max() - a.iloc[i-12:i+1,1].min())
    if i >= 18: df.loc[i,'FastK18'] = 100*(a.iloc[i,5]-a.iloc[i-18:i+1,1].min())/\
                                        (a.iloc[i-18:i+1,0].max() - a.iloc[i-18:i+1,1].min())
    if i >= 24 : df.loc[i,'FastK24'] = 100*(a.iloc[i,5]-a.iloc[i-24:i+1,1].min())/\
                                        (a.iloc[i-24:i+1,0].max() - a.iloc[i-24:i+1,1].min())
        
    
    ### fastD ###
    if i >= 15 : df.loc[i,'FastD12'] = SMA(df.loc[:,'FastK12'], 3, i)
    if i >= 21: df.loc[i,'FastD18'] = SMA(df.loc[:,'FastK12'], 3, i)
    if i >= 27 : df.loc[i,'FastD24'] = SMA(df.loc[:,'FastK12'], 3, i)
        
    ### slow K ###
    if i >= 18 : df.loc[i,'SlowK12'] = SMA(df.loc[:,'FastD12'], 3, i)
    if i >= 24: df.loc[i,'SlowK18'] = SMA(df.loc[:,'FastD12'], 3, i)
    if i >= 30 : df.loc[i,'SlowK24'] = SMA(df.loc[:,'FastD12'], 3, i)
        
    ### Slow D ### 
    if i >= 21 : df.loc[i,'SlowD12'] = SMA(df.loc[:,'SlowK12'], 3, i)
    if i >= 27: df.loc[i,'SlowD18'] = SMA(df.loc[:,'SlowK12'], 3, i)
    if i >= 33 : df.loc[i,'SlowD24'] = SMA(df.loc[:,'SlowK12'], 3, i)
        

### CHO ###
df.loc[:,'CHO'] = df.loc[:, 'EMADL3'] - df.loc[:, 'EMADL10']
    
### Mom ###
df.loc[12:,'Mom12'] = a.iloc[12:,5].values-a.iloc[0:-12,5].values
df.loc[18:,'Mom18'] = a.iloc[18:,5].values-a.iloc[0:-18,5].values
df.loc[24:,'Mom24'] = a.iloc[24:,5].values-a.iloc[0:-24,5].values

### ACC ###
df.iloc[13:,22] = df.iloc[13:,19].values-df.iloc[12:-1,19].values # ACC 12
df.iloc[19:,23] = df.iloc[19:,20].values-df.iloc[18:-1,20].values # ACC 18
df.iloc[25:,24] = df.iloc[25:,21].values-df.iloc[24:-1,21].values # ACC 24
    
### ROC ###
df.loc[10:,'ROC10'] = 100*(a.iloc[10:,5].values-a.iloc[0:-10,5].values)/(a.iloc[0:-10,5].values)
df.loc[16:,'ROC16'] = 100*(a.iloc[16:,5].values-a.iloc[0:-16,5].values)/(a.iloc[0:-16,5].values)
df.loc[22:,'ROC22'] = 100*(a.iloc[22:,5].values-a.iloc[0:-22,5].values)/(a.iloc[0:-22,5].values)

### MACD ###
df.loc[:,'MACD1812'] = df.loc[:,'EMA18'].values - df.loc[:,'EMA12'].values
df.loc[:,'MACD2412'] = df.loc[:,'EMA24'].values - df.loc[:,'EMA12'].values
df.loc[:,'MACD3012'] = df.loc[:,'EMA30'].values - df.loc[:,'EMA12'].values


print(df)
