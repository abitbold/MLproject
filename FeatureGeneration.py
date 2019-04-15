import numpy as np
import pandas as pd
import pandas_datareader as web

a = web.get_data_yahoo('ADS', start='2019-01-01', end='today')


df = pd.DataFrame(columns = ['EMA10','EMA16', 'EMA22', 'SMA10', 'SMA16', 'SMA22', 'Return', 'Variance', 'ValueAtRisk', 'VarScalar',
                            'SMA20', 'SMA26', 'SMA32', 'Bollu20','Bollu26', 'Bollu32', 'Bolld20', 'Bolld26', 'Bolld32', 'Mom12', 
                             'Mom18', 'Mom24'], 
                  index = range(len(a.index)))

df['Date'] = a.index
### parameters ###

delta = 0.94

### EMA ###
df.iloc[0,0] = a.iloc[0,5]
df.iloc[0,1] = a.iloc[0,5]
df.iloc[0,2] = a.iloc[0,5]

### SMA ###
df.iloc[0,3] = a.iloc[0,5]
df.iloc[0,4] = a.iloc[0,5]
df.iloc[0,5] = a.iloc[0,5]
df.iloc[0,10] = a.iloc[0,5]
df.iloc[0,11] = a.iloc[0,5]
df.iloc[0,12] = a.iloc[0,5]

### Returns ###
df.iloc[1:,6] = (a.iloc[1:,5].values-a.iloc[0:-1,5].values)/a.iloc[0:-1,5].values

### Variance ###

df.iloc[0,7] = np.power(df.iloc[1:32,6].std(),2)

### Value at Risk ###

df.iloc[0,8] = 1.96*np.sqrt(df.iloc[0,7])
df.iloc[0,9] = 1

for i in range(1, len(a.index)):
    ### EMA ###
    df.iloc[i, 0] = a.iloc[i,5]*(2/11)+df.iloc[i-1,0]*(1-2/11) # EMA n=10
    df.iloc[i, 1] = a.iloc[i,5]*(2/17)+df.iloc[i-1,1]*(1-2/17) # EMA n=16
    df.iloc[i, 2] = a.iloc[i,5]*(2/23)+df.iloc[i-1,2]*(1-2/23) # EMA n=22
    
    ### SMA ###
    if i >= 10:                                                # SMA n=10
        df.iloc[i, 3] = a.iloc[i-10:i+1,5].mean()
    else:
        df.iloc[i, 3] = a.iloc[0:i+1,5].mean()
    if i >= 16:                                                # SMA n=16  
        df.iloc[i, 4] = a.iloc[i-16:i+1,5].mean()
    else:
        df.iloc[i, 4] = a.iloc[0:i+1,5].mean()
    if i >= 22:                                                # SMA n=22
        df.iloc[i, 5] = a.iloc[i-22:i+1,5].mean()
    else:
        df.iloc[i, 5] = a.iloc[0:i+1,5].mean()
    if i >= 20:                                                # SMA n=20
        df.iloc[i, 10] = a.iloc[i-20:i+1,5].mean()
    else:
        df.iloc[i, 10] = a.iloc[0:i+1,5].mean()
    if i >= 26:                                                # SMA n=26  
        df.iloc[i, 11] = a.iloc[i-26:i+1,5].mean()
    else:
        df.iloc[i, 11] = a.iloc[0:i+1,5].mean()
    if i >= 32:                                                # SMA n=32
        df.iloc[i, 12] = a.iloc[i-32:i+1,5].mean()
    else:
        df.iloc[i, 12] = a.iloc[0:i+1,5].mean()

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
    

print(df)
