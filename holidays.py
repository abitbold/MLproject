import pandas as pd
import numpy as np
import pandas_datareader as web

def get_market_holidays(start='2000-01-01', end='2019-01-01'):
    temp = web.get_data_yahoo('^GDAXI', start='2000-01-01', end='2019-01-01')
    opens=set(temp.index.values)
    opens = pd.to_datetime(list(opens))
    weekmask = [1, 1, 1, 1, 1, 0, 0]
    busdays = np.arange('2000-01-01', '2019-01-01', dtype='datetime64[D]')
    busdays = pd.to_datetime(busdays[np.is_busday(busdays, weekmask=weekmask)])
    holidays = set(busdays).difference(opens)
    df = pd.DataFrame(pd.to_datetime(list(holidays)).sort_values())
    return df
