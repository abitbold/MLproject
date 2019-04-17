import numpy as np
import pandas as pd
import pandas_datareader as web
from datetime import date
import holidays as hl
import datetime as dt
import FeatureGeneration as fg

#### Code to retrieve holiday outside of week ends ####
def get_market_holidays(start='2000-01-01', end='2019-01-01'):
    temp = web.get_data_yahoo('^GDAXI', start=start, end=end)
    opens=set(temp.index.values)
    opens = pd.to_datetime(list(opens))
    weekmask = [1, 1, 1, 1, 1, 0, 0]
    busdays = np.arange(start, end, dtype='datetime64[D]')
    busdays = pd.to_datetime(busdays[np.is_busday(busdays, weekmask=weekmask)])
    holidays = set(busdays).difference(opens)
    df = pd.to_datetime(list(holidays)).sort_values()
    return df


#### All of the week ends
def get_fridays(start='2000-01-01', end='2019-01-01'):
    weekmask = [0, 0, 0, 0, 1, 0, 0]
    fridays = np.arange(start, end, dtype='datetime64[D]')
    fridays = pd.to_datetime(fridays[np.is_busday(fridays, weekmask=weekmask)]).sort_values()
    return fridays

### Begining of each month
def get_first_days(start='2000-01-01', end='2019-01-01'):
    days = pd.date_range(start, end, freq='MS')
    return days

### Get random tickers
def get_random_tickers(n, ticklist):
    if n >len(ticklist):
        raise Exception('n is bigger than ticklist')
    return np.random.choice(ticklist, size=n, replace=False)
