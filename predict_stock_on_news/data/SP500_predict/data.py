# imports
import pandas as pd
from pandas_datareader import data as pdr
from datetime import datetime
import numpy as np

import yfinance as yf

# override yahoo finance with pandas_datareader
def override_yf_pdr():
    return yf.pdr_override()


# create date column for merge
def create_date_col():
    df_date = pd.DataFrame(pd.date_range(start="2008-01-01", end=datetime.today().strftime('%Y-%m-%d')))
    df_date.columns = ['Date']
    return df_date


# interpolate function
def interpolate_function(df):
    for i in df:
        df[i].iloc[0] = df[i].loc[df[i].first_valid_index()]
        df[i].iloc[-1] = df[i].loc[df[i].last_valid_index()]

        df[i] = df[i].interpolate(method='from_derivatives')

    return df


# import SP500 as dataframe
def data_sp():
    override_yf_pdr()

    data_sp = pdr.get_data_yahoo("^GSPC", start="2008-01-01", end=datetime.today().strftime('%Y-%m-%d'), utc=True).reset_index()
    df_date = create_date_col()
    data_sp.merge(df_date, on='Date', how='right')
    data_sp.set_index('Date', inplace = True)

    data_sp = interpolate_function(data_sp)

    return data_sp


### Vu's stolen code ###

#Create functions to calculate the SMA, & the EMA
#Create the Simple Moving Average Indicator
#Typical time periods for moving averages are 15, 20,& 30
#Create the Simple Moving Average Indicator
def SMA(data, period=30, column='Close'):
    return data[column].rolling(window=period).mean()

#Create the Exponential Moving Average Indicator
def EMA(data, period=20, column='Close'):
    return data[column].ewm(span=period, adjust=False).mean()


def MACD(data, period_long=26, period_short=12, period_signal=9, column='Close'):
    #Calculate the Short Term Exponential Moving Average
    ShortEMA = EMA(data, period_short, column=column) #AKA Fast moving average
    #Calculate the Long Term Exponential Moving Average
    LongEMA = EMA(data, period_long, column=column) #AKA Slow moving average
    #Calculate the Moving Average Convergence/Divergence (MACD)
    data['MACD'] = ShortEMA - LongEMA
    #Calcualte the signal line
    data['Signal_Line'] = EMA(data, period_signal, column='MACD')#data['MACD'].ewm(span=period_signal, adjust=False).mean()

    return data


def RSI(data, period = 14, column = 'Close'):
    delta = data[column].diff(1) #Use diff() function to find the discrete difference over the column axis with period value equal to 1
    delta = delta.dropna() # or delta[1:]
    up =  delta.copy() #Make a copy of this object's indices and data
    down = delta.copy() #Make a copy of this object's indices and data
    up[up < 0] = 0
    down[down > 0] = 0
    data['up'] = up
    data['down'] = down
    AVG_Gain = SMA(data, period, column='up')#up.rolling(window=period).mean()
    AVG_Loss = abs(SMA(data, period, column='down'))#abs(down.rolling(window=period).mean())
    RS = AVG_Gain / AVG_Loss
    RSI = 100.0 - (100.0/ (1.0 + RS))

    data['RSI'] = RSI
    return data

###############################################

# create target column
def target(df):
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    return df


# call final df
def df_main():
    df_main = data_sp()
    MACD(df_main)
    RSI(df_main)
    df_main['SMA'] = SMA(df_main)
    df_main['EMA'] = EMA(df_main)

    df_main = interpolate_function(df_main)

    target(df_main)

    return df_main




### OLD CRAP ###

# #Create the target column
# df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0) # if tomorrows price is greater than todays price put 1 else put 0

# # smooth the curve
# for i in df.columns:
#   df[i].iloc[0] = df[i].loc[df[i].first_valid_index()]
#   df[i].iloc[-1] = df[i].loc[df[i].last_valid_index()]

#   df[i] = df[i].interpolate(method='from_derivatives')
