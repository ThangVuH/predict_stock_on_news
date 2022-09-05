#Create functions to calculate the SMA, & the EMA
#Create the Simple Moving Average Indicator
#Typical time periods for moving averages are 15, 20,& 30
#Create the Simple Moving Average Indicator
def SMA(data, period=30, column='Close'):
  return data[column].rolling(window=period).mean()
#Create the Exponential Moving Average Indicator
def EMA(data, period=20, column='Close'):
  return data[column].ewm(span=period, adjust=False).mean()

#Create a function to calculate the Moving Average Convergence/Divergence (MACD)
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

#Create a function to calculate the Relative Strength Index (RSI)
def RSI(data, period = 14, column = 'Close'):
  delta = data[column].diff(1) #Use diff() function to find the discrete difference over the column axis with period value equal to 1
  delta = delta.dropna() # or delta[1:]
  up =  delta.copy() #Make a copy of this object’s indices and data
  down = delta.copy() #Make a copy of this object’s indices and data
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
