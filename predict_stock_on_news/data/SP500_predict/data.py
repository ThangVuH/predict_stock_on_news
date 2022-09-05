#Add the indicators to the data set
#Creating the data set
MACD(df)
RSI(df)
df['SMA'] = SMA(df)
df['EMA'] = EMA(df)

#Create the target column
df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0) # if tomorrows price is greater than todays price put 1 else put 0

# smooth the curve
for i in df.columns:
  df[i].iloc[0] = df[i].loc[df[i].first_valid_index()]
  df[i].iloc[-1] = df[i].loc[df[i].last_valid_index()]

  df[i] = df[i].interpolate(method='from_derivatives')
