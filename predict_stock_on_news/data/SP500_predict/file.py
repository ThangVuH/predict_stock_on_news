import yfinance as yf
import plotly.express as px
from pandas_datareader import data as pdr
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# load data
yf.pdr_override()
df = pdr.get_data_yahoo("^GSPC", start="2008-01-01", end="2021-12-31", utc=True).reset_index()
df.set_index('Date', inplace = True)
