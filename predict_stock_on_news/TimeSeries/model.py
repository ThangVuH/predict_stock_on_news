# Time series with FBprophet

### Packages and libraries to install
##General libraries

import pandas as pd
import plotly.express as px
from statsmodels.tools.eval_measures import rmse
import numpy as np
import plotly.graph_objects as go

# Facebook specific
# pip install pystan~=2.14  #This specific version of pystan must be installed to have fbprophet
# pip install fbprophet


from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics


#### Import data from a function in another .py file (optional)

import sys
sys.path.insert(1, "../data/SP500_predict/")
import data
from data import df_main
data = df_main()


## Choose your data by installing yahoo finance for example
# Enter a dataframe with columns 'ds' for date in format yyyy-mm-dd and 'y' for the data
# Tip, it is better to run the model on a 2 to 4 years time


def data_processing(self):
    self = self.reset_index()
    self = self.loc[self['Date'] >= '2018-01-01']
    df = self[['Date', 'Close']]
    df.columns = ['ds', 'y']
    return df



def fit_model(self):
    """### Creating and Training the model
    You fit the model ***on all the dataset *** it is like this in FBprophet
    """
    fbp = Prophet() # model named fbp
    fbp.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model = fbp.fit(self)
    return model

# Test your model
def test_model(self, data):
    test = data[round(len(data)*0.8):]
    forecast = self.predict(test)
    predictions = forecast.iloc[-len(test):]['yhat']
    actuals = test['y']
    return f"RMSE: {round(rmse(predictions, actuals))}"

# Predict the future
def forecasting(self):
  future = self.make_future_dataframe(periods=60)
  return self.predict(future)

# Plot your graph (model and prediction)
def plot_forecast(model, prediction):
  return plot_plotly(model, prediction)

# custom function to set fill color






############# Example of code ###############
## Prepocess data
# p = data_processing(data)
### Fit model
# m = fit_model(p)
#### Test model
# t = test_model(m, p)
##### Forecast the future
# f = forecasting(m)
###### Plot forecast and data
# plot_forecast(m, f)
