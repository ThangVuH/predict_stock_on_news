import streamlit as st
import numpy as np
import pandas as pd
import datetime
import plotly.graph_objects as go


import sys
sys.path.insert(1, "../../data/SP500_predict/")
import data
from data import df_main
data = df_main()


sys.path.insert(2, "../")
import model
from model import data_processing
from model import fit_model
from model import test_model
from model import forecasting



st.set_page_config(
    page_title="NLP predict"
)

st.title('Prediction of Time Series')

data = df_main()

############# Example of code ###############
# Prepocess data
p = data_processing(data)
## Fit model
m = fit_model(p)
### Test model
t = test_model(m, p)
#### Forecast the future
f = forecasting(m)
##### Plot forecast and data
def binar(df):
    df['Target'] = np.where(df['yhat'].shift(-1) > df['yhat'], 1, 0)
    return df

dataf = binar(f)

merged_df = dataf.merge(p, how='left', on='ds')

#start_date = st.date_input("Start Date", value=pd.to_datetime("2021-01-31", format="%Y-%m-%d"))
#end_date = st.date_input("End Date", value=pd.to_datetime("today", format="%Y-%m-%d"))
#start_date = datee
datum = st.date_input("Start Date", datetime.date(2020,7,6),
                      min_value= datetime.date(2019,7,6),
                      max_value=datetime.date(2022,10,2))
datee = datum.strftime("%Y-%m-%d")
st.write(datee)
start_date = datee



### as a function for date input ###

# custom function to set fill color
### as a function for date input ###

# custom function to set fill color
def graph_daterange(df, start_date):

    ### GRAPH START ###
    grph = go.Figure()

    # trim df to daterange
    date_in_index = df.loc[df['ds'] == start_date].index[0]
    date_out_index = date_in_index + 31

    df = df[(df['ds'] > start_date) & (df['ds'] < df.loc[[date_out_index]]['ds'].values[0])]

    # plot y observed
    grph.add_trace(go.Scatter(x=df['ds'],
                              y=df['y'],
                              name='Stock Market',
                              mode='lines+markers',
                              marker=dict(color='black')))

    # plot yhat
    grph.add_trace(go.Scatter(x=df['ds'],
                              y=df['yhat'],
                              name='Predictions',
                              mode='lines',
                              line=dict(color='red', width=3)
                              ))

    # plot upper CI
    grph.add_trace(go.Scatter(x=df['ds'],
                              y=df['yhat_upper'],
                              name='yhat_upper',
                              mode='lines',
                              line=dict(color='blue', width=3)
                              ))

    # plot lower CI
    grph.add_trace(go.Scatter(x=df['ds'],
                              y=df['yhat_lower'],
                              name='yhat_lower',
                              mode='lines',
                              fill='tonexty',
                              line=dict(color='blue', width=3)
                              ))

    grph.update_layout(title="Stock Market",
                      template= "plotly_white")

    return grph

figu = graph_daterange(merged_df, start_date)

st.plotly_chart(figu, use_container_width=True)

# #st.table(f)
# trend = f.loc[f.ds == start_date]["Target"]
# st.write(f'Price: {trend}')

# fig = plot_forecast(m, f)
# fig.update_layout(plot_bgcolor="rgba(255, 255, 255, 0.8)", paper_bgcolor="rgba(255, 255, 255, 0.8)")
# st.plotly_chart(fig, use_container_width=True)
