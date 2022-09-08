import streamlit as st
import numpy as np


st.set_page_config(
    page_title="NLP predict"
)

import sys
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
from model import plot_forecast

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


st.plotly_chart(plot_forecast(m, f))
