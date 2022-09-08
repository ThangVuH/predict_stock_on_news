import streamlit as st
import matplotlib.pyplot as plt
#Header
st.header("""
         # Predicting stock-market movement with newspaper headlines
         # """)

#Sidebar for choosing Date
col1 = st.sidebar
date = col1.date_input('Please choose a Date:')

#Expandable About bar
expander_bar = st.expander("About")
expander_bar.markdown("""
This is a LeWagon Alumni **Data** **Science** Project about predicting if the
stock market will go up or down from analyzing news headlines.\n
Made by [Laura](https://www.linkedin.com/in/laura-martel-133692159/),
[Vu](https://www.linkedin.com/in/thang-vu-hong/) and
[Philip](https://www.linkedin.com/in/philip-steffen-71b5b823a/).
""")

st.subheader(f'New York Times Headline from {date}')


st.text("""

        """)

col2, col3 = st.columns((1,1))
col2.text('Placeholder text:')

col3.text('Placeholder text')
