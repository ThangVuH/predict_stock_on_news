# sentiment score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import plotly.express as px

# create function to get the sentiment score:

def getSIA(text):
    sia = SentimentIntensityAnalyzer()
    sentiment= sia.polarity_scores(text)
    return sentiment

def sentiment_score(data):
    # get Sentiment score for each day:
    compound=[]
    neg=[]
    pos=[]
    neu=[]
    SIA=0

    for i in range (0, len(data)):
        SIA= getSIA(data[i])
        compound.append(SIA['compound'])
        neg.append(SIA['neg'])
        pos.append(SIA['pos'])
        neu.append(SIA['neu'])

    return compound, neg, pos, neu


def label_sentiment(score):
    sentiment = []
    for i in score:
        if i >= 0.05 :
            sentiment.append('Positive')
        elif i <= -0.05 :
            sentiment.append('Negative')
        else:
            sentiment.append('Neutral')
    return sentiment

def plot_dataframe():

    df_headlines = create_dataframe()

    # Sentiment analysis
    fig_1 = px.scatter(df_headlines, x = df_headlines.index , y ='compound', color = 'Sentiment' )
    plot1 = fig_1.show()

    #Pie chart
    count1 = df_headlines['Sentiment'].value_counts()
    dff = pd.DataFrame()
    dff['name']= [str(i)for i in count1.index]
    dff['number'] = count1.values
    fig_2 = px.pie(dff, values="number", names="name")
    plot2 = fig_2.show()

    return plot1, plot2
