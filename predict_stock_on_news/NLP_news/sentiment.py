# sentiment score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


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
