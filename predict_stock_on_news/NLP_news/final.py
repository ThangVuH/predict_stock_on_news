import string
import pandas as pd
import numpy as np
import re
# nltk
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer,PorterStemmer
# sentiment score
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# sentiment analysis
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader
# preprocess
from sklearn.preprocessing import LabelEncoder
# split dataset
from sklearn.model_selection import train_test_split
# evaluate model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_score
from sklearn.metrics import r2_score
# ML model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# ML regressor
# from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
# from sklearn.svm import SVR
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.neural_network import MLPRegressor
# from sklearn.naive_bayes import MultinomialNB

from xgboost import XGBRegressor
#from lightgbm import LGBMRegressor
# ML classifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.naive_bayes import MultinomialNB

from xgboost import XGBClassifier
#from lightgbm import LGBMClassifier
# DL model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import matthews_corrcoef
import plotly.graph_objects as go

# plot
from collections import  Counter
import plotly.express as px
from transformers import pipeline

#--------------
# 1. Clean
def basic_cleaning(sentence):
    sentence = sentence.lower()
    sentence = ''.join(char for char in sentence if not char.isdigit())

    weird_char = string.punctuation + '’' + '‘' + '—'
    for punctuation in weird_char:
        sentence = sentence.replace(punctuation, '')

    sentence = sentence.strip()
    STOPWORDS = stopwords.words('english') + ['th']
    sentence = ' '.join([word for word in sentence.split() if word.lower() not in STOPWORDS])

    return sentence
def tokenizing_text(text):
  new= text.str.split()
  new=new.values.tolist()
  return new
def stem_and_lem(text):
  corpus=[]
  stem=PorterStemmer()
  lem=WordNetLemmatizer()
  for i in text:
    words=[lem.lemmatize(w) for w in i if len(w)>2]
    words=[stem.stem(w) for w in i if len(w)>2]
    corpus.append(words)
  return corpus

# 2. Sentiment
# get subjectivity:

def get_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity

# get polarity:

def get_polarity(text):
    return TextBlob(text).sentiment.polarity

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

# 3. Dataframe
def create_dataframe():

    # Retrieve raw data
    #data_raw_path = os.path.join(LOCAL_DATA_PATH, "raw", f"train_{DATASET_SIZE}.csv")
    #data = pd.read_csv(data_raw_path, dtype=DTYPES_RAW_OPTIMIZED)
    data = pd.read_csv("dataset_NYTimes.csv")

    # Clean data using ml_logic.data.clean_data
    # $CODE_BEGIN
    data = data.dropna()
    data['News'] = data['News'].apply(basic_cleaning)
    data.set_index('Date', inplace = True)

    data_cleaned = data
    # $CODE_END

    #Subjectivity and polarity
    data_cleaned['Subjectivity']=data_cleaned['News'].apply(get_subjectivity)
    data_cleaned['Polarity']=data_cleaned['News'].apply(get_polarity)

    # Sentiment analysis
    #1. Sentiment score
    compound, neg, pos, neu = sentiment_score(data_cleaned['News'])
    data_cleaned['compound']= compound
    data_cleaned['neg']= neg
    data_cleaned['pos']=pos
    data_cleaned['neu']=neu

    #2. Sentiment label
    score =(data_cleaned["compound"].values)
    data_cleaned["Sentiment"]=label_sentiment(score)

    return data_cleaned

def plot_common_word(text):
  new =tokenizing_text(text)
  corpus=[word for i in new for word in i]
  counter=Counter(corpus)
  most=counter.most_common()
  x, y=[], []
  for word,count in most[:40]:
    x.append(word)
    y.append(count)

  fig =px.bar(text,x=y,y=x)
  return fig.show()

def plot_sentimentByTime(data):
  fig_1 = px.scatter(data, x = data.index , y ='compound', color = 'Sentiment' )
  return fig_1.show()

def plot_percentageSentimentWord(data):
  count1 = np.round(data["Sentiment"].value_counts()/len(data)*100)
  dff = pd.DataFrame()
  dff['name']= [str(i)for i in count1.index]
  dff['number'] = count1.values
  fig_6 = px.pie(dff, values="number", names="name", color='name',
                color_discrete_map={'Negative':'navy',
                                  'Positive':'green',
                                  'Neutral':'red',
                                  })
  fig_6.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                      marker=dict(line=dict(color='#faf7f7', width=2)))
  return fig_6.show()

def split_data():
    df_headlines = create_dataframe()
    X=df_headlines[['Subjectivity','Polarity','compound','neg','pos','neu']]
    X=np.array(X)
    y= np.array(df_headlines['Label'])
    x_train, x_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state= 0)
    return x_train, x_test, y_train, y_test
def data_sentiment(data):
  a = []
  for i in data['News']:
    a.append(i)
  return a

def predict_sentiment(data):
  sentiment_pipeline = pipeline("sentiment-analysis")
  data = data_sentiment(data)
  y_pred=sentiment_pipeline(data)
  return y_pred

# 4. Model
xgb = XGBClassifier(eval_metric="mlogloss",random_state=42)

clfs = {
    "XGBoost": xgb,
}

x_train,y_train,x_test, y_test = split_data()

def fit_model(clf,x_train,y_train):
    clf.fit(x_train,y_train)
    return clf

def predict_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    return y_pred

def accuracy(y_pred):
    return accuracy_score(y_pred, y_test)

def MCC(y_pred):
    return matthews_corrcoef(y_pred, y_test)
