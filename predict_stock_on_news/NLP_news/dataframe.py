## Preprocessing
### Library to import

from NLP_news.preprocessor import basic_cleaning, tokenizing_text
from NLP_news.polar_subj import get_subjectivity, get_polarity
from NLP_news.sentiment import sentiment_score, label_sentiment
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
from collections import  Counter
import matplotlib.pyplot as plt
# split dataset
from sklearn.model_selection import train_test_split
stopwords = set(STOPWORDS)

# import sys
# sys.path.insert(1, "data/SP500_predict/")
# import data
# from data import df_main
# data = df_main()


def create_dataframe():
    # Retrieve raw data
    #data_raw_path = os.path.join(LOCAL_DATA_PATH, "raw", f"train_{DATASET_SIZE}.csv")
    #data = pd.read_csv(data_raw_path, dtype=DTYPES_RAW_OPTIMIZED)
    data = pd.read_csv("dataset_NYTimes.csv")

    # Clean data using ml_logic.data.clean_data
    # $CODE_BEGIN
    data = data.dropna()
    data_cleaned = data['News'].apply(basic_cleaning)
    data_cleaned = data_cleaned.reset_index()
    # $CODE_END

    #Subjectivity and polarity
    data_cleaned['Subjectivity']=data_cleaned.apply(get_subjectivity)
    data_cleaned['Polarity']=data_cleaned.apply(get_polarity)

    # Sentiment analysis
    #1. Sentiment score
    compound, neg, pos, neu = sentiment_score(data_cleaned)
    data_cleaned['compound']= compound
    data_cleaned['neg']= neg
    data_cleaned['pos']=pos
    data_cleaned['neu']=neu

    #2. Sentiment label
    score =(data_cleaned["compound"].values)
    data_cleaned["Sentiment"]=label_sentiment(score)

    return data_cleaned

def split_data():
    df_headlines = create_dataframe()
    X=df_headlines[['Subjectivity','Polarity','compound','neg','pos','neu']]
    X=np.array(X)
    y= np.array(df_headlines['Label'])
    x_train, x_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state= 0)
    return x_train, x_test, y_train, y_test

# def plot_dataframe():
#     df_headlines = create_dataframe()

#     # Polarity vs subjectivity
#     fig_2 =px.scatter(df_headlines, x = 'Polarity' , y ='Subjectivity', color = 'compound')
#     plot2 = fig_2.show()

#     # Stock vs sentiment
#     fig_4 =px.scatter(df_headlines, x = df_headlines.index , y ='compound', color = 'Label', color_continuous_scale='portland')
#     plot4 = fig_4.show()
#     return  plot2, plot4


# # no need this show_wordcloud() function
# def show_wordcloud(data, title = None):
#     wordcloud = WordCloud(
#         background_color='white',
#         stopwords=stopwords,
#         max_words=200,
#         max_font_size=40,
#         scale=3,
#         random_state=1 # chosen at random by flipping a coin; it was heads
#     ).generate(str(data))

#     fig = plt.figure(1, figsize=(12, 12))
#     plt.axis('off')
#     if title:
#         fig.suptitle(title, fontsize=20)
#         fig.subplots_adjust(top=2.3)

#     plt.imshow(wordcloud)
#     return plt.show()

# def plot_common_word(text):
#   new =tokenizing_text(text)
#   corpus=[word for i in new for word in i]
#   counter=Counter(corpus)
#   most=counter.most_common()
#   x, y=[], []
#   for word,count in most[:40]:
#     x.append(word)
#     y.append(count)

#   fig =px.bar(text,x=y,y=x)
#   return fig.show()
