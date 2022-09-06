from NLP_news.preprocessor import basic_cleaning
from NLP_news.polar_subj import get_subjectivity, get_polarity
from NLP_news.sentiment import sentiment_score, label_sentiment
import pandas as pd
import os

# load dataset
def data_kaggle():
    #df_NYT = pd.read_csv("dataset_NYTimes.csv")
    df_NYT= df_NYT.dropna()
    return

def create_dataframe():

    # Retrieve raw data
    data_raw_path = os.path.join(LOCAL_DATA_PATH, "raw", f"train_{DATASET_SIZE}.csv")
    data = pd.read_csv(data_raw_path, dtype=DTYPES_RAW_OPTIMIZED)

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
