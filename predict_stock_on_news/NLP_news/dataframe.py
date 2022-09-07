from NLP_news.preprocessor import basic_cleaning
from NLP_news.polar_subj import get_subjectivity, get_polarity
from NLP_news.sentiment import sentiment_score, label_sentiment
import pandas as pd
import plotly.express as px
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
stopwords = set(STOPWORDS)

# load dataset
def data_kaggle():
    #df_NYT = pd.read_csv("dataset_NYTimes.csv")
    df_NYT= df_NYT.dropna()
    return

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


def plot_dataframe():
    df_headlines = create_dataframe()
    def plot_dataframe1():
        fig_1 = px.scatter(df_headlines, x = create_dataframe.index , y ='compound', color = 'Sentiment' )
        return fig_1.show()
    def plot_dataframe2():
        fig_2 =px.scatter(df_headlines, x = 'Polarity' , y ='Subjectivity', color = 'compound')
        return fig_2.show()
    def plot_dataframe3():
        fig_3, ax = plt.subplots(figsize=(8, 8))

        counts = df_headlines['Sentiment'].value_counts(normalize=True) * 100

        sns.barplot(x=counts.index, y=counts, ax=ax)

        ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
        ax.set_ylabel("Percentage")

        plt.show()
        return fig_3.show()
    def plot_dataframe4():
        fig_4 =px.scatter(df_headlines, x = df_headlines.index , y ='compound', color = 'Label', color_continuous_scale='portland')
        return fig_4.show()

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40,
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()
