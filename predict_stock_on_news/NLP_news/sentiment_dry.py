from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



def sentiment(df, journal):
  list_names = ['compound', 'neg', 'pos', 'neu', 'Subjectivity_', 'Polarity_']
  list_a, list_b, list_c, list_d, list_e, list_f = ([] for i in range(6))

  sia = SentimentIntensityAnalyzer()

  for i in range (len(df[journal])):
    SIA = sia.polarity_scores(df[journal][i])
    list_a.append(SIA['compound'])
    list_b.append(SIA['neg'])
    list_c.append(SIA['pos'])
    list_d.append(SIA['neu'])

  for i in range(len(df[journal])):
    list_e.append(TextBlob(df[journal][i]).sentiment.subjectivity)
    list_f.append(TextBlob(df[journal][i]).sentiment.polarity)

  list_sum = [list_a, list_b, list_c, list_d, list_e, list_f]

  for i, j in zip(list_names, list_sum):
    df[str(i) + journal] = j
  locals().update(df)

  return df[str(i) + journal]
