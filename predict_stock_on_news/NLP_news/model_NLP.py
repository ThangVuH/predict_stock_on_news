#!pip install -q transformers


# this model is using for predict the Sentiment of sentence
# ----------

from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis")
data = ["I love you", "I hate you"]
sentiment_pipeline(data)
