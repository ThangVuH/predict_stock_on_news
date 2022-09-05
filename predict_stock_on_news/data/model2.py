#!pip install -q transformers
from transformers import pipeline

# Using a specific model for sentiment analysis
specific_model = pipeline(model="shashanksrinath/News_Sentiment_Analysis")
