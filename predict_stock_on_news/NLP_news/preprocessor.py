import string
import pandas as pd
import numpy as np
import re
from collections import  Counter

# nltk
import nltk
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords


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
