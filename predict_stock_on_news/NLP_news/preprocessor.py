import string
import pandas as pd
import numpy as np
import re

# nltk
import nltk
nltk.download('stopwords')
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
