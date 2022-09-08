# ML classifier
import pandas as pd
from sklearn.metrics import matthews_corrcoef, accuracy_score
from xgboost import XGBClassifier
from NLP_news.dataframe import split_data

# this model is using NLP sentiment for predict stock go up or down
#----------------

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
