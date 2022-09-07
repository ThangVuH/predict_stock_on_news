from NLP_news.model import fit_model

# evaluate model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_score
from sklearn.metrics import r2_score
from tqdm.auto import tqdm


accuracys = []

def evaluate_model():
    for name,clf in tqdm(clfs.items()):
        #curr_acc = fit_model(clf,x_train,y_train,x_test,y_test)
        y_pred = fit_model(clf,x_train,y_train,x_test,y_test)
        curr_acc = accuracy_score(y_pred, y_test)
        accuracys.append(curr_acc)
    return accuracys
