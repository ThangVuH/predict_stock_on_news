from tqdm.auto import tqdm


accuracys = []

for name,clf in tqdm(clfs.items()):
    curr_acc = fit_model(clf,x_train,y_train,x_test,y_test)
    accuracys.append(curr_acc)
