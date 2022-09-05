#Split the data set into a feature or independent data set (X) and a target or dependent data set (Y)
keep_columns = ['Close', 'MACD', 'Signal_Line', 'RSI', 'SMA', 'EMA']
#keep_columns = ['Close', 'MACD', 'Signal_Line', 'RSI', 'EMA']
X = df[keep_columns].values
Y = df['Target'].values

#Split the data again but this time into 80% training and 20% testing data sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)


#Create and train the model
tree = DecisionTreeClassifier().fit(X_train, Y_train)

#Check how well the SVC Model on training data
print(tree.score(X_train, Y_train))

#Check the SVC Model on the test data set
print(tree.score(X_test, Y_test))
