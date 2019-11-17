import sklearn.datasets
import sklearn.neighbors
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.model_selection

vino=sklearn.datasets.load_wine()
train_inputs=vino.data
train_outputs=vino.target
min_max_scaler = sklearn.preprocessing.MinMaxScaler()
train_inputs = min_max_scaler.fit_transform(train_inputs)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(train_inputs, train_outputs, test_size=0.4)

clasif=sklearn.neighbors.KNeighborsClassifier(5).fit(X_train,y_train)
print(clasif.score(X_test,y_test))

clasif2=sklearn.linear_model.LogisticRegression(multi_class='auto',solver='liblinear').fit(X_train,y_train)
print(clasif2.score(X_test,y_test))
