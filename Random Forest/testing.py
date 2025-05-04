from sklearn import datasets 
from sklearn.model_selection import train_test_split 
import numpy as np 
from random_forest import RandomForest 

data = datasets.load_breast_cancer()
X = data.data 
y = data.target 

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

def accuracy(y_test,y_pred): 

    accuracy = np.sum(y_test == y_pred) / len(y_test)
    return accuracy

clf = RandomForest(n_trees=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy(y_test, y_pred)
print(acc)