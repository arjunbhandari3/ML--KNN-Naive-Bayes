import pandas as pd
iris_df = pd.read_csv('iris.csv')
iris_df.head(5)

print (iris_df.isnull().sum())
print (iris_df.info())

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
iris_df['Class'] = LE.fit_transform(iris_df['Class'])
iris_df.head(5)

X = iris_df.drop('Class', axis = 1)
y = iris_df['Class']

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

test_size = 0.30 # taking 70:30 training and test set
seed = 7  # Random numbmer seeding for reapeatability of the code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

NN = KNeighborsClassifier()
NN.fit(X_train,y_train)
y_pred = NN.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)

metrics.confusion_matrix(y_train, NN.predict(X_train))
metrics.confusion_matrix(y_test, y_pred)

print(("Test accuracy: ", NN.score(X_test, y_test)))
print(("Train accuracy: ",NN.score(X_train, y_train)))

## Cross Validation to find optimal K
from sklearn.model_selection import cross_val_score
import numpy as np
print(X_train.shape[0])
print (int(np.sqrt(X_train.shape[0])))
maxK = int(np.sqrt(X_train.shape[0]))
print(maxK)

# creating odd list of K for KNN
myList = list(range(1,15))
# subsetting just the odd ones
neighbors = list(filter(lambda x: x % 2 != 0, myList))


# empty list that will hold cv scores
cv_scores = []
# perform 10-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# changing to misclassification error
misError = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[misError.index(min(misError))]
print("The optimal number of neighbors is %d" % optimal_k)