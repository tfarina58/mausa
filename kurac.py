# evaluate label propagation on the semi-supervised learning dataset
from numpy import concatenate
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.semi_supervised import LabelPropagation
import pandas as pd
import numpy as np

SEED = 42
print("new line ---------------------------------\n")
data = pd.read_csv('editPDE_R3_1.csv')
labels = np.array(data['bug_cnt'])
labels = labels.astype(int)
#features = pd.read_pickle("save.pkl")
features = data.drop(['bug_cnt'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.50, random_state = SEED)
print("new line ---------------------------------\n")
# split train into labeled and unlabeled
X_train_lab, X_test_unlab, y_train_lab, y_test_unlab = train_test_split(X_train, y_train, test_size=0.50, random_state=1, stratify=y_train)
print("new line ---------------------------------\n")
# create the training dataset input
X_train_mixed = concatenate((X_train_lab, X_test_unlab))
print("new line ---------------------------------\n")
# create "no label" for unlabeled data
nolabel = [-1 for _ in range(len(y_test_unlab))]
print("new line ---------------------------------\n")
# recombine training dataset labels
y_train_mixed = concatenate((y_train_lab, nolabel))
print("new line ---------------------------------\n")
# define model
model = LabelPropagation()
print("new line ---------------------------------\n")
# fit model on training dataset
print(X_train_mixed)
print("new data______________________")
print(y_train_mixed)
model.fit(X_train_mixed, y_train_mixed)
print("new line ---------------------------------\n")
# make predictions on hold out test set
yhat = model.predict(X_test)
print("new line ---------------------------------\n")
# calculate score for test set
score = accuracy_score(y_test, yhat)
print("new line ---------------------------------\n")
# summarize score
print('Accuracy: %.3f' % (score*100))