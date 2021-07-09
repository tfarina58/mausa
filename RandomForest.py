import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import f1_score
import math
from sklearn.manifold import MDS

SEED = 42
n_splits = 10

MDSFlag = True    

if MDSFlag:
    inp = pd.read_csv('KC1.data')
    labels = np.array(inp.iloc[: , -1])
    labels = labels.astype(int)
    data = pd.read_pickle('KC1dataMDS5pkl')
else:
    data = pd.read_csv('KC1.data')
    labels = np.array(data.iloc[: , -1])
    labels = labels.astype(int)
    data = data.drop(data.columns[-1], axis = 1)

from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

X = np.array(data) # features
y = labels         # labels
X, y = shuffle(X, y, random_state=42) # 2, 5, 10, 25, 50   | KC1 | JDT_R3_2 | PDE_R2_0

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

labeled = [0.02, 0.05, 0.1, 0.25, 0.5]

x_values = np.array([0.1, 0.5, 0.75])

for (i, threshold) in enumerate(x_values):
    
    for labeledammount in labeled:
        test_f1 = 0
        geo_score = 0 

        X_cpy = np.copy(X[:(int)(len(X) * labeledammount)])
        y_cpy = np.copy(y[:(int)(len(y) * labeledammount)])

        skfolds = StratifiedKFold(n_splits=n_splits)
        for index, (train_index, test_index) in enumerate(skfolds.split(X_cpy, y_cpy)):
            X_train = X_cpy[train_index]
            y_train = y_cpy[train_index]
            X_test = X_cpy[test_index] 
            y_test = y_cpy[test_index]
            classifier = RandomForestClassifier(max_depth=2, random_state=SEED, n_estimators = 15)
            classifier.fit(X_train, y_train)

            y_pred = classifier.predict_proba(X_test)
            predicted = (y_pred [:,1] >= threshold).astype('int')

            test_f1 += f1_score(y_test, predicted, average='binary', zero_division = 0)
            geo_score += geometric_mean_score(y_test, predicted, labels=None, pos_label=1, average='binary', sample_weight=None, correction=0.0)

            if (math.isclose(threshold, 0.1) or math.isclose(threshold, 0.5) or math.isclose(threshold, 0.75)) and index == 9: 
                print("LABELED AMMOUNT = ", labeledammount)
                print("Threshold: ", round(threshold,2))
                print("F1_score: ", round(test_f1/n_splits, 4))
                print("Geo score: ", round(geo_score/n_splits, 4))

# Predicting the Test set results