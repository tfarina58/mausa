import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
import pandas as pd
from imblearn.metrics import geometric_mean_score
import math
# razlicite dimenzije i svi setovi podataka
# number of splits

n_splits = 10
MDSFlag = False    

if MDSFlag:
    inp = pd.read_csv('editPDE_R2_0.csv')
    labels = np.array(inp.iloc[: , -1])
    labels = labels.astype(int)
    data = pd.read_pickle('editPDE_R2_045pkl')
else:
    data = pd.read_csv('editPDE_R2_0.csv')
    labels = np.array(data.iloc[: , -1])
    labels = labels.astype(int)
    data = data.drop(data.columns[-1], axis = 1)
    
print(data.shape)
X = np.array(data) # features
y = labels         # labels
X, y = shuffle(X, y, random_state=42) # shuffle the data
print(type(X))     # ignore
print(type(y))
y_true = y.copy()  # save labels for testing
y[(int)(len(y) * 0.5):] = -1       # unlabel some(50%) of the data
total_samples = y.shape[0]  
                   # RandomForestClassifier(max_depth = 2, random_state = 42) 
base_classifier = RandomForestClassifier(max_depth = 2, random_state = 42, n_jobs = -1) # can be any algo with prediction certenty

x_values = np.arange(0.1, 1.0, 0.05)
x_values = np.append(x_values, 0.99999)
scores = np.empty((x_values.shape[0], n_splits))
amount_labeled = np.empty((x_values.shape[0], n_splits))
amount_iterations = np.empty((x_values.shape[0], n_splits))

for (i, threshold) in enumerate(x_values):  # i [0, X], threshold = 0.4, 0.45.....
    self_training_clf = SelfTrainingClassifier(base_classifier, threshold=threshold)
    test_f1 = 0
    mean_val = 0    # make a SelfTrainingClassifier 

    # We need manual cross validation so that we don't treat -1 as a separate
    # class when computing accuracy
    skfolds = StratifiedKFold(n_splits=n_splits)
    for fold, (train_index, test_index) in enumerate(skfolds.split(X, y)):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index] 
        y_test = y[test_index]
        y_test_true = y_true[test_index]

        self_training_clf.fit(X_train, y_train)
        
        # The amount of labeled samples that at the end of fitting
        amount_labeled[i, fold] = total_samples - np.unique(self_training_clf.labeled_iter_, return_counts=True)[1][0]
        # The last iteration the classifier labeled a sample in
        amount_iterations[i, fold] = np.max(self_training_clf.labeled_iter_)

        y_pred = self_training_clf.predict(X_test)
        scores[i, fold] = accuracy_score(y_test_true, y_pred)
        test_f1 += f1_score(y_test_true, y_pred, average='binary')
        mean_val += geometric_mean_score(y_test_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, correction=0.0)
    if math.isclose(threshold, 0.1) or math.isclose(threshold, 0.5) or math.isclose(threshold, 0.75): 
        print("Threshold: ", round(threshold,2))
        print("F1_score: ", test_f1/n_splits)
        print("Mean value: ", mean_val/n_splits)
                


ax1 = plt.subplot(211)
ax1.errorbar(x_values, scores.mean(axis=1), yerr=scores.std(axis=1), capsize=2, color='b')
ax1.set_ylabel('Accuracy', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.errorbar(x_values, amount_labeled.mean(axis=1), yerr=amount_labeled.std(axis=1), capsize=2, color='g')
ax2.set_ylim(bottom=0)
ax2.set_ylabel('Amount of labeled samples', color='g')
ax2.tick_params('y', colors='g')

ax3 = plt.subplot(212, sharex=ax1)
ax3.errorbar(x_values, amount_iterations.mean(axis=1), yerr=amount_iterations.std(axis=1), capsize=2, color='b')
ax3.set_ylim(bottom=0)
ax3.set_ylabel('Amount of iterations')
ax3.set_xlabel('Threshold')
plt.show()
