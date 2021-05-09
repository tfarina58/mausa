import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle

import pomegranate as pg
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_breast_cancer

import warnings

def label_prop_test(kernel, params_list, X_train, X_test, y_train, y_test):
    plt.figure(figsize=(20,10))
    n, g = 0, 0
    roc_scores = []
    if kernel == 'rbf':
        for g in params_list:
            lp = LabelPropagation(kernel=kernel, n_neighbors=n, gamma=g, max_iter=100000, tol=0.0001)
            lp.fit(X_train, y_train)
            roc_scores.append(roc_auc_score(y_test, lp.predict_proba(X_test)[:,1]))
    if kernel == 'knn':
        for n in params_list:
            lp = LabelPropagation(kernel=kernel, n_neighbors=n, gamma=g, max_iter=100000, tol=0.0001)
            lp.fit(X_train, y_train)
            roc_scores.append(roc_auc_score(y_test, lp.predict_proba(X_test)[:,1]))
    plt.figure(figsize=(16,8))
    plt.plot(params_list, roc_scores)
    plt.title('Label Propagation ROC AUC with ' + kernel + ' kernel')
    #plt.show()
    print('Best metrics value is at {}'.format(params_list[np.argmax(roc_scores)]))

def labels_spread_test(kernel, hyperparam, alphas, X_train, X_test, y_train, y_test):
    plt.figure(figsize=(20,10))
    n, g = 0, 0
    roc_scores = []
    if kernel == 'rbf':
        g = hyperparam
    if kernel == 'knn':
        n = hyperparam
    for alpha in alphas:
        ls = LabelSpreading(kernel=kernel, n_neighbors=n, gamma=g, alpha=alpha, max_iter=1000, tol=0.001)
        ls.fit(X_train, y_train)
        roc_scores.append(roc_auc_score(y_test, ls.predict_proba(X_test)[:,1]))
    plt.figure(figsize=(16,8))
    plt.plot(alphas, roc_scores)
    plt.title('Label Spreading ROC AUC with ' + kernel + ' kernel')
    #plt.show()
    print('Best metrics value is at {}'.format(alphas[np.argmax(roc_scores)]))

warnings.simplefilter('ignore') #we don't wanna see that
np.random.seed(1)

MDSFlag = False

if MDSFlag:
    #take bug_cnt from data 
    inp = pd.read_csv('editPDE_R3_1.csv')
    #take MDS data 
    data = pd.read_pickle('editPDE_R3_1MDS.pkl')
    df = pd.DataFrame(data)
    df['bug_cnt'] = inp.iloc[: , -1]
    df.head()
    df.info()
    df.describe()
    df = shuffle(df, random_state=1)
    X = df.drop(['bug_cnt'], axis = 1)
    y = df['bug_cnt']
    #sns.pairplot(X)
else:
    inp = pd.read_csv('editPDE_R3_1.csv')
    df = pd.DataFrame(inp)
    df.head()
    df.info()
    df.describe()
    df = shuffle(df, random_state=1)
    X = df.drop(df.columns[-1], axis = 1)
    y = df.iloc[: , -1]
    

X_1, X_2, X_3  = np.split(X, [int(.1*len(X)), int(.5*len(X))])
y_1, y_2, y_3  = np.split(y, [int(.1*len(y)), int(.5*len(y))])
y_1_2 = np.concatenate((y_1, y_2.apply(lambda x: -1)))
X_1_2 = np.concatenate((X_1, X_2))
print(X_1_2)
index = ['Algorithm', 'ROC AUC']
results = pd.DataFrame(columns=index)
logreg = LogisticRegression(random_state=1, class_weight='balanced')
logreg.fit(X_1, y_1)
results = results.append(pd.Series(['Logistic Regression', roc_auc_score(y_3, logreg.predict_proba(X_3)[:,1])], index=index), ignore_index=True)

#print(results)

logreg_test = LogisticRegression(random_state=1, class_weight='balanced')
logreg_test.fit(df, y)
logreg_test.predict_proba(df)
#print(results)  
'''gammas = [9e-6, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5]
label_prop_test('rbf', gammas, X_1_2, X_3, y_1_2, y_3)'''

ns = np.arange(50,60)
label_prop_test('knn', ns, X_1_2, X_3, y_1_2, y_3)

'''lp_rbf = LabelPropagation(kernel='rbf', gamma=9e-6, max_iter=100000, tol=0.0001)
lp_rbf.fit(X_1_2, y_1_2)
results = results.append(pd.Series(['Label Propagation RBF',  roc_auc_score(y_3, lp_rbf.predict_proba(X_3)[:,1])], index=index), ignore_index=True)'''

lp_knn = LabelPropagation(kernel='knn', n_neighbors=53, max_iter=100000, tol=0.0001)
lp_knn.fit(X_1_2, y_1_2)
results = results.append(pd.Series(['Label Propagation KNN', roc_auc_score(y_3, lp_knn.predict_proba(X_3)[:,1])], index=index), ignore_index=True)

'''rbf_lp_test = LabelPropagation(kernel='rbf')
rbf_lp_test.fit(df, y)
rbf_lp_test.predict_proba(df)'''

knn_lp_test = LabelPropagation(kernel='knn')
knn_lp_test.fit(df, y)
knn_lp_test.predict_proba(df)

'''
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  
labels_spread_test('rbf', 1e-5, alphas, X_1_2, X_3, y_1_2, y_3)

alphas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]  
labels_spread_test('knn', 53, alphas, X_1_2, X_3, y_1_2, y_3)

ls_rbf = LabelSpreading(kernel='rbf', gamma=9e-6, alpha=0.6, max_iter=1000, tol=0.001)
ls_rbf.fit(X_1_2, y_1_2)
results = results.append(pd.Series(['Label Spreading RBF', roc_auc_score(y_3, ls_rbf.predict_proba(X_3)[:,1])], index=index), ignore_index=True)
ls_knn = LabelSpreading(kernel='knn', n_neighbors=53, alpha=0.08, max_iter=1000, tol=0.001)
ls_knn.fit(X_1_2, y_1_2)
results = results.append(pd.Series(['Label Spreading KNN', roc_auc_score(y_3, ls_knn.predict_proba(X_3)[:,1])], index=index), ignore_index=True)
'''
print(results)

