import pandas as pd
import numpy as np
from sklearn.manifold import MDS
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

SEED = 42

MDSFlag = False

if MDSFlag:
    inp = pd.read_csv('editPDE_R3_1.csv')
    labels = np.array(inp.iloc[: , -1])
    labels = labels.astype(int)
    features = pd.read_pickle('editPDE_R3_1MDS.pkl')
else:
    features = pd.read_csv('editPDE_R3_1.csv')
    labels = np.array(features.iloc[: , -1])
    labels = labels.astype(int)
    features = features.drop(features.columns[-1], axis = 1)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from imblearn.under_sampling import RandomUnderSampler

#train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = SEED)

features, labels = shuffle(features, labels, random_state=42) # shuffle the data

test_ind = round(len(features)*0.25) #Za testiranje
train_ind = test_ind + round(len(features)*0.05) #Labeled
unlabeled_ind = train_ind + round(len(features)*0.70) #Unlabeled

test = features.iloc[:test_ind]
train = features.iloc[test_ind:train_ind]
unlabeled = features.iloc[train_ind:unlabeled_ind]

X_train = train
y_train = pd.DataFrame(labels[test_ind:train_ind])
print(y_train)
X_unlabeled = unlabeled

X_test = test
y_test = labels[:test_ind]
undersample = RandomUnderSampler(sampling_strategy='majority')

X_train, y_train = undersample.fit_resample(X_train, y_train)
iterations = 0

# Containers to hold f1_scores and # of pseudo-labels
train_f1s = []
test_f1s = []
pseudo_labels = []

# Assign value to initiate while loop
high_prob = [1] 
from sklearn.metrics import  roc_auc_score
# Loop will run until there are no more high-probability pseudo-labels
while len(high_prob) > 0:
        
    # Fit classifier and make train/test predictions
    clf = LogisticRegression(max_iter=10000)
    clf.fit(X_train, y_train.values.ravel())
    y_hat_train = clf.predict(X_train)
    y_hat_test = clf.predict(X_test)

    # Calculate and print iteration # and f1 scores, and store f1 scores
    train_f1 = f1_score(y_train, y_hat_train)
    test_f1 = f1_score(y_test, y_hat_test)
    print(f"Iteration {iterations}")
    print(f"Train f1: {train_f1}")
    print(f"Test f1: {test_f1}")
    train_f1s.append(train_f1)
    test_f1s.append(test_f1)
    print("AUC: ", roc_auc_score(y_test, y_hat_test))
    # Generate predictions and probabilities for unlabeled data
    print(f"Now predicting labels for unlabeled data...")

    pred_probs = clf.predict_proba(X_unlabeled)
    preds = clf.predict(X_unlabeled)
    prob_0 = pred_probs[:,0]
    prob_1 = pred_probs[:,1]

    # Store predictions and probabilities in dataframe
    df_pred_prob = pd.DataFrame([])
    df_pred_prob['preds'] = preds
    df_pred_prob['prob_0'] = prob_0
    df_pred_prob['prob_1'] = prob_1
    df_pred_prob.index = X_unlabeled.index
    
    # Separate predictions with > 99% probability
    high_prob = pd.concat([df_pred_prob.loc[df_pred_prob['prob_0'] > 0.90],
                           df_pred_prob.loc[df_pred_prob['prob_1'] > 0.90]],
                          axis=0)
    
    print(f"{len(high_prob)} high-probability predictions added to training data.")
    
    pseudo_labels.append(len(high_prob))

    # Add pseudo-labeled data to training data
    X_train = pd.concat([X_train, X_unlabeled.loc[high_prob.index]], axis=0)
    y_train = pd.concat([y_train, high_prob.preds])      
    
    # Drop pseudo-labeled instances from unlabeled data
    X_unlabeled = X_unlabeled.drop(index=high_prob.index)
    print(f"{len(X_unlabeled)} unlabeled instances remaining.\n")
    
    # Update iteration counter
    iterations += 1



