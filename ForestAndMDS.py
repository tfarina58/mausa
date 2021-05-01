import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
'''data = pd.read_csv('C:\\Users\\Juicero\\Downloads\\PDE_R3_2.csv')

# display
print("Original 'input.csv' CSV Data: \n")
print(data)

data.drop('File', inplace=True, axis=1)

data['bug_cnt'] = data['bug_cnt'].astype(bool)

data.to_csv('C:\\Users\\Juicero\\Downloads\\editPDE_R3_2.csv', index = False)'''

from sklearn.manifold import MDS

SEED = 42

data = pd.read_csv('editJDT_R2_0.csv')
print(data.shape)

print("\nCSV Data after deleting the column 'year':\n")
print(data)

labels = np.array(data['bug_cnt'])
labels = labels.astype(int)
data = data.drop('bug_cnt', axis = 1)

feature_list = list(data.columns)
features = np.array(data)

from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
embedding = MDS(n_components=15)

features = sc.fit_transform(features)
features = embedding.fit_transform(features)

print(features)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = SEED)

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_depth=2, random_state=SEED)
classifier.fit(train_features, train_labels)

# Predicting the Test set results
y_pred = classifier.predict(test_features)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cm = confusion_matrix(test_labels, y_pred)
print(cm)
print('Accuracy', accuracy_score(test_labels, y_pred))