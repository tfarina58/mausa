import pandas as pd
import numpy as np
from sklearn.manifold import MDS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('editJDT_R2_0.csv')
print(data.shape)
print(data)

labels = np.array(data['bug_cnt'])
labels = labels.astype(int)
data = data.drop('bug_cnt', axis = 1)

feature_list = list(data.columns)
features = np.array(data)

sc = StandardScaler()
embedding = MDS(n_components=15)

features = sc.fit_transform(features)
features = embedding.fit_transform(features)
features = pd.DataFrame(features)
features.to_pickle("editJDT_R2_0MDS.pkl")
print(features)