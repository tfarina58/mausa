import pandas as pd
import numpy as np
from sklearn.manifold import MDS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('CM1.data')
labels = np.array(data.iloc[: , -1])
labels = labels.astype(int)
data = data.drop(data.columns[-1], axis = 1)

print(data)

print(labels)

feature_list = list(data.columns)
features = np.array(data)

sc = StandardScaler()
embedding = MDS(n_components=15)

features = sc.fit_transform(features)
features = embedding.fit_transform(features)
features = pd.DataFrame(features)
features.to_pickle("CM1dataMDS.pkl")
print(features)