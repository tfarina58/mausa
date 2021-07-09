import pandas as pd
import numpy as np
from sklearn.manifold import MDS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('editPDE_R3_2.csv')
labels = np.array(data.iloc[: , -1])
labels = labels.astype(int)
data = data.drop(data.columns[-1], axis = 1)
print(data)

print(labels)

feature_list = list(data.columns)
features = np.array(data)

sc = StandardScaler()
#embedding = MDS(n_components=6)

for i in range (1, 47, 5):
    if i > 1:
        i -= 1
    embedding = MDS(n_components=i, n_jobs = -1, random_state = 42)
    features = sc.fit_transform(features)
    features = embedding.fit_transform(features)
    features = pd.DataFrame(features)
    features.to_pickle("editPDE_R3_2" +str(i) +"pkl")
    #print(features)
    print(embedding.stress_)
    print(i)