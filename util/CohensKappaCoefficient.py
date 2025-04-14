import pandas as pd
from sklearn.metrics import cohen_kappa_score

data = pd.read_csv("../info/random_sample_partition_label_2.csv")
print(data.columns)
# data = data[data['cluster_label']==1]
# print(data)

data1 = data['cluster_label']
data2 = data['manual_label']

print(data1)
print(data2)

kappa = cohen_kappa_score(data1, data2)
print("CohensKappa:")
print(kappa)