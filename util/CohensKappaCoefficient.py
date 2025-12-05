import pandas as pd
from sklearn.metrics import cohen_kappa_score

from sklearn.metrics import cohen_kappa_score
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../info/manual_label_res.csv")
colunmns = ['label','a', 'b', 'c']
df = df[colunmns]

display_names = {
    'label': 'Model',
    'a': 'Annotator 1',
    'b': 'Annotator 2',
    'c': 'Annotator 3'
}

kappa_matrix = pd.DataFrame(index=['label','a', 'b', 'c'], columns=colunmns)

# 计算四组的Kappa值
for col1 in kappa_matrix.columns:
    for col2 in kappa_matrix.index:
        kappa_matrix.loc[col1, col2] = cohen_kappa_score(df[col1], df[col2])

kappa_matrix = kappa_matrix.astype(float)
kappa_matrix.index = [display_names[col] for col in kappa_matrix.index]
kappa_matrix.columns = [display_names[col] for col in kappa_matrix.columns]

# 画热力图
sns.heatmap(kappa_matrix, annot=True, cmap='Blues',annot_kws={"size": 14})
plt.figure(figsize=(8, 6))
# sns.heatmap(
#     kappa_matrix,
#     xticklabels=raters,
#     yticklabels=raters,
#     annot=True,
#     fmt=".2f",
#     cmap="Blues",
#     annot_kws={"size": 14}  # 放大字体
# )

# plt.title("Cohen's Kappa between predictions")
plt.show()