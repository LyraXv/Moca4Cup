import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import config

df = pd.read_csv("../info/ranked_data_res_train.csv")

# 确保 'Label' 列存在且为分类变量
df['label'] = df['label'].astype(str)  # 转为字符串便于图例显示

# 要画图的特征列（除Label列之外）
feature_cols = config.CONTENT_COLUMNS

# 将宽表转为长表
df_long = pd.melt(df, id_vars='label', value_vars=feature_cols,
                  var_name='Feature', value_name='Value')

# 画箱型图
plt.figure(figsize=(10, 5))
sns.boxplot(x='Feature', y='Value', hue='label', data=df_long)

plt.title('Boxplot of Features by Label')
plt.xlabel('Feature')
plt.ylabel('Value')
plt.yscale('log')
plt.xticks(rotation=45)
plt.legend(title='Label')
plt.tight_layout()
plt.show()