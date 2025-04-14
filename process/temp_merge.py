import pandas as pd

df_main = pd.read_csv("../info/random_sample_partition_res.csv")

df_label = pd.read_csv("../info/random_sample_partition_label.csv")

df_res = df_main.merge(df_label[['sample_id','index','manual_label']],on=['sample_id','index'],how='left')
df_res.to_csv("../info/random_sample_partition_merge.csv")