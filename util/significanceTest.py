import pandas as pd
from scipy.stats import mannwhitneyu
from cliffs_delta import cliffs_delta

def analyze_feature(group0,group1,feature):
    x = group0[feature].values
    y = group1[feature].values

    # Mann-Whitney U Test
    u_stat, p_value = mannwhitneyu(x, y, alternative='two-sided')

    # Cliff's Delta
    delta, magnitude= cliffs_delta(x, y)

    return u_stat, p_value, delta ,magnitude

def read_splitted_data():
    df = pd.read_csv("../info/ranked_data_res_test.csv")
    return df

if __name__ == "__main__":
    data = read_splitted_data()
    key_columns = ['sample_id','index']
    df = data.drop(columns=key_columns)

    # 分组数据
    group0 = df[df['label'] == 0]
    group1 = df[df['label'] == 2]
    # group1['label'] =1

    # 逐特征分析
    results = {}
    features = [col for col in df.columns if col!='label']
    print("\nfeatures:")
    print(features)
    for feature in features:
        u_stat, p_value, delta, magnitude= analyze_feature(group0,group1,feature)
        results[feature] = {'U-stat': u_stat, 'p-value': p_value, 'Cliff\'s Delta': delta,'Magnitude': {magnitude}}

    # 输出结果
    results_df = pd.DataFrame(results).T
    print(results_df)
    results_df.to_csv("../info/mannwhitneyu_all_dims.csv")


