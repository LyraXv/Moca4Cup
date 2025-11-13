import json
from collections import defaultdict

import pandas as pd
from tqdm import tqdm


def analyze_change_type_info(changeTypeList):
    refined_type = ['STATEMENT_UPDATE', "STATEMENT_INSERT", "STATEMENT_DELETE", "CONDITION_EXPRESSION_CHANGE",
                    "RETURN_TYPE_CHANGE"]
    change_type_set = set()
    if not changeTypeList:
        return change_type_set
    for changeInfo in changeTypeList:
        significantLevel = changeInfo[1]
        if significantLevel == 'NONE':
            continue
        changeType = changeInfo[0]
        # if changeType in refined_type:
        changeContent = changeInfo[2]
        prefix = changeContent.split(':', 1)[0].strip()
        combined = f"{changeType}:{prefix}"
        change_type_set.add(combined)

    return change_type_set

def load_change_type_jsonl(dataPath):
    lookup_dict = defaultdict(dict)
    with open(dataPath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            key = (data['sample_id'], data['index'])
            lookup_dict[key] = data  # 存储整行数据
    return lookup_dict


DATA_TYPE ='valid'
# 加载特征数据
features_df = pd.read_csv(f'../info/ranked_data_res_{DATA_TYPE}.csv')
change_label_df = features_df[['sample_id','index','label']]
features_df = features_df.drop(columns=['Unnamed: 0','label', 'size_label',
       'size_score', 'structure_label', 'structure_score', 'content_label',
       'content_score'])
features_df.set_index(['sample_id', 'index'], inplace=True)
change_label_df.set_index(['sample_id', 'index'], inplace=True)
# change Type
total_change_type_set = set()
change_type_path = f"../info/ChangeTypeRes_{DATA_TYPE}.jsonl"
change_type_dict = load_change_type_jsonl(change_type_path)



## 取600组样本
samples_num = [0,0,0]

# 打开新的jsonl文件用于写入
with open(f'../info_process/{DATA_TYPE}_cup_withFeatures.jsonl', 'w') as fout, open(f'../dataset/{DATA_TYPE}_clean.jsonl', 'r') as fin:
    for line in fin:
        data = json.loads(line)

        id_ = data['sample_id']
        index_ = data['index']

        try:
            feature_row = features_df.loc[(id_, index_)]
        except KeyError:
            print(f"Feature not found for id={id_}, index={index_}, skipping.")
            continue

        # 加入features
        data['features'] = feature_row.to_dict()

        try:
            change_row = change_label_df.loc[(id_, index_)]
        except KeyError:
            print(f"ChangeLabel not found for id={id_}, index={index_}, skipping.")
            continue

        # current_change_level = change_row.values[0]
        # samples_num[current_change_level] += 1
        # if samples_num[current_change_level] >200:
        #     continue

        data['change_level_label'] = int(change_row.values[0])

        # 提取并保存变化类型数据
        key = (str(id_), str(index_))
        matched_data = change_type_dict.get(key, None)


        change_type_set = analyze_change_type_info(matched_data['codeChangeType'])
        total_change_type_set.update(change_type_set)
        data['ChangeType'] = list(change_type_set)

        # data['codeChangeType'] = matched_data['codeChangeType']

        # 保存到新文件
        fout.write(json.dumps(data) + '\n')

