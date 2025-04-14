import json
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    df_data = pd.read_csv("../info/codeChangeFeatures_test_2.csv") # target file


    output_sample = []

    dataPath = "../dataset/test_clean.jsonl"
    with open(dataPath,"r",encoding="utf-8") as f:
        for i, item in enumerate(tqdm(f.readlines())):
            item = json.loads(item)

            item_id = item['sample_id']
            item_index = item['index']

            # 限定只读表中数据
            matching_row = df_data[(df_data['sample_id'] == item_id) & (df_data['index'] == item_index)]
            # if matching_row.empty:
            #     continue
            output_sample.append(
                {
                    "sample_id":item_id,
                    "index":item_index,
                    "src_method":item['src_method'],
                    "dst_method":item['dst_method'],
                    'src_comment':item['src_desc'],
                    'dst_comment':item['dst_desc'],
                    "sim_score":matching_row['sim_score'].values[0],
                    'sim_cocom': matching_row['sim_cocom'].values[0],
                    'refactoringTypeNum':matching_row['refactoringTypeNum'].values[0],
                }
            )
    df_output = pd.DataFrame(output_sample)
    df_output.to_csv("../info/changeCodes_test_2.csv")


