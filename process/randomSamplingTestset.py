import json
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    df_clustered_set = pd.read_csv("../info/ranked_data_all_intersectionRes.csv")

    df_label_0 = df_clustered_set[df_clustered_set['label']==0]
    df_label_1 = df_clustered_set[df_clustered_set['label']==1]
    df_label_2 = df_clustered_set[df_clustered_set['label']==2]


    df_0_id = df_label_0.sample(n=384,random_state=42)
    df_1_id = df_label_1.sample(n=384,random_state=42)
    df_2_id = df_label_2.sample(n=384,random_state=42)

    output_sample = []

    dataPath = "../dataset/test_clean.jsonl"
    with open(dataPath,"r",encoding="utf-8") as f:
        for i, item in enumerate(tqdm(f.readlines())):
            item = json.loads(item)

            item_id = item['sample_id']
            item_index = item['index']

            matching_row_0 = df_0_id[(df_0_id['sample_id'] == item_id) & (df_0_id['index'] == item_index)]
            matching_row_1 = df_1_id[(df_1_id['sample_id'] == item_id) & (df_1_id['index'] == item_index)]
            matching_row_2 = df_2_id[(df_2_id['sample_id'] == item_id) & (df_2_id['index'] == item_index)]

            if matching_row_0.empty and matching_row_1.empty and matching_row_2.empty:
                continue
            if not matching_row_0.empty:
                label = 0
            elif not matching_row_1.empty:
                label = 1
            else:
                label = 2
            output_sample.append(
                {
                    "sample_id":item_id,
                    "index":item_index,
                    "src_method":item['src_method'],
                    "dst_methhod":item['dst_method'],
                    "cluster_label":label
                }
            )
    df_output = pd.DataFrame(output_sample)
    df_output.to_csv("../info/random_sample_partition_res_intersection.csv")


