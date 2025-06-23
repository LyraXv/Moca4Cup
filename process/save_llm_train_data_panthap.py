import difflib

import pandas as pd
import json

data_type = "valid"
d = difflib.Differ()
split_flag = True
# 读取 JSONL 格式的数据
df = pd.read_json(f'../dataset/{data_type}_panthap.json')

# 构造成 JSON 数组格式（一个列表）
data = []

# if split_flag:
#     split_ratio = 0.1
#     num_finutune = int(80591 * split_ratio)

for i, row in df.iterrows():
    # diff code
    diff = d.compare(row['old_code'].splitlines(), row['new_code'].splitlines())
    diff_str = ""
    for line in diff:
        if line.startswith("?"):
            continue
        diff_str = diff_str + line + "\n"
    print(diff_str)
    exit()

    item = {
        "instruction": "Please write a new comment according to the old comment and the changes between the old and new code:",
        # "instruction":"Please update the comment of the method:",
        "input": f"old code: {row['old_code']}\nnew code: {row['new_code']}\nold comment: {row['old_comment']}\nnew comment: ",
        "output": f"{row['new_comment']}"
    }
    data.append(item)
    # if i == num_finutune :
    #     break


with open(f'../dataset/llm/{data_type}_cup_panthap.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

