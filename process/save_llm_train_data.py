import difflib

import pandas as pd
import json

data_type = "valid"
d = difflib.Differ()
split_flag = True
# 读取 JSONL 格式的数据
df = pd.read_json(f'../dataset/{data_type}_clean.jsonl', lines=True)

# 构造成 JSON 数组格式（一个列表）
data = []

# if split_flag:
#     split_ratio = 0.1
#     num_finutune = int(80591 * split_ratio)

for i, row in df.iterrows():

    # diff code
    diff = d.compare(row['src_method'].splitlines(), row['dst_method'].splitlines())
    diff_str = ""
    for line in diff:
        if line.startswith("?"):
            continue
        diff_str = diff_str + line + "\n"

    item = {
        "instruction": "Please write a new comment according to the old comment and the diff hunk:",
        # "instruction":"Please update the comment of the method:",
        "input": f"diff_hunk:\n{diff_str}\nold comment: {row['src_desc']}\nnew comment: ",
        "output": f"{row['dst_desc']}"
    }
    data.append(item)
    # if i == num_finutune :
    #     break


with open(f'../dataset/llm/{data_type}_cup_all_diff.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

