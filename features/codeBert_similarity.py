import json

import pandas as pd
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
import torch
import torch.nn.functional as F

# 加载 CodeBERT 预训练模型和分词器
model_name = "microsoft/codebert-base" #使用 huggingface/transformers 加载 microsoft/codebert-base 模型。
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)

# 计算代码的嵌入表示
def get_code_embedding(code):
    tokens = tokenizer(code, return_tensors="pt", padding=True, truncation=True, max_length=512)
    print(f"Token 数量: {len(tokens['input_ids'][0])}")
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state[:, 0, :]  # 取 [CLS] 位置的向量作为代码表示

# 计算余弦相似度
def cosine_similarity(embedding1, embedding2):
    return F.cosine_similarity(embedding1, embedding2).item()


def get_features_labels(dataPath, for_clf = True):
    item_list = []
    head_flag = False
    with open(dataPath,"r",encoding="utf-8") as f:
        for i, item in enumerate(tqdm(f.readlines())):
            # if i <=len:
            #     continue
            item = json.loads(item)

            item_id = item['sample_id']
            item_index = item['index']

            # 获取嵌入并计算相似度
            embedding1 = get_code_embedding(item['src_method'])
            embedding2 = get_code_embedding(item['dst_method'])
            print("src_method",item['src_method'])
            print("dst_method",item['dst_method'])

            vec1 = embedding1.squeeze().numpy()
            vec2 = embedding2.squeeze().numpy()
            print(f"vec1 mean: {vec1.mean()}, std: {vec1.std()}")
            print(f"vec2 mean: {vec2.mean()}, std: {vec2.std()}")
            print(f"vec1[:10]: {vec1[:10]}")
            print(f"vec2[:10]: {vec2[:10]}")
            similarity = cosine_similarity(embedding1, embedding2)
            print("sim",similarity)

            item_list.append([item_id, item_index,similarity])
            if i == 10:
                exit()
            # df_temp = pd.DataFrame(item_list, columns=['sample_id', 'index', 'sim_score'])
            # if head_flag:
            #     df_temp.to_csv("sim_score_train.csv",index=False)
            #     head_flag =False
            # else:
            #     df_temp.to_csv("sim_score_train.csv",mode='a',index=False,header=False)

    df_sim = pd.DataFrame(item_list,columns=['sample_id','index','sim_score'])
    df_sim.to_csv(f"sim_score_codeBert_{data_type}.csv",index=False)

if __name__ == "__main__":
    data_type = 'test' # train/valid/test
    get_features_labels(f"../dataset/{data_type}_clean.jsonl")