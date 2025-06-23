import json

import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
import json
# import torch
# import faiss
import numpy as np
from tqdm import tqdm
# from transformers import RobertaTokenizer, RobertaModel

# def save_structure_sim_faiss(dataPath):
#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     samples = []
#     with open(dataPath, "r", encoding="utf-8") as f:
#         for line in f:
#             item = json.loads(line)
#             samples.append(item)
#
#     # 先按等级分类
#     level_groups = {}
#     for sample in samples:
#         level = sample["change_level_label"]
#         level_groups.setdefault(level, []).append(sample)
#
#     # 为每个等级构建 FAISS 向量库
#     faiss_indices = {}
#     for level, samples in level_groups.items():
#         texts = [s["src_code"] for s in samples]
#         vecs = model.encode(texts, normalize_embeddings=True)
#
#         index = faiss.IndexFlatIP(vecs.shape[1]) # Inner Product (cosine similarity if normalized)
#
#         index.add(vecs)
#
#         faiss_indices[level] = index
#
#     faiss.write_index(faiss_indices[0],"index_simple.faiss")
#     faiss.write_index(faiss_indices[1],"index_normal.faiss")
#     faiss.write_index(faiss_indices[2],"index_complex.faiss")


# def structure_sim_search(model,change_level,faiss_indices,content,topN):
#     index = faiss_indices[change_level] # 需要将数字改成字符
#
#     vector = model.encode([content],normalize_embedding=True)
#
#     D, I = index.search(vector,topN) # D: similarity score; I: similar samples index
#
#     print(D,I)
#     exit()
#     return D,I




#
#
# def query_with_label_filter(old_index, new_index, id_mapping, data_mapping, label_mapping,
#                             query_old_code, query_new_code, label_a, top_k=5):
#     # 筛选出 label 为 a 的索引项
#     allowed_keys = [key for key, label in label_mapping.items() if str(label) == str(label_a)]
#     key_to_faiss_id = {v: k for k, v in id_mapping.items()}
#     allowed_faiss_ids = [k for k in key_to_faiss_id if id_mapping[k] in allowed_keys]
#
#     if len(allowed_faiss_ids) == 0:
#         print("No samples found for label:", label_a)
#         return []
#
#     # 获取允许范围的向量
#     new_vecs = np.vstack([new_index.reconstruct(i) for i in allowed_faiss_ids])
#     old_vecs = np.vstack([old_index.reconstruct(i) for i in allowed_faiss_ids])
#
#     q_old_vec = encode([query_old_code])
#     q_new_vec = encode([query_new_code])
#
#     # 相似度
#     sim_new = np.dot(q_new_vec, new_vecs.T)[0]
#     sim_old = np.dot(q_old_vec, old_vecs.T)[0]
#     combined_score = 0.5 * sim_new + 0.5 * sim_old
#
#     sorted_idx = np.argsort(-combined_score)[:top_k]
#
#     top_results = []
#     for i in sorted_idx:
#         real_id = allowed_faiss_ids[i]
#         key = id_mapping[real_id]
#         sample = data_mapping[key]
#         top_results.append({
#             "key": key,
#             "score": float(combined_score[i]),
#             "label": label_mapping[key],
#             "id": sample["id"],
#             "index": sample["index"],
#             "old_code": sample["old_code"],
#             "new_code": sample["new_code"]
#         })
#
#     return top_results

def change_type_sim(sample_list,query_list):
    sample_set = set(sample_list)
    query_set = set(query_list)

    intersection = len(sample_set&query_set)
    union = len(sample_set|query_set)

    return intersection/union if union!=0 else 0.0


def test_change_type_sim():
    datapath = "../info_process/train_cup_withFeaturesandLabel.jsonl"
    list_a,list_b =[],[]
    with open(datapath, 'r', encoding='utf8') as f:
        for i, x in enumerate(tqdm(f.readlines())):
            fileInfo = json.loads(x)
            if i == 8:
                list_a =fileInfo['ChangeType']
            if i > 15:
                list_b = fileInfo['ChangeType']
                print(list_a)
                print(list_b)
                print(change_type_sim(list_a, list_b))
            if i == 35:
                exit()





def main(dataPath):
    model = joblib.load("../model/lightGBM.pkl")
    scaler = joblib.load("../model/scaler")

    # from openai import OpenAI
    # client = OpenAI(
    #     api_key="{}".format(os.getenv("API_KEY", "0")),
    #     base_url="http://114.212.190.204:{}/v1".format(os.getenv("API_PORT", 8001)),
    # )

    semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    # faiss_indices = {}
    # for level in ['simple', 'normal', 'complex']:
    #     index_tmp = faiss.read_index(f"index_{level}.faiss")
    #     faiss_indices[level] = index_tmp

    with open(dataPath, 'r', encoding='utf8') as f:
        for i, x in enumerate(tqdm(f.readlines())):
            if i == 1:
                exit()
            print("*" * 5, "new code change")
            fileInfo = json.loads(x)

            src_code = fileInfo['src_method']
            dst_code = fileInfo['dst_method']
            src_comments = fileInfo['src_desc']
            ref_comments = fileInfo['dst_desc']
            print("src_comments:", src_comments, "\nref_comments:", ref_comments)

            # classifier
            features = pd.DataFrame([fileInfo["features"]])
            features_scaled = scaler.transform(features)
            change_level = model.predict(features_scaled)
            print("changeLevel: ", change_level)

            # 语义相似代码检索
            # D,I = structure_sim_search(semantic_model, change_level, faiss_indices, src_code, topN=20)


            # 变化相似检索
            # data:变化前后的代码+变化类型集合

            # 先差分再计算相似度

            # 变化类型集合计算

            # 混合计算

            # 输出 topK






    # 分类器

    # 语义相似代码检索

    # 变化相似检索

    # 构建模版



if __name__ == "__main__":
    # model_name = "microsoft/graphcodebert-base"
    # tokenizer = RobertaTokenizer.from_pretrained(model_name)
    # model = RobertaModel.from_pretrained(model_name).eval().to("cuda" if torch.cuda.is_available() else "cpu")
    # device = next(model.parameters()).device

    print("Loaded_model!")

    # 构建索引和映射
    # build_dual_faiss_indices_with_labels(
    #     jsonl_path="../dataset/train_clean.jsonl",
    #     old_index_path="../model/faiss_old.index",
    #     new_index_path="../model/faiss_new.index",
    #     mapping_path="../retriever_sample_map.json"
    # )

    # main()


    test_change_type_sim()
    # save_semantic_sim_faiss()
