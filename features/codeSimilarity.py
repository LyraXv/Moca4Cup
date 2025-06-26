import json
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

def method_semantic_similiarity(src_method, dst_method):
    model = SentenceTransformer("flax-sentence-embeddings/st-codesearch-distilroberta-base")
    src_emb = model.encode(src_method,convert_to_tensor=True)
    dst_emb = model.encode(dst_method,convert_to_tensor=True)
    hits = util.semantic_search(src_emb,dst_emb)[0]
    top_hit = hits[0]
    score = top_hit['score']
    return score

def get_features_labels(dataPath, for_clf = True):
    item_list = []
    head_flag = False
    with open(dataPath,"r",encoding="utf-8") as f:
        for i, item in enumerate(tqdm(f.readlines())):
            # if i <=len:
            #     continue
            item_list = []
            item = json.loads(item)

            item_id = item['sample_id']
            item_index = item['index']

            sim_score = method_semantic_similiarity(item['src_method'],item['dst_method'])

            item_list.append([item_id, item_index,sim_score])

            # df_temp = pd.DataFrame(item_list, columns=['sample_id', 'index', 'sim_score'])
            # if head_flag:
            #     df_temp.to_csv("sim_score_train.csv",index=False)
            #     head_flag =False
            # else:
            #     df_temp.to_csv("sim_score_train.csv",mode='a',index=False,header=False)

    df_sim = pd.DataFrame(item_list,columns=['sample_id','index','sim_score'])
    df_sim.to_csv(f"{data_type}_sim_score.csv",index=False)
    df_features = pd.read_csv(f"../info/{data_type}_features_size_CC.csv")
    df_sim.set_index(['sample_id','index'],inplace=True)
    df_features.set_index(['sample_id','index'],inplace=True)
    df_features['sim_score'] = df_sim['sim_score']
    df_features.to_csv(f"../info/{data_type}_part_features.csv")


if __name__ == "__main__":
    data_type = 'train' # train/valid/test
    save_data = pd.read_csv(f"../info/sim_score_{data_type}.csv")
    # len = save_data.shape[0]
    get_features_labels(f"../dataset/{data_type}_clean.jsonl")
