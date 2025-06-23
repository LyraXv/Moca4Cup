'''
    code and comments similarity

'''
import json
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("flax-sentence-embeddings/st-codesearch-distilroberta-base")

def method_semantic_similiarity(src_method, dst_method):
    src_emb = model.encode(src_method,convert_to_tensor=True)
    dst_emb = model.encode(dst_method,convert_to_tensor=True)

    # print("src_method:",src_method)
    # print("dst_method" , dst_method)
    # print("src_emb",src_emb)
    # print("dst_emb",dst_emb)

    # hits = util.semantic_search(src_emb,dst_emb)[0]
    # top_hit = hits[0]
    # score = top_hit['score']
    score = util.cos_sim(src_emb,dst_emb)
    score = round(score.item(), 5)
    return score

def get_features_labels(dataPath, for_clf = True):
    item_list = []
    with open(dataPath,"r",encoding="utf-8") as f:
        for i, item in enumerate(tqdm(f.readlines())):
            item = json.loads(item)

            item_id = item['sample_id']
            item_index = item['index']

            sim_score = method_semantic_similiarity(item['src_method'],item['dst_method'])
            sim_srcComment_dstCode = method_semantic_similiarity(item['src_desc'],item['dst_method'])

            item_list.append([item_id, item_index,sim_score,sim_srcComment_dstCode])

    df_sim = pd.DataFrame(item_list,columns=['sample_id','index','sim_score','sim_cocom'])
    df_sim.to_csv(f"../info/sim_score_{data_type}.csv",index=False)


if __name__ == "__main__":
    data_type = 'valid' # train/valid/test
    get_features_labels(f"../dataset/{data_type}_clean.jsonl")
