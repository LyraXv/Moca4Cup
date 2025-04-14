import csv
import difflib
import json
import math
import string
import lizard
import pandas as pd
from tqdm import tqdm
# from sentence_transformers import SentenceTransformer, util

def itemIsConnect(item):
    connectOp = {'<con>'}
    if item[0] in connectOp or item[1] in connectOp:
        return True
    else:
        return False

# feature : NMT,NNTRP
# <con>,Refactoring seq list,then use func to get NMT and NNTRP
def get_token_change_seq(code_change_seq):
    new_seq_list = []
    # refactor
    # for i in range(len(code_change_seq)):
    i = 0
    while i < len(code_change_seq):
        if not itemIsConnect(code_change_seq[i]):
            new_seq_list.append(code_change_seq[i])
            i += 1
        else:
            temp = new_seq_list.pop()
            src_string = temp[0]
            dst_string = temp[1]
            src_string += code_change_seq[i+1][0]
            dst_string += code_change_seq[i+1][1]
            if src_string==dst_string:
                temp_list = [src_string,dst_string,'equal']
            elif src_string == '':
                temp_list = [src_string,dst_string,'insert']
            elif dst_string == '':
                temp_list = [src_string, dst_string, 'delete']
            else:
                temp_list = [src_string, dst_string, 'replace']
            new_seq_list.append(temp_list)
            i += 2
    # print("new_seq_list",new_seq_list)
    return new_seq_list


# feature : NMS
def get_modified_sub_tokens_num(code_change_seq):
    modified_sub_tokens_num = 0
    for token in code_change_seq:
        if not itemIsConnect(token):
            if token[2] != "equal":
                modified_sub_tokens_num += 1
    return modified_sub_tokens_num

# feature : NML
def get_modified_lines(src_method, dst_method):
    old_code = src_method  # 旧代码
    new_code = dst_method # 新代码

    d = difflib.Differ()
    diff = list(d.compare(old_code.splitlines(), new_code.splitlines()))
    # for line in diff:
    #     print(line)

    added_lines = len([line for line in diff if line.startswith('+')])
    removed_lines = len([line for line in diff if line.startswith('-')])

    return max(added_lines,removed_lines)

# feature: NMC
def get_modified_chunks(code_change_seq):
    continous = False
    cnt = 0
    for token in code_change_seq:
        if token[2] != "equal" and continous==False:
            continous = True
            cnt +=1
        elif token[2] == "equal":
            continous = False
    return cnt

# NMC_new
def get_modified_chunk(src_method, dst_method):
    old_code = src_method  # 旧代码
    new_code = dst_method # 新代码

    d = difflib.Differ()
    diff = list(d.compare(old_code.splitlines(), new_code.splitlines()))

    change_chunk_count = 0
    current_chunk = False  # 标志变量，表示是否在一个变化块中

    for line in diff:
        print("x",line)
        if line.startswith("+ ") or line.startswith("- "):  # 变化的行
            if not current_chunk:  # 如果当前不在变化块中，开始一个新的变化块
                change_chunk_count += 1
                current_chunk = True
        elif line.startswith("? "):
            # 跳过 ? 行，因为它只是标示差异位置
            continue
        else:
            current_chunk = False  # 变化结束，标志清除
    print(old_code)
    print(new_code)
    print(change_chunk_count)


#feature : NNSRP
def get_NNSRP(code_change_seq):
    modified_sub_tokens= set()
    for token in code_change_seq:
        if not itemIsConnect(token):
            if token[2] != "equal":
                # print(token)
                modified_sub_tokens.add(tuple(token))
    return len(modified_sub_tokens)

# feature : NSOD
def get_NSOD(code_change_seq,desc_change_seq):
    consisitency_tokens = set()
    dist_code_tokens = set()

    for token in code_change_seq:
        dist_code_tokens.add(token[1])
        for desc_token in desc_change_seq:
            if token[0]==desc_token[0]:
                consisitency_tokens.add(token[0])

    # filter punctuations and "<con>"
    consisitency_tokens = symbol_filter(consisitency_tokens)
    dist_code_tokens = symbol_filter(dist_code_tokens)

    NTOD = 0
    for token in consisitency_tokens:
        if token not in dist_code_tokens:
            NTOD += 1
    return NTOD


def symbol_filter(tokens_set):
    for elem in list(tokens_set):
        if elem in string.punctuation or elem == '<con>':
            tokens_set.discard(elem)
    return tokens_set


# feature : CC
def get_CC(src_method,dst_method):
    # 计算圈复杂度
    src_analysis = lizard.analyze_file.analyze_source_code("", src_method)
    dst_analysis = lizard.analyze_file.analyze_source_code("",dst_method)

    try:
        src_func = src_analysis.function_list[0]
        dst_func = dst_analysis.function_list[0]
    except IndexError:
        print(src_method)
        print("===")
        print(dst_method)
        return 0

    delta_cc = src_func.cyclomatic_complexity - dst_func.cyclomatic_complexity
    # print("src",src_func.cyclomatic_complexity,"dst",dst_func.cyclomatic_complexity,"diff_cc",diff_cc)
    return abs(delta_cc)

def save_csv_file(savaPath, data, headers):
    headers = headers
    with open(savaPath , 'w' , newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        writer.writerows(data)
    print("Data has been saved to", savaPath)

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

    with open(dataPath,"r",encoding="utf-8") as f:
        for i, item in enumerate(tqdm(f.readlines())):
            item = json.loads(item)

            item_id = item['sample_id']
            item_index = item['index']

            NMC = get_modified_chunk(item['src_method'],item['dst_method'])
            exit()

            # sim_score = method_semantic_similiarity(item['src_method'],item['dst_method'])

            # item_list.append([item_id, item_index,sim_score])

    # df_sim = pd.DataFrame(item_list,columns=['sample_id','index','sim_score'])
    # df_sim.to_csv(f"{data_type}_sim_score.csv",index=False)
    # df_features = pd.read_csv(f"../info/{data_type}_features_size_CC.csv")
    # df_sim.set_index(['sample_id','index'],inplace=True)
    # df_features.set_index(['sample_id','index'],inplace=True)
    # df_features['sim_score'] = df_sim['sim_score']
    # df_features.to_csv(f"../info/{data_type}_part_features.csv")

    '''# Dim: Size
            token_code_list = get_token_change_seq(item['code_change_seq'])
            NMT = get_modified_sub_tokens_num(token_code_list)
            NMS = get_modified_sub_tokens_num(item['code_change_seq'])
            NML = get_modified_lines(item['src_method'],item['dst_method'])
            NMC = get_modified_chunks(item['code_change_seq'])
            NNTRP = get_NNSRP(token_code_list)
            NNSRP = get_NNSRP(item['code_change_seq'])


            NSOD = get_NSOD(item['code_change_seq'],item['desc_change_seq'])
            token_desc_list = get_token_change_seq(item['desc_change_seq'])
            NTOD = get_NSOD(token_code_list,token_desc_list)

            #Dim: Structre
            CC_delta = get_CC(item['src_method'],item['dst_method'])

            # item_list.append([item_id,item_index,NMT,NMS,NML,NMC,NNTRP,NNSRP,NTOD,NSOD,CC_delta,sim_score])
            '''
    return item_list



if __name__ == "__main__":
    data_type = 'valid' # train/valid/test
    get_features_labels(f"../dataset/{data_type}_clean.jsonl")
    # 获取特征数据
    # item_features = get_features_labels(f"../dataset/{data_type}_clean.jsonl")
    # Save
    # headers = ['sample_id', 'index','NMT', 'NMS', 'NML', 'NMC', 'NNTRP', 'NNSRP','NTOD','NSOD', 'CC_delta']
    # save_csv_file(f"../info/{data_type}_features_size_CC.csv", item_features, headers)