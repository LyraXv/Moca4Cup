import argparse
import difflib
import json
import os

os.environ["MKL_THREADING_LAYER"] = "GNU" # 避免MKL和GNU数据库冲突
import time
from collections import defaultdict

import joblib
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import torch
import faiss
from transformers import RobertaTokenizer, RobertaModel,logging
import sys
from pathlib import Path
from vllm import LLM, SamplingParams

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from dfg_parser import (remove_comments_and_docstrings,
                    tree_to_token_index,
                    index_to_code_token,
                    tree_to_variable_index, DFG_java)
from tree_sitter import Language, Parser

logging.set_verbosity_error()

def encode(texts):
    batch = tokenizer(texts, padding='max_length', truncation=True, return_tensors="pt", max_length=512)
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        out = roberta_model(**batch)
        cls_vecs = out.last_hidden_state[:, 0, :]
        cls_vecs = torch.nn.functional.normalize(cls_vecs, p=2, dim=1)
    return cls_vecs.cpu().numpy()


def _batch_encode_texts(texts, batch_size=16):
    all_vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i:i + batch_size]
        vec = encode(batch)
        all_vecs.append(vec)
    return np.vstack(all_vecs)


# remove comments, tokenize code and extract dataflow
def extract_dataflow(code, parser, lang):
    # remove comments
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
        # obtain dataflow
    try:
        tree = parser[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split('\n')
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except:
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG
    except:
        dfg = []
    # print("index_to_code", index_to_code)
    # print("DFG", DFG)
    # print("dfg", dfg)

    return index_to_code, dfg


def extract_subcode_by_dfg(index_to_code, dfg, code):
    # 1. 构建变量到行号的映射字典
    var_line_mapping = {}
    for var_info in dfg:
        var_name = var_info[0]
        index = var_info[1]  # 获取变量在index_to_code中的关键索引

        # 找到变量对应的代码位置
        for pos, (idx, token) in index_to_code.items():
            if idx == index:
                line_number = pos[0][0]  # 获取行号
                if var_name not in var_line_mapping:
                    var_line_mapping[var_name] = set()
                var_line_mapping[var_name].add(line_number)
                break

    # 2. 收集所有涉及的行号
    all_lines = set()
    for lines in var_line_mapping.values():
        all_lines.update(lines)

    # 3. 按原始行号排序并抽取代码
    code_lines = code.split('\n')
    subcode_lines = []
    for line_num in sorted(all_lines):
        if line_num < len(code_lines):
            subcode_lines.append(code_lines[line_num])

    # 4. 返回结果
    return '\n'.join(subcode_lines)


def extract_dfg_and_code(parser, code):
    index_to_code, dfg = extract_dataflow(code, parser, 'java')
    var_list = []
    for df in dfg:
        var_list.append(df[0])
    var_str = str(var_list).replace(',', ' ').replace('\'', '')
    # print(var_str)
    new_code = extract_subcode_by_dfg(index_to_code, dfg, code)
    input_str = "[CLS]" + new_code + "[SEP]" + var_str
    return input_str


def get_dfg_parser():
    LANGUAGE = Language('dfg_parser/my-languages.so', 'java')
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, DFG_java]
    return parser


def build_dual_faiss_indices(jsonl_path, save_path):
    samples = []
    parser = get_dfg_parser()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))

    change_level_dict = {"easy": 0, "normal": 1, "complex": 2}
    for level_name, change_level in change_level_dict.items():
        sub_samples = [s for s in samples if s['change_level_label'] == change_level]
        print("sub_samples_num:", len(sub_samples))
        keys = [f"{s['sample_id']}@@{s['index']}" for s in sub_samples]

        # AST, DFG, VariableList
        old_texts = [extract_dfg_and_code(parser, s["src_method"]) for s in sub_samples]
        new_texts = [extract_dfg_and_code(parser, s["dst_method"]) for s in sub_samples]

        old_emb = _batch_encode_texts(old_texts)
        new_emb = _batch_encode_texts(new_texts)

        faiss.normalize_L2(old_emb)
        faiss.normalize_L2(new_emb)

        dim = old_emb.shape[1]
        ids = np.arange(len(sub_samples)).astype("int64")

        old_index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        new_index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))

        old_index.add_with_ids(old_emb, ids)
        new_index.add_with_ids(new_emb, ids)

        old_index_path = f"{save_path}/dfg_{level_name}_faiss_old.index"
        new_index_path = f"{save_path}/dfg_{level_name}_faiss_new.index"
        faiss.write_index(old_index, old_index_path)
        faiss.write_index(new_index, new_index_path)

        id_mapping = {int(i): keys[i] for i in range(len(sub_samples))}
        data_mapping = {keys[i]: sub_samples[i] for i in range(len(sub_samples))}
        mapping_path = f"{save_path}/dfg_{level_name}_faiss_mapping.json"
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump({
                "id_mapping": id_mapping,
                "data_mapping": data_mapping,
            }, f, indent=2)


def load_dual_indices(model_path):
    change_level_dict = {"easy": 0, "normal": 1, "complex": 2}
    faiss_index_dict = {}
    for level_name, change_level in change_level_dict.items():
        old_index_path = f"{model_path}/dfg_{level_name}_faiss_old.index"
        new_index_path = f"{model_path}/dfg_{level_name}_faiss_new.index"
        mapping_path = f"{model_path}/dfg_{level_name}_faiss_mapping.json"

        old_index = faiss.read_index(old_index_path)
        new_index = faiss.read_index(new_index_path)
        with open(mapping_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        id_mapping = {int(k): v for k, v in data["id_mapping"].items()}
        data_mapping = data["data_mapping"]
        faiss_index_dict[str(change_level)] = {"old_index": old_index, "new_index": new_index,
                                               "data_mapping": data_mapping, "id_mapping": id_mapping}
    return faiss_index_dict


def remove_duplicate_comments(data_list):
    seen_comments = set()
    result_list = []

    for data in data_list:
        src_method = data["src_method"]
        dst_method = data["dst_method"]

        combine_comments = src_method + "<con>" + dst_method
        if combine_comments not in seen_comments:
            result_list.append(data)
            seen_comments.add(combine_comments)
    return result_list


def query_structural_sim(parser, old_index, new_index, id_mapping, data_mapping, query_old_code, query_new_code,
                         top_k=5):
    print("Begin to query:")
    # convert code

    old_code = extract_dfg_and_code(parser, query_old_code)
    new_code = extract_dfg_and_code(parser, query_new_code)

    q_old_vec = encode([old_code])
    q_new_vec = encode([new_code])

    old_score, sim_old_index = old_index.search(q_old_vec, top_k)
    new_score, sim_new_index = new_index.search(q_new_vec, top_k)

    sim_old_dict = {int(i): s for i, s in zip(sim_old_index[0], old_score[0]) if i != -1}
    sim_new_dict = {int(i): s for i, s in zip(sim_new_index[0], new_score[0]) if i != -1}

    common_ids = set(sim_old_dict.keys()) & set(sim_new_dict.keys())
    if not common_ids:
        print("No intersection found between topN results.")
        return []

    # 计算综合得分并排序
    result_list = []
    for idx in common_ids:
        combined_score = 0.5 * sim_old_dict[idx] + 0.5 * sim_new_dict[idx]
        key = id_mapping[idx]
        sample = data_mapping[key]
        result_list.append({
            "key": key,
            "score": float(combined_score),
            "sample_id": sample["sample_id"],
            "index": sample["index"],
            "src_method": sample["src_method"],
            "dst_method": sample["dst_method"],
            "src_desc": sample["src_desc"],
            "dst_desc": sample["dst_desc"],
            "change_type": sample["ChangeType"],
        })

    ## 按sample_id 过滤
    unique_by_sample = {}
    for item in result_list:
        sid = item["sample_id"]
        if sid not in unique_by_sample or item["score"] > unique_by_sample[sid]["score"]:
            unique_by_sample[sid] = item
    unique_results = list(unique_by_sample.values())

    unique_results.sort(key=lambda x: -x["score"])
    # print(f"旧相似集合：{len(sim_old_dict)},新相似集合：{len(sim_new_dict)},过滤前:{len(result_list)}过滤后：{len(unique_results)}")
    top_results = remove_duplicate_comments(unique_results)
    top_results.sort(key=lambda x: -x["score"])
    print(
        f"旧相似集合：{len(sim_old_dict)},新相似集合：{len(sim_new_dict)},过滤前:{len(result_list)}过滤后：{len(top_results)}")

    return top_results



def get_specific_diff(src_code, dst_code):
    diff = list(difflib.ndiff(src_code.splitlines(), dst_code.splitlines()))
    changes_add = [line[2:] for line in diff if line.startswith('+ ')]
    changes_sub = [line[2:] for line in diff if line.startswith('- ')]
    return "\n".join(changes_add),"\n".join(changes_sub)

def diff_change_similarity(src_code, dst_code, rec_src_code, rec_dst_code):
    src_diff_add,src_diff_sub = get_specific_diff(src_code, dst_code)
    rec_diff_add,rec_diff_sub = get_specific_diff(rec_src_code, rec_dst_code)
    return difflib.SequenceMatcher(None, src_diff_add, rec_diff_add).ratio()+difflib.SequenceMatcher(None, src_diff_sub, rec_diff_sub).ratio()



def change_type_similarity(sample_list, query_list):
    sample_set = set(sample_list)
    query_set = set(query_list)

    intersection = len(sample_set & query_set)
    union = len(sample_set | query_set)

    return intersection / union if union != 0 else 0.0


def get_list_topk(dataList, k, key_colunmns='score'):
    sorted_data_list = sorted(dataList, key=lambda x: x[key_colunmns], reverse=True)
    return sorted_data_list[:k]


def update_comments(dataPath, ignore_case , K, sample_K,top_p,outputPath):
    classifier = joblib.load("ours/model/lightGBM.pkl")
    scaler = joblib.load("ours/model/scaler")
    parser = get_dfg_parser()

    # 加载索引和映射
    print("Loading index")
    faiss_index_dict = load_dual_indices("ours/model")


    if code_template == 'diff':
        d = difflib.Differ()

    if client_method == "fine-tuning":
        client = OpenAI(
            api_key="{}".format(os.getenv("API_KEY", "0")),
            base_url="http://xxxxxx:{}".format(os.getenv("API_PORT", port)),
        )
    elif client_method == "remote":
        client = OpenAI(
            api_key="xxxxxxx",
            base_url="xxxxxxxxxxxxxxxxxxxxxxxx"
        )
        # # deepseek-v3.1
        # client = OpenAI(
        #     api_key="xxxxxxx",
        #     base_url="https://api.deepseek.com",
        # )


    elif client_method == "local":
        llm = LLM(
            model=f"../model/{LLM_model}",
            tensor_parallel_size=2,  # 使用2张GPU
        )
    else:
        print("Client method not recognized")
        sys.exit(0)

    os.makedirs('ours/output', exist_ok=True)

    # 检查文件是否存在且非空
    out_file_path = Path(outputPath)
    file_exists = out_file_path.exists()
    if file_exists:
        existing_pairs = set()
        with open(outputPath, "r", encoding="utf-8") as f:

            for line in f:
                try:
                    record = json.loads(line.strip())
                    record_id = int(record.get("sample_id"))
                    record_index = int(record.get("index"))
                    existing_pairs.add((record_id, record_index))
                except Exception as e:
                    print(e)

    if file_exists and out_file_path.stat().st_size > 0:
        mode = 'a'
    else:
        mode = 'w'  # 写入模式

    queries_list = []
    with open(outputPath, mode) as fout, open(dataPath, 'r', encoding='utf8') as f,open(readablePath,mode) as readableOut:
        for i, x in enumerate(tqdm(f.readlines())):
            # if i == 100:
            #     break

            fileInfo = json.loads(x)
            if file_exists:
                pair = (int(fileInfo['sample_id']), int(fileInfo['index']))
                if pair in existing_pairs:
                    continue

            # if i %10 ==0:
            #     time.sleep(30)

            src_code = fileInfo['src_method']
            dst_code = fileInfo['dst_method']
            src_comments = fileInfo['src_desc']
            ref_comments = fileInfo['dst_desc']
            print(f"Currently processing {fileInfo['sample_id']} index: {fileInfo['index']}")
            print("src_comments:",src_comments,"\nref_comments:",ref_comments)

            # classifier
            features = pd.DataFrame([fileInfo["features"]])
            features_scaled = scaler.transform(features)
            features_scaled = pd.DataFrame(features_scaled, columns=features.columns)
            change_level =int(classifier.predict(features_scaled)[0])
            real_change_level = fileInfo['change_level_label']
            print("changeLevel: ",change_level ,": ",real_change_level)
            if sample_K!=0:
                if change_level ==0:
                    sample_K = 11
                elif change_level==1:
                    sample_K = 7
                elif change_level==2:
                    sample_K = 3

            demonstrations = f"There are {sample_K} examples for you, please learn from them.\n"
            # retriever
            if sample_K != 0:
                old_idx = faiss_index_dict[str(change_level)]['old_index']
                new_idx = faiss_index_dict[str(change_level)]['new_index']
                id_mapping = faiss_index_dict[str(change_level)]['id_mapping']
                data_mapping = faiss_index_dict[str(change_level)]['data_mapping']
                tmp_k = 20
                while (True):
                    query_result = query_structural_sim(
                        parser, old_idx, new_idx, id_mapping, data_mapping,
                        query_old_code=src_code,
                        query_new_code=dst_code,
                        top_k=tmp_k)
                    if (len(query_result) >= sample_K or tmp_k > 10000):
                        break
                    else:
                        tmp_k = tmp_k + 20
                query_result = get_list_topk(query_result, sample_K, "score")  # only_structural

                # Fusion-based Retrieval

                for item in query_result:
                    # diff_sim = diff_similarity(src_code,dst_code,item['src_method'],item['dst_method'])
                    diff_sim = diff_change_similarity(src_code,dst_code,item['src_method'],item['dst_method'])  # Distinguish add and sub
                    item["diff_similarity"] = diff_sim
                    change_type_sim = change_type_similarity(fileInfo["ChangeType"],item['change_type'])
                    item["change_type_similarity"] = change_type_sim
                    item['fused_score'] = diff_sim + change_type_sim

                query_result = get_list_topk(query_result, sample_K, "fused_score")


                for sample_index, sample in enumerate(query_result):
                    example_old_code = sample["src_method"]
                    example_new_code = sample["dst_method"]
                    example_old_comment = sample["src_desc"]
                    example_new_comment = sample["dst_desc"]

                    if code_template == 'diff':
                        diff = d.compare(example_old_code.splitlines(), example_new_code.splitlines())
                        diff_str = ""
                        for line in diff:
                            if line.startswith("?"):
                                continue
                            diff_str = diff_str + line + "\n"
                        demonstrations = demonstrations + f"# example {sample_index + 1}:\ndiff hunk:\n{diff_str}\nold comment: {example_old_comment}\n# Output:\nnew comment: {example_new_comment}\n\n"
                    else:
                        demonstrations = demonstrations + f"# example {sample_index + 1}:\nold code: {example_old_code}\nnew code: {example_new_code}\nold comment: {example_old_comment}\nnew comment: {example_new_comment}\n\n"

            # template
            if code_template == 'diff':
                instruction = f"Please update the comment precisely according to the old comment and the diff hunk. "
                if LLM_model == "DeepSeek-V3":
                    instruction += "Follow these rules: \n1. **Strict Alignment**: Only reflect what the diff hunk explicitly shows\n2. **Minimal Changes**: Preserve the original comment's structure and tone\n3. **No Additions**: Never add information not visible in the diff\n4. **Format Consistency**: Maintain identical formatting\n"
            else:
                instruction = f"Please write a new comment according to the old comment and the changes between the old and new code. "

            if sample_K != 0 and code_template != 'diff':
                input_content = instruction + demonstrations + "# The old comment and the changes between the old and new code:\nold code: " + src_code + "\nnew code: " + dst_code + "\nold comment: " + src_comments + "\nnew comment:"
            elif sample_K != 0 and code_template == 'diff':
                diff = d.compare(src_code.splitlines(), dst_code.splitlines())
                diff_str = ""
                for line in diff:
                    if line.startswith("?"):
                        continue
                    diff_str = diff_str + line + "\n"
                input_content = instruction + demonstrations + "# Input:\ndiff hunk:\n" + diff_str + "\nold comment: " + src_comments + "\n# Output:\nnew comment:"
            elif sample_K == 0 and code_template == 'diff':
                diff = d.compare(src_code.splitlines(), dst_code.splitlines())
                diff_str = ""
                for line in diff:
                    if line.startswith("?"):
                        continue
                    diff_str = diff_str + line + "\n"
                input_content = instruction  + "# Input:\ndiff hunk:\n" + diff_str + "\nold comment: " + src_comments + "\n# Output:\nnew comment:"

            # # save query
            # queries_list.append(input_content)
            # continue

            messages = [
                {"role": "system", "content": "You are an AI code comment updating assistant."},
                {
                    "role": "user",
                    "content": input_content
                }
            ]


            # LLM
            new_comments = []

            if client_method == 'local':
                print(input_content)
                llm_result = llm.generate(
                    input_content,
                    SamplingParams(
                        max_tokens=100,
                        temperature=0,
                        stop=["\n\n"]
                    )
                )
                new_comment = llm_result[0].outputs[0].text
                if new_comment == "":
                    new_comment = src_comments
                if new_comment.startswith(" "):
                    new_comment = new_comment.lstrip()
                new_comments.append(new_comment)

            else:
                if top_p == -1.0:
                    llm_result = client.chat.completions.create(messages=messages, model=LLM_model_name, n=K,
                                                                max_tokens=100, temperature=temperature, )
                else:
                    llm_result = client.chat.completions.create(messages=messages, model=LLM_model_name, n=K, top_p=top_p,
                                                                max_tokens=100, temperature=temperature)

                for k in range(0,K):
                    content = llm_result.choices[k].message.content
                    if content is None:
                        new_comment = src_comments
                        with open("log.txt", "a", encoding="utf-8") as f:
                            f.write(f"{LLM_model_name},{fileInfo['sample_id']},{fileInfo['index']}\n")
                    else:
                        new_comment = content.split('new comment:')[-1].strip()
                    if new_comment == "":
                        new_comment = src_comments
                    new_comments.append(new_comment)
                # print(f"content: {content}\n\n>>>old comment: {src_comments}\n>>>new comment:{new_comment}")
                # print(f">>>ref_comments: {ref_comments}\n")

            print(input_content)
            print("new_comments: ", new_comments)


            output_res ={
                'sample_id':fileInfo['sample_id'],
                'index':fileInfo['index'],
                'change_level':real_change_level,
                'predict_level':change_level,
                'src_comments':src_comments,
                'ref_comments':ref_comments,
                'new_comments':new_comments
            }
            # save results
            fout.write(json.dumps(output_res) + '\n')
            readableOut.write(json.dumps(output_res, indent=4) + '\n')

    # TMP: 存查询结果
    # with open("ours/info/our_diff_3samples.jsonl", 'w', encoding='utf-8') as f:
    #     for query in queries_list:
    #         f.write(json.dumps(query, ensure_ascii=False) + '\n')


def parse_args():
    parser = argparse.ArgumentParser(description='CupUpdater')
    parser.add_argument('--model', type=str, required=True, help='LLMmodel')
    parser.add_argument('--sample_k', type=int, default=-1, help='sample_K')
    parser.add_argument('--top_p', type=float,default=-1.0, help='top_p')
    parser.add_argument('--port',type=int,default=8001,help='port')
    return parser.parse_args()

'''

    当前版本的我们的方法

'''
if __name__ == "__main__":

    model_name = "../model/graphcodebert-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    roberta_model = RobertaModel.from_pretrained(model_name).eval().to("cuda" if torch.cuda.is_available() else "cpu")
    device = next(roberta_model.parameters()).device

    args = parse_args()
    LLM_model = args.model
    top_p = args.top_p
    sample_k = args.sample_k
    port = args.port
    client_method = "remote" # remote/local/fine-tuning
    LLM_model_name = LLM_model  # cmtUpdater
    if LLM_model_name == "DeepSeek-V3.1":
        LLM_model_name = "deepseek-chat" #"deepseek-ai/DeepSeek-V3"
    # if LLM_model_name == "DeepSeek-V3":
    #     # LLM_model_name = "deepseek-ai/DeepSeek-V3"  # "deepseek-ai/DeepSeek-V3"
    #     LLM_model_name = "deepseek-v3"

    temperature = 0.0
    code_template = 'diff'

    save_fold = "ours/output/resMain/"

    if code_template == 'diff':
        output_file = f"{LLM_model}_dfg_p{top_p}t{temperature}_{sample_k}samples.jsonl"
        readablePath = f"{save_fold}{LLM_model}_dfg_p{top_p}t{temperature}_{sample_k}samples.json"


# 不同的k
    update_comments("ours/dataset/test_cup_withFeatures.jsonl",ignore_case=True,K=1,sample_K=sample_k,top_p=top_p,outputPath=f'{save_fold}{output_file}')

