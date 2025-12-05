import json
import os.path
import re
from typing import Iterable, List

import nltk
import pandas as pd
from tqdm import tqdm

from eval.compute_gleu_cup import calcGleu
from eval.new_match import NLGMetrics, EditDistance
from eval.SARI import SARIsent

stop_words = {}
connectOp = {'.', '<con>'}
symbol = {"{", "}", ":", ",", "_", ".", "-", "+", ";", "<con>"}

# class EditDistance():
#     def __init__(self, *args, **kwargs):
#         super(EditDistance, self).__init__()
#
#     @staticmethod
#     def edit_distance(sent1: List[str], sent2: List[str]) -> int:
#         return word_level_edit_distance(sent1, sent2)
#
#     def eval(self, hypos: Iterable[List[List[str]]], references: Iterable[List[str]],
#              src_references: Iterable[List[str]], *args, **kwargs) -> dict:
#         src_distances = []
#         hypo_distances = []
#         for idx, (hypo_list, ref, src_ref) in enumerate(zip(hypos, references, src_references)):
#             hypo = hypo_list[0]
#             hypo_ref_dis = self.edit_distance(hypo, ref)
#             src_ref_dis = self.edit_distance(src_ref, ref)
#             src_distances.append(src_ref_dis)
#             hypo_distances.append(hypo_ref_dis)
#             # rel_distances.append(self.relative_distance(src_ref_dis, hypo_ref_dis))
#         # rel_dis = float(np.mean(rel_distances))
#         src_dis = float(np.mean(src_distances))
#         hypo_dis = float(np.mean(hypo_distances))
#         rel_dis = float(hypo_dis / src_dis)
#         # return float(np.mean(distances))
#         return {"rel_distance": rel_dis, "hypo_distance": hypo_dis, "src_distance": src_dis}

def word_level_edit_distance(a: List[str], b: List[str]) -> int:
    max_dis = max(len(a), len(b))
    distances = [[max_dis for j in range(len(b)+1)] for i in range(len(a)+1)]
    for i in range(len(a)+1):
        distances[i][0] = i
    for j in range(len(b)+1):
        distances[0][j] = j

    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            distances[i][j] = min(distances[i-1][j] + 1,
                                  distances[i][j-1] + 1,
                                  distances[i-1][j-1] + cost)
    return distances[-1][-1]

def getTokenStream(fileInfo):
    """
    Extract infomation stream from preprocessed data file.
    :param fileInfo: Preprocessed data of single file
    :return: old code token stream, new code token stream, old comment token stream, new comment token stream, changed token.
    """
    if "code_change_seq" not in fileInfo:
        return False
    codeSeq = fileInfo["code_change_seq"]
    buggyStream = []
    fixedStream = []
    changed = set()
    for x in codeSeq:
        buggyStream.append(x[0])
        fixedStream.append(x[1])
        if x[2] != "equal":
            changed.add(x[0].lower()) if x[0] != '' and x[0] != '<con>' and x[0].isalpha() and x[
                0] not in stop_words else None
            changed.add(x[1].lower()) if x[1] != '' and x[1] != '<con>' and x[1].isalpha() and x[
                1] not in stop_words else None
    buggyStream = [x.lower() for x in buggyStream if x != '' and x != '<con>' and x not in stop_words]
    fixedStream = [x.lower() for x in fixedStream if x != '' and x != '<con>' and x not in stop_words]
    oldComment = [x for x in fileInfo["src_desc_tokens"] if x != '']
    newComment = [x for x in fileInfo["dst_desc_tokens"] if x != '']
    return buggyStream, fixedStream, oldComment, newComment, changed

# tokenize_comments(main)
def tokenize_string_literal(str_literal, with_con=False):
    """
    str_literal: str, STRING_LITERAL.text
    return: list of tokens
    """
    if with_con:
        str_tokens = tokenize_text_with_con(str_literal[1:-1])
    else:
        str_tokens = tokenize_text(str_literal[1:-1])
    return str_tokens

def tokenize_text_with_con(text):
    def _tokenize_word(word):
        new_word = re.sub(r'([-!"#$%&\'()*+,./:;<=>?@\[\\\]^`{|}~])', r' \1 ', word)
        subwords = nltk.word_tokenize(new_word)
        new_subwords = []
        for w in subwords:
            new_subwords += tokenize_identifier_raw(w, keep_underscore=True)
        return new_subwords

    tokens = []
    text = text or ""
    for word in text.split():
        if not word:
            continue
        tokens += " <con> ".join(_tokenize_word(word)).split()
    return tokens

def tokenize_identifier_raw(token, keep_underscore=True):
    regex = r'(_+)' if keep_underscore else r'_+'
    id_tokens = []
    for t in re.split(regex, token):
        if t:
            id_tokens += camel_case_split(t)
    # note: do not use lowercase!
    return list(filter(lambda x: len(x) > 0, id_tokens))

def tokenize_identifier(token, with_con=False):
    if with_con:
        id_tokens = " <con> ".join(tokenize_identifier_raw(token, keep_underscore=True)).split()
    else:
        id_tokens = [t.lower() for t in tokenize_identifier_raw(token, keep_underscore=False)]
    return id_tokens

def tokenize_text(text):
    str_tokens = []
    nltk_tokenized = " ".join(nltk.word_tokenize(text))
    # split according to punctuations
    # string.punctuation
    # NOTE: do not care _, which will be taken care of by tokenized_identifier
    content_tokens = re.sub(r'([-!"#$%&\'()*+,./:;<=>?@\[\\\]^`{|}~])', r' \1 ', nltk_tokenized).split()
    for t in content_tokens:
        str_tokens += tokenize_identifier(t)
    return str_tokens

def camel_case_split(identifier):
    return re.sub(r'([A-Z][a-z])', r' \1', re.sub(r'([A-Z]+)', r' \1', identifier)).strip().split()


def comments_acc_and_recall(ref_comments, new_comments, K ,ignore_case):
    # print(f">>>Method: comments_acc_and_recall,K:{K}")
    # print(ref_comments)
    # print(new_comments)

    correct = comments_equal(ref_comments,new_comments[0],ignore_case)
    if correct:
        return 1,1

    # for k in range(1,K):
    #     # print("acc_recall_current_k:",k)
    #     correct_k = comments_equal(ref_comments,new_comments[k],ignore_case)
    #     if correct_k:
    #         return 0,1
    return 0,0

def comments_recall(ref_comments, new_comments, K, ignore_case):
    for k in range(0, K):
        correct_k = comments_equal(ref_comments, new_comments[k], ignore_case)
        if correct_k:
            return 1
    return 0

def comments_acc(ref_comments, new_comment, ignore_case):
    correct = comments_equal(ref_comments, new_comment, ignore_case)
    if correct:
        return 1
    else:
        return 0

def Equal_1(pred, oracle):
    predStr = "".join(pred).replace("<con>", '')
    oracleStr = "".join(oracle).replace("<con>", '')
    if predStr.lower() == oracleStr.lower():
        return True
    else:
        return False

# 考虑是否忽略最后一个标点？
def comments_equal(str1, str2, ignore_case=True):
    # str1:ref str2:pred
    str1_tokenized = tokenize_text_with_con(str1)
    str2_tokenized = tokenize_text_with_con(str2)
    str1_processed = "".join(str1_tokenized).replace("<con>", '')
    str2_processed = "".join(str2_tokenized).replace("<con>", '')
    if ignore_case:
        str1_processed = str1_processed.lower()
        str2_processed = str2_processed.lower()

    # print("comments_equal:",str1_processed,"    ",str2_processed)
    return str1_processed == str2_processed

def recover_desc(sent: Iterable[str]) -> str:
    return re.sub(r' <con> ', "", " ".join(sent))

def prepare_sent(tokens: List[str]) -> str:
    return recover_desc(tokens)

def cal_sari(pred_comments_tokenized_list, ref_comments_tokenized_list, src_comments_tokenized_list):
    hypos = [prepare_sent(hypo_list) for hypo_list in pred_comments_tokenized_list]
    ref = [prepare_sent(ref_list) for ref_list in ref_comments_tokenized_list]
    src = [prepare_sent(src_list) for src_list in src_comments_tokenized_list]

    score = []
    for l in range(0,len(hypos)):
        target = [ref[l]]
        # print(src[l])
        # print(hypos[l])
        # print(target)
        output = SARIsent(src[l],hypos[l],target)
        # print(output)
        # print("....")
        score.append(output)

    sari_score = sum(score)/len((score))
    print(f"sum:{sum(score)},len:{len(score)},sari:{sari_score}")
    return sari_score

def eval_cup(changeLevel, ignore_case, K,top_p,temperature, llm_res,output_path,output_eval_res=False):
    src_comments_tokenized_list = []
    ref_comments_tokenized_list = []
    pred_comments_tokenized_list = []
    accuracy_list = []
    recall_list = []
    total_accuracy_per_k = [0] * K

    change_level_list = []
    all_items = []

    id_index_to_record = {}
    with open('../dataset/test_cup_withFeaturesandLabel.jsonl', 'r', encoding='utf8') as f_b:
    # with open('../dataset/valid_cup_withFeatures_600.jsonl', 'r', encoding='utf8') as f_b:  # paraK

        for line in f_b:
            data = json.loads(line)
            id_b = int(data['sample_id'])
            index_b = int(data['index'])
            id_index_to_record[(id_b, index_b)] = data

    accuracy_key_list = []
    with open(llm_res, 'r', encoding='utf8') as f:
        for i,line in enumerate(tqdm(f.readlines())):
            # if i==1000:
            #     break

            # print(line)
            data = json.loads(line)
            sample_id= int(data['sample_id'])
            index = int(data['index'])
            change_level = int(data['change_level'])
            new_comments = data['new_comments']

            # if change_level != changeLevel:
            #     continue

            key = (sample_id, index)
            fileInfo = id_index_to_record[key]
            if fileInfo['change_level_label'] != changeLevel and changeLevel != -1:
                continue
            ref_comments = fileInfo['dst_desc']

            pred_desc = tokenize_text_with_con(new_comments[0])  # 分词

            # print("pred_desc_tokenized: ", pred_desc)
            buggyStream, fixedStream, src_desc, dst_desc, changed = getTokenStream(fileInfo)  # 读取分词
            # print("src_desc", src_desc)

            # 以下其实都是用来计算结果的？所以其实可以写成eval()?
            if ignore_case:
                ref_comments_tokenized_list.append(" ".join(dst_desc).lower().split())
                src_comments_tokenized_list.append(" ".join(src_desc).lower().split())
                pred_comments_tokenized_list.append(" ".join(pred_desc).lower().split())
            else:
                ref_comments_tokenized_list.append(" ".join(dst_desc).split())
                src_comments_tokenized_list.append(" ".join(src_desc).split())
                pred_comments_tokenized_list.append(" ".join(pred_desc).split())
            # print("src:", src_comments_tokenized_list, "\nref:", ref_comments_tokenized_list, "\npred:",
            #       pred_comments_tokenized_list, "\nchange:", change_level_list)
            # all_items.append(fileInfo)


            accuracy, recall = comments_acc_and_recall(ref_comments, new_comments, K, ignore_case)
            accuracy_list.append(accuracy)
            recall_list.append(recall)

            # if accuracy ==1:
            #     accuracy_key_list.append(key)


            # print("accuracy: ", accuracy, "recall:", recall)

            # for res_k in range(0,K):
            #     total_accuracy_per_k[res_k] += comments_acc(ref_comments, new_comments[res_k], ignore_case)
            #
            # recall = comments_recall(ref_comments, new_comments, K, ignore_case)
            # recall_list.append(recall)
        # print(total_accuracy_per_k)
        # with open("gemini_bm25_noLevel_list.json", "w", encoding="utf-8") as f:
        #     json.dump(accuracy_key_list, f, ensure_ascii=False, indent=2)
        # exit()
        # # Metrics
        accuracy = sum(accuracy_list)/len(recall_list)
        recall = sum(recall_list)/len(recall_list)
        print(f"acc: {sum(accuracy_list)}, len: {len(accuracy_list)}")
        print(f"recall: {sum(recall_list)}, len: {len(recall_list)}")

        res_editDist = EditDistance().eval(pred_comments_tokenized_list, ref_comments_tokenized_list,ref_comments_tokenized_list,src_comments_tokenized_list)
        # print(sample_id,pred_comments_tokenized_list)
        res_nlg = NLGMetrics().eval(pred_comments_tokenized_list, ref_comments_tokenized_list)
        sari = cal_sari(pred_comments_tokenized_list, ref_comments_tokenized_list, src_comments_tokenized_list)
        gleu = calcGleu(src_comments_tokenized_list, ref_comments_tokenized_list, pred_comments_tokenized_list,lowercase=True)


        # if temperature==-1:
        #     temperature = "default"
        if top_p == -1:
            top_p = "default"
        # temperature = "default"
        # top_p = "default"

        if output_eval_res:
            eval_res = {
                'ChangeLevel': changeLevel,
                'Dims': 'all',
                # 'Round': round_count, # paraK
                'Method': LLM_model,
                'top_p': top_p,
                'temperature': temperature,
                'sample_k': sample_k,
                'Accuracy': f"{(accuracy * 100):.2f}%",
                'correct_count_1': f"({sum(accuracy_list)}/{len(accuracy_list)})",
                'Recall_5': f"{(recall * 100):.2f}%",
                'correct_count_5': f"({sum(recall_list)}/{len(recall_list)})",
                'ESS_ratio': f"{(res_editDist['ESS Ratio'] * 100):.2f}%",
                'SARI': f"{(sari * 100):.2f}%",
                "AED": round(res_editDist['hypo_distance'], 3),
                "RED": round(res_editDist['rel_distance'], 3),
                'GLEU': f"{(gleu * 100):.2f}",
                "Bleu_4": f"{(res_nlg['Bleu_4'] * 100):.2f}",
                "METEOR": f"{(res_nlg['METEOR'] * 100):.2f}",
                "ROUGE_L":f"{(res_nlg['ROUGE_L'] * 100):.2f}",
            }
            if not os.path.exists(output_path):
                new_df = pd.DataFrame([eval_res])
                new_df.to_csv(output_path,index=False)
            else:
                df_res = pd.read_csv(output_path)
                new_df = pd.DataFrame([eval_res])
                df_res = pd.concat([df_res, new_df], ignore_index=True)
                df_res.to_csv(output_path, index=False)
            print(f"The results have saved in {output_path}.")


if __name__ == "__main__":
    # 读取test文件，逐条分析，分类器分类，判断结果，（分级查询），生产prompt，调用api,结果存储，输出
    folder="ablation"
    LLM_model = "gemini-2.5-flash-nothinking_dfg_woStruct"
    sample_k = -1
    top_p = -1.0
    temperature = 0.0
    save_file = "../eval_res/res_11.csv"


# ParaK
#     sample_k_list =[0,1,3,5,7,9,11,13,15]
#     for sample_k in sample_k_list:
#         for round_count in [5]:
#             for level in [0,1,2]:
#                     llm_res = f"../result/{folder}{round_count}/{LLM_model}_p{top_p}t{temperature}_{sample_k}samples.jsonl"
#                     eval_cup(changeLevel=level, ignore_case=True, K=1, top_p=top_p, temperature=temperature, llm_res=llm_res,
#                              output_path=save_file, output_eval_res=True, )

    # # 分等级评估结果
    llm_res = f"../result/{folder}/{LLM_model}_p{top_p}t{temperature}_{sample_k}samples.jsonl"
    for level in range(0,3):
        eval_cup(changeLevel=level, ignore_case=True, K=1, top_p=top_p, temperature=temperature, llm_res=llm_res,
                 output_path=save_file, output_eval_res=True, )

    # 全部结果
    eval_cup(changeLevel=-1, ignore_case=True, K=1, top_p=top_p, temperature=temperature, llm_res=llm_res,
             output_path=save_file, output_eval_res=True, )
    #
    # for fused_weight in [0.9]:
    #     for level in range(0,3):
    #         print("change_level:",level)
    #         # llm_res = f"../result/resMain/{LLM_model}_p{top_p}t{temperature}_{sample_k}samples.jsonl"
    #         llm_res = f"../result/resWeighted/gemini-2.5-flash-nothinking_processed_dfg_{fused_weight}diffW.jsonl"
    #         eval_cup(changeLevel=level,ignore_case=True,K=1,top_p=top_p,temperature=temperature,llm_res=llm_res,output_path = "../eval_res/tmp_res.csv",output_eval_res=True,)
