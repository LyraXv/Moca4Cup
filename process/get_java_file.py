import json
import os

from tqdm import tqdm

def compose_into_java_file(method,commit_type,data_type,item_id):
    class_content = "public class test{\n" + method + "\n}"
    output = f"../../java_file/cup_{data_type}_java_file/{commit_type}_file/"
    file_name = f"{item_id}.java"
    file_path = output +file_name

    if os.path.exists(file_path):
        print(item_id,"type:",commit_type,"may be depulicated")

    # 将内容写入Java文件
    with open(file_path, 'w',encoding='utf-8') as file:
        file.write(class_content)


def main(dataPath, data_type): # data_type(train/valid/test)
    with open(dataPath,"r",encoding="utf-8") as f:
        for i, item in enumerate(tqdm(f.readlines())):
            item = json.loads(item)
            item_id = item['sample_id']
            index = item['index']
            if index !=0:
                item_name = str(item_id) +"_"+str(index)
            else:
                item_name = item_id
            # method content
            src_method = item['src_method']
            dst_method = item['dst_method']
            # save as java file
            compose_into_java_file(src_method,"src",data_type,item_name)
            compose_into_java_file(dst_method,"dst",data_type,item_name)



if __name__ == "__main__":
    # 获取特征数据
    # main("../dataset/train_clean.jsonl","train") # train
    # main("../dataset/valid_clean.jsonl","valid") # valid
    main("../dataset/test_clean.jsonl","test") # test