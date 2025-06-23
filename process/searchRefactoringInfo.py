import json
import pandas as pd
from tqdm import tqdm

def main(datapath):
    refactoring_type_set = set()
    max_type_num =0
    with open(datapath,"r",encoding='utf-8') as f:
        for i, item in enumerate(tqdm(f.readlines())):
            item = json.loads(item)
            if len(item['refactoringType'])>max_type_num:
                max_type_num = len(item['refactoringType'])
            for changeType in item['refactoringType']:
                if changeType not in refactoring_type_set:
                    refactoring_type_set.add(changeType)
    print(refactoring_type_set)
    print(len(refactoring_type_set))
    print(f"单个method最大重构数：{max_type_num}")
if __name__ == "__main__":
    main("D:/ComplexCmtUpdater/ComplexCup-main/info/RefactoringType_train.jsonl")

