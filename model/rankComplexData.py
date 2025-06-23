import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from model.dataPartition import readData


def rank_data(data,dims):
    columns_to_normalize = data.columns.drop(['sample_id','index'])
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[columns_to_normalize])

    data['score'] = data_scaled.sum(axis=1)
    data = data.sort_values(by='score',ascending=False)
    data['rank']=data['score'].rank(ascending=False,method='min').astype(int)
    # data.to_csv(f"../info/ranked_data_test_{dims}.csv",index=False)
    return data

def rank_data_weightedDims(data,percent):
    size_columns = ['NMT', 'NMS', 'NML', 'NMC', 'NNTRP', 'NNSRP','NTOD', 'NSOD']
    structure_columns = ['CC_delta', 'delta_dependency']
    content_columns = ['sim_cocom', 'codeChangeNum', 'codeChangeScore', 'refactoringTypeNum']

    # 提取各个维度数据
    size = data[size_columns]
    structure = data[structure_columns]
    content = data[content_columns]

    scaler = MinMaxScaler()
    scaled_size = scaler.fit_transform(size)
    scaled_structure = scaler.fit_transform(structure)
    scaled_content = scaler.fit_transform(content)
    data['size_score'] = scaled_size.sum(axis=1)
    data['structure_score'] = scaled_structure.sum(axis=1)
    data['content_score'] = scaled_content.sum(axis=1)


    scaler = MinMaxScaler()
    score_columns= ['size_score','structure_score','content_score']
    scaled_score = scaler.fit_transform(data[score_columns])

    data['score'] = data['size_score']*1 + data['structure_score']*2+ data['content_score']*-1
    # data['score'] = scaled_score.sum(axis=1)
    data = data.sort_values(by='score', ascending=False)
    data['rank'] = data['score'].rank(ascending=False, method='min').astype(int)

    # 计算前30%和后30%的阈值
    total_rows = len(data)
    top_percent = int(np.ceil(total_rows * percent))
    bottom_percent = int(np.ceil(total_rows * percent))

    # 标注前30%为1，后30%为0
    data['label'] = np.where(data['rank'] <= top_percent, 1,
                           np.where(data['rank'] > (total_rows - bottom_percent), 0, np.nan))

    percent_num = percent*100
    data.to_csv(f"../info/ranked_data_weightedScoreTEST_{percent_num}percent.csv")



def get_top_ids(ranked_data,percent):
    top_percent = int(np.ceil(len(ranked_data)*percent))
    top_ids = ranked_data.nsmallest(top_percent,'rank')[['sample_id','index']]
    top_ids_set = set(top_ids.itertuples(index=False,name=None))

    return top_ids_set

def get_bottom_ids(ranked_data,percent):
    top_percent = int(np.ceil(len(ranked_data)*percent))
    top_ids = ranked_data.nlargest(top_percent,'rank')[['sample_id','index']]
    top_ids_set = set(top_ids.itertuples(index=False,name=None))
    return top_ids_set


def label_by_dims(data, percent):
    top_ids = get_top_ids(data,percent)
    bottom_ids = get_bottom_ids(data,percent)
    data['label'] = 1
    data['label'] = data.apply(lambda row :2 if(row['sample_id'],row['index']) in top_ids else row['label'], axis=1)
    data['label'] = data.apply(lambda row :0 if(row['sample_id'],row['index']) in bottom_ids else row['label'], axis=1)
    return data

def intersection_data(data,dataset,percent,dims_percent):

    size_columns = ['sample_id','index', 'NMT', 'NMS', 'NML', 'NMC', 'NNTRP', 'NNSRP','NTOD', 'NSOD']
    structure_columns = ['sample_id','index', 'CC_delta', 'delta_dependency']
    content_columns = ['sample_id','index', 'sim_cocom', 'codeChangeNum', 'codeChangeScore', 'refactoringTypeNum']

    # 提取各个维度数据
    test_data_size = data[size_columns]
    test_data_structure = data[structure_columns]
    test_data_content = data[content_columns]

    test_data_size = rank_data(test_data_size,'size')
    test_data_structure = rank_data(test_data_structure,'structure')
    test_data_content = rank_data(test_data_content,'content')

    # different Level (LOW MEDIUM HIGH 0,1,2）
    result_df = data
    result_df['label'] = 1


    # high
    ids_size = get_top_ids(test_data_size,percent=percent)
    ids_structure = get_top_ids(test_data_structure,percent=percent)
    ids_content = get_top_ids(test_data_content,percent=percent)
    common_pairs = ids_size.intersection(ids_structure, ids_content)
    result_df['label'] = result_df.apply(lambda row :2 if(row['sample_id'],row['index']) in common_pairs else row['label'], axis=1)

    # low
    ids_size = get_bottom_ids(test_data_size, percent=percent)
    ids_structure = get_bottom_ids(test_data_structure, percent=percent)
    ids_content = get_bottom_ids(test_data_content, percent=percent)
    common_pairs = ids_size.intersection(ids_structure, ids_content)
    result_df['label'] = result_df.apply(lambda row :0 if(row['sample_id'],row['index']) in common_pairs else row['label'], axis=1)

    # labelByDims
    test_data_size = label_by_dims(test_data_size, dims_percent)
    test_data_size = test_data_size[['sample_id', 'index', 'label', 'score']].rename(
        columns={'score': 'size_score', 'label': 'size_label'})
    result_df = result_df.merge(test_data_size,on=['sample_id','index'],how='left')
    test_data_structure = label_by_dims(test_data_structure, dims_percent)
    test_data_structure = test_data_structure[['sample_id', 'index', 'label', 'score']].rename(
        columns={'score': 'structure_score', 'label': 'structure_label'})
    result_df = result_df.merge(test_data_structure,on=['sample_id','index'],how='left')
    test_data_content = label_by_dims(test_data_content, dims_percent)
    test_data_content = test_data_content[['sample_id', 'index', 'label', 'score']].rename(
        columns={'score': 'content_score', 'label': 'content_label'})
    result_df = result_df.merge(test_data_content,on=['sample_id','index'],how='left')

    cols_to_int = ['label','size_label','structure_label','content_label']
    result_df[cols_to_int] = result_df[cols_to_int].astype(int)

    result_df.to_csv(f"../info/ranked_data_res_{dataset}.csv")

def rankDataByDims(data,dims,percent,top=True):
    data = rank_data(data,dims)
    result_df = labeledByRankAndPercent(data, percent,top)
    if top:
        top_name = "top"
    else: top_name = "bottom"
    result_df.to_csv(f"../info/ranked_data_{dims}_{top_name}_{percent*100}percent.csv")

def labeledByRankAndPercent(data, percent,top):
    if top:
        threshold = np.quantile(data['rank'], percent)
        data['label'] = np.where(data['rank']<=threshold,1,0)
    else:
        threshold = np.quantile(data['rank'],1-percent)
        data['label'] = np.where(data['rank']>=threshold,1,0)
    # # 计算前30%和后30%的阈值
    # total_rows = len(data)
    # top_percent = int(np.ceil(total_rows * percent))
    # bottom_percent = int(np.ceil(total_rows * percent))
    # # 标注前30%为1，后30%为0
    # data['label'] = np.where(data['rank'] <= top_percent, 1,
    #                          np.where(data['rank'] > (total_rows - bottom_percent), 0, np.nan))
    return data


if __name__ == "__main__":
    dims = 'all'
    save_results = True
    dataset = 'train'
    dims_percent = 0.2

    data = readData(dataset,dims)
    # data.to_csv("../info/codeChangeFeatures_test_2.csv") # save features info

    # rankDataByDims(data,dims,percent=0.2,top=True)
    intersection_data(data,dataset=dataset,percent=0.5,dims_percent=dims_percent)  # 三个维度50%的交集 + 各维度前n%
    # rank_data_weightedDims(data,percent=0.15)