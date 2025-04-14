"""
Complex Code Change measurement

return: csv file(clustered info)


"""
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

from model.clusterMethod import get_size_features, get_structure_features, get_content_features


def readData(data_type,dims):
    if dims == "size":
        data = get_size_features(data_type)
    elif dims == 'structure':
        data = get_structure_features(data_type)
    elif dims == 'content':
        data = get_content_features(data_type)
    elif dims == 'all':
        size = get_size_features(data_type)
        structure = get_structure_features(data_type)
        content = get_content_features(data_type)
        data = pd.merge(size, structure, on=['sample_id', 'index'], how='left')
        data = pd.merge(data, content, on=['sample_id', 'index'], how='left')
    return data

def print_k_means_labels(y_kmeans):
    unique_labels, counts = np.unique(y_kmeans, return_counts=True)
    # 输出结果
    for label, count in zip(unique_labels, counts):
        print(f"Cluster {label}: {count} data points,")


def dataPartition(train,test,K,save_results=False,save_model=False):
    key_columns = ['sample_id','index']
    X_train = train.drop(columns=key_columns)
    X_test = test.drop(columns=key_columns)

    print(X_train['sim_score'])


    scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(X_train)

    kmeans = KMeans(n_clusters=K,n_init=10,random_state=42)
    kmeans.fit(train_data_scaled)

    if save_model:
        joblib.dump(scaler, f'../info_process/scaler_{dims}.pkl')
        joblib.dump(kmeans, f'../info_process/kmeans_model_{dims}.pkl')
        print("Scaler and KMeans model saved successfully")

    # 获取训练结果的聚类中心
    cluster_centers = kmeans.cluster_centers_
    print("聚类中心 (Cluster Centers):")
    print(cluster_centers)
    print("\norigin Cluster Centers:")
    print(scaler.inverse_transform(cluster_centers))

    # 获取训练集的簇标签
    train_labels = kmeans.labels_
    print("\n训练集簇标签 (Train Labels):")
    print_k_means_labels(train_labels)

    # 计算训练集的轮廓系数
    sil_score_train = silhouette_score(train_data_scaled, train_labels)
    print(f"\n训练集轮廓系数 (Silhouette Score - Train): {sil_score_train:.4f}")

    # 步骤 3: 对测试集进行相同的归一化
    test_data_scaled = scaler.transform(X_test)

    # 使用训练好的模型对测试集进行聚类划分
    test_labels = kmeans.predict(test_data_scaled)
    print("\n测试集簇标签 (Test Labels):")
    print_k_means_labels(test_labels)

    if save_results:
        test_data['label'] = test_labels
        test_data.to_csv(f"../info/clustered_data_test_{dims}.csv",index=False)

if __name__ == "__main__":
    dims = 'content'
    save_results = True
    save_model = True

    train_data = readData('train',dims)
    test_data = readData('test',dims)
    dataPartition(train_data,test_data,2,save_results=save_results,save_model=save_model)
