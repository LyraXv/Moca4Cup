import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter, FuncFormatter, LogLocator
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


def k_means(X,n_clusters):
    kmeans = KMeans(n_clusters=n_clusters,n_init=10,random_state=42)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)

    # 获取每个聚类标签及其对应的数量
    unique_labels, counts = np.unique(y_kmeans, return_counts=True)
    # 输出结果
    for label, count in zip(unique_labels, counts):
        print(f"Cluster {label}: {count} data points,")

    # 输出每个簇的簇心
    print("簇心：")
    centroids = kmeans.cluster_centers_
    for i, centroid in enumerate(centroids):
        print(f"簇 {i} 的簇心: {centroid}")

    if X.shape[1]==2:
        draw_cluster_figure(X, kmeans, y_kmeans)
    else:
        draw_pca_figure(X, kmeans, y_kmeans)

    return y_kmeans,centroids


def draw_cluster_figure(X, kmeans, y_kmeans):
    X = pd.DataFrame(X)
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5,marker='X')

    # 标注簇心的编号
    for i, centroid in enumerate(centers):
        plt.text(centroid[0] + 0.05, centroid[1] + 0.05, f'Cluster {i}', color='red', fontsize=12)
    # 在右上方显示每个簇的大小
    cluster_sizes = [np.sum(y_kmeans == i) for i in range(kmeans.n_clusters)]
    # 创建颜色映射，手动指定颜色
    # colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33FF']  # 指定每个簇的颜色
    for i, size in enumerate(cluster_sizes):
        plt.text(0.95, 0.9 - i * 0.05, f'Cluster {i}: {size} points', color='black', fontsize=12, ha='right', va='top', transform=plt.gca().transAxes)

    plt.title("K-means Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


def draw_pca_figure(X, kmeans, y_kmeans):
    # PCA降维到二维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # 绘制降维后的数据点和聚类中心
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    # 将聚类中心转换为PCA坐标
    centers_pca = pca.transform(centers)
    plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=200, alpha=0.5, marker='X')
    # 标注簇心的编号
    for i, centroid in enumerate(centers_pca):
        plt.text(centroid[0] + 0.05, centroid[1] + 0.05, f'Cluster {i}', color='red', fontsize=12)
    # 在右上方显示每个簇的大小
    cluster_sizes = [np.sum(y_kmeans == i) for i in range(kmeans.n_clusters)]
    # 创建颜色映射，手动指定颜色
    # colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33FF']  # 指定每个簇的颜色
    for i, size in enumerate(cluster_sizes):
        plt.text(0.95, 0.9 - i * 0.05, f'Cluster {i}: {size} points', color='black', fontsize=12, ha='right', va='top', transform=plt.gca().transAxes)

    plt.title("PCA Projection of K-means Clusters")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()


def get_size_features(data_type):
    file_path = f"../info/{data_type}_features_size_CC.csv"
    df = pd.read_csv(file_path)
    df = df.drop(columns=['CC_delta'])
    return df

def get_structure_features(data_type):
    cc_file_path = f"../info/{data_type}_features_size_CC.csv"
    df_cc = pd.read_csv(cc_file_path)
    df_cc = df_cc[['sample_id','index','CC_delta']]
    # df_cc['CC_delta'] = df_cc['CC_delta'].abs()

    dep_file_path = f"../info/datadependency_{data_type}.json"
    dep_data = read_independent_JSON_object(dep_file_path)

    df_dep = pd.DataFrame(dep_data)
    # print(df_dep)
    df_dep['sample_id'] = pd.to_numeric(df_dep['sample_id'])
    df_dep['index'] = pd.to_numeric(df_dep['index'])
    df_dep['delta_dependency'] = df_dep['delta_dependency'].abs()

    df = pd.merge(df_cc,df_dep[['sample_id','index','delta_dependency']],on=['sample_id','index'],how='left')
    # print(df)
    return df

def get_content_features(data_type):
    # code sim
    cs_file_path = f"../info/sim_score_{data_type}.csv"
    df_cs = pd.read_csv(cs_file_path)
    df_cs['codeChangeNum'] = 0
    df_cs['codeChangeScore'] = 0
    df_cs['refactoringTypeNum'] = 0
    df = df_cs

    # Number of Code Change Type/Score
    print("Begin to analyze number/score of code chage type.")
    change_type_path = f"../info/ChangeTypeRes_{data_type}.jsonl"
    with open(change_type_path,"r",encoding="utf-8") as f:
        for i, item in enumerate(tqdm(f.readlines())):
            item = json.loads(item)
            sample_id = int(item['sample_id'])
            index = int(item['index'])
            change_type_num, change_type_score = analyze_change_type(item['codeChangeType'])

            mask = (df['sample_id'] == sample_id) & (df['index'] == index)
            df.loc[mask,'codeChangeNum'] = change_type_num
            df.loc[mask,'codeChangeScore'] = change_type_score

    # Number of Code Refactoring Type
    print("Begin to analyze number of code refactoring type.")
    refactoring_type_path = f"../info/RefactoringType_{data_type}.jsonl"
    with open(refactoring_type_path,"r",encoding="utf-8") as f:
        for i, item in enumerate(tqdm(f.readlines())):
            item = json.loads(item)
            sample_id = int(item['sample_id'])
            index = int(item['index'])

            mask = (df['sample_id'] == sample_id) & (df['index'] == index)
            df.loc[mask,'refactoringTypeNum'] = len(item['refactoringType'])
    print(df.head())
    return df

def analyze_change_type(changeTypeList):
    score = 0
    num = len(changeTypeList)
    if not changeTypeList:
        return 0,0;
    for changeType in changeTypeList:
        level = changeType[1]
        if level == 'CRUCIAL':
            score += 4
        elif level == 'HIGH':
            score +=3
        elif level == 'MEDIUM':
            score += 2
        elif level == 'LOW':
            score += 1
        elif level == 'NONE':
            score += 0
            num -= 1
        else:
            print(level,changeType)
            print("Significant LEVEL maybe occur ERROR;")
    return num,score


def read_independent_JSON_object(filepath):
    with open(filepath, 'r') as file:
        content = file.read()

    # 每行是一个独立的 JSON 对象，逐行解析
    json_data = []
    for line in content.splitlines():
        try:
            json_data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Error decoding line: {line}")
            print(e)
    return json_data

def elbow_rule(X,feature_dim):
    # Calculate SSE for different k values (total error sum of squares)
    sse = []
    k_range = range(1, 26)  # range of K
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)  # inertia_ 为聚类的总误差平方和
    save_different_k_res('elbow_sse',sse,feature_dim)
    # Elbow rule figure
    plt.figure(figsize=(8, 6))
    plt.plot(k_range, sse, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.grid(True)
    plt.show()

def Sihouette_Score(X,features_dim,save_score=False):
    print("Sihouette_Score:")
    silhouette_scores = []
    K_range = range(2, 26)  # 测试2到25个簇
    for k in K_range:
        labels = KMeans(n_clusters=k,random_state=42).fit(X).labels_
        score = metrics.silhouette_score(X, labels)
        silhouette_scores.append(score)
        print(f"{k}: {score}")

    # save sihouette score
    if save_score:
        save_different_k_res("silhouette_scores",silhouette_scores,features_dim)
    # 绘制轮廓系数图
    plt.figure(figsize=(8, 6))
    plt.plot(K_range, silhouette_scores, marker='o', linestyle='--')
    plt.title("Silhouette Score for Different K Values")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.show()

    # 找到最佳K值
    best_k = K_range[np.argmax(silhouette_scores)]
    print(f"The best K value is {best_k}")


def save_different_k_res(k_type,score,dim):
    df_sihouette = pd.DataFrame(score, columns=[k_type])
    df_sihouette['dim'] = dim
    df_sihouette['k'] = range(1, len(df_sihouette) + 1)
    df_sihouette.to_csv(f"../info_process/{k_type}_train_{dim}.csv",index=False)


def main(data_type,K,feature_dims='single'):
    list_features_dim = ['size','structure','content']
    if feature_dims =='all':
        list_features_dim = ['all']
    df = pd.DataFrame()
    for features_dim in list_features_dim:
        print(f">>>features_dim: {features_dim}")
        if features_dim == "size":
            data = get_size_features(data_type)
        elif features_dim == 'structure':
            data = get_structure_features(data_type)
        elif features_dim == 'content':
            data = get_content_features(data_type)
        elif features_dim == 'all':
            size = get_size_features(data_type)
            structure = get_structure_features(data_type)
            content = get_content_features(data_type)
            data = pd.merge(size,structure,on=['sample_id','index'],how='left')
            data = pd.merge(data,content,on=['sample_id','index'],how='left')

        # save all features as csv/jsonl
        save_features = False
        if feature_dims == 'all' and save_features:
            data.to_csv(f'../info/allFeatures_{data_type}.csv',index=False)
            # data.to_json(f'../info/allFeatures_{data_type}.jsonl',orient='records',lines=True)

        X = data.drop(columns=['sample_id','index'])
        # X = X.drop(columns=['NMT','NMS','NML','NNTRP','codeChangeNum'])
        print("features: ",X.columns)

        # draw_Boxplot(X)

        # preprocess
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        elbow_rule(X,features_dim) # 肘部法则
        # Sihouette_Score(X,features_dim,save_score=True) # 轮廓系数
        continue

        Y_means,centroids = k_means(X,K)

        # centroid
        centroids = scaler.inverse_transform(centroids)
        print(centroids)

        #
        exit()

        # initial df
        if df.empty:
            df = data[['sample_id','index']]
        data_info = data[['sample_id','index']]
        data_info[features_dim] = Y_means

        df = pd.merge(df,data_info,on=['sample_id','index'])
        # print(df)

    # 定义自定义函数来进行条件判断并返回对应结果
    def check_row(row):
        values = row[list_features_dim]
        if (values == 1).all():
            return 1
        elif (values == 0).all():
            return 0
        return 2

    df['inter_set'] = df.apply(check_row, axis=1)
    # df.to_csv(f"../info/clusterResults_{data_type}.csv")
    print(df[df['inter_set']==1].shape)


def draw_Boxplot(data):
    plt.figure(figsize=(10, 6))
    import seaborn as sns
    sns.boxplot(data=data)
    plt.title('Boxplot of Size, Length, Similarity, and Complexity')
    plt.xlabel('Features')
    plt.ylabel('Values')
    plt.yscale('log')
    plt.show()


if __name__ == "__main__":
    data_type = "train"
    k = 2
    feature_dims = 'single' #all/single
    main(data_type,k,feature_dims)