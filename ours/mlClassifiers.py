import joblib
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix

import config


def read_origin_data(datapath):
    data = pd.read_csv(datapath)
    data_columns = config.ALL_COLUMNS
    x_data = data[data_columns]
    y_data = data['label']
    return x_data,y_data

def train(train_x_scaled,train_y,test_x_scaled,test_y):
    models = {
        # 'NaiveBayes': GaussianNB(var_smoothing=0.001),
        # 'RandomForest': RandomForestClassifier(max_depth=15, n_estimators=150,random_state=42),
        # 'DecisionTree': DecisionTreeClassifier(max_depth=15,random_state=42),
        'LightGBM': LGBMClassifier(learning_rate=0.1, max_depth=15, n_estimators=200, random_state=42),
        # 'Logistic Regression': LogisticRegression(C=4),
        # 'SVM':SVC(C= 16, gamma=2**-10, kernel= 'sigmoid',probability=True)

    }
    # 多分类 one-hot 编码（用于 AUC）
    classes = [0, 1, 2]
    test_y_binarized = label_binarize(test_y, classes=classes)

    all_results = []

    for name, model in models.items():
        model.fit(train_x_scaled, train_y)
        y_pred = model.predict(test_x_scaled)
        y_proba = model.predict_proba(test_x_scaled)

        # save model
        # if name == 'LightGBM':
        #     joblib.dump(model,'../model/lightGBM.pkl')

        acc = accuracy_score(test_y, y_pred)
        precision = precision_score(test_y, y_pred, average=None, labels=classes)
        recall = recall_score(test_y, y_pred, average=None, labels=classes)
        f1 = f1_score(test_y, y_pred, average=None, labels=classes)
        auc = roc_auc_score(test_y_binarized, y_proba, average=None, multi_class='ovr')

        # 宏平均
        macro_precision = precision.mean()
        macro_recall = recall.mean()
        macro_f1 = f1.mean()
        macro_auc = auc.mean()

        print(f"📊 {name} Results:")
        for i, cls in enumerate(classes):
            print(
                f"  Class {cls}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}, AUC={auc[i]:.4f}")
            all_results.append({
                "Model": name,
                "Class": cls,
                "Metric": "Per-Class",
                "Precision": round(precision[i]*100, 1),
                "Recall": round(recall[i]*100, 1),
                "F1": round(f1[i]*100, 1),
                "AUC": round(auc[i]*100, 1),
                "Accuracy": None
            })
        print(
            f"  ➕ Macro-Average: Precision={macro_precision:.4f}, Recall={macro_recall:.4f}, F1={macro_f1:.4f}, AUC={macro_auc:.4f}")
        print(f"  ✅ Accuracy: {acc:.4f}")
        all_results.append({
            "Model": name,
            "Class": "Macro-Average",
            "Metric": "Macro",
            "Precision": round(macro_precision*100, 1),
            "Recall": round(macro_recall*100, 1),
            "F1": round(macro_f1*100, 1),
            "AUC": round(macro_auc*100, 1),
            "Accuracy": round(acc*100, 1)
        })
        print("-" * 60)

        results_df = pd.DataFrame(all_results)
        results_df.to_csv("../info/mlClassifiers_metrics_results_svm2.csv", index=False)

def train_noLabel(train_x_scaled,train_y,test_x_scaled,test_y):
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(train_x_scaled)
    y_test_cluster = kmeans.predict(test_x_scaled)
    # 第一步：获得训练集聚类标签
    y_train_cluster = kmeans.predict(train_x_scaled)
    # 第二步：创建聚类标签与真实标签的映射
    y_test_pred = map_clusters_to_labels(train_y, y_test_cluster,y_train_cluster)
    print(y_test_pred)

    # 评估
    acc = accuracy_score(test_y, y_test_pred)
    precision = precision_score(test_y, y_test_pred, average=None)
    recall = recall_score(test_y, y_test_pred, average=None)
    f1 = f1_score(test_y, y_test_pred, average=None)
    y_pred_bin = label_binarize(test_y, classes=[0, 1, 2])
    y_test_bin = label_binarize(test_y, classes=[0, 1, 2])
    auc = roc_auc_score(y_test_bin, y_pred_bin, average=None, multi_class='ovr')

    print(f"📊 KMeans (unsupervised) Results:")
    for i, cls in enumerate([0, 1, 2]):
        print(f"  Class {cls}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}, AUC={auc[i]:.4f}")
    print(
        f"  ➕ Macro-Average: Precision={precision.mean():.4f}, Recall={recall.mean():.4f}, F1={f1.mean():.4f}, AUC={auc.mean():.4f}")
    print(f"  ✅ Accuracy: {acc:.4f}")
    print("-" * 60)

# 由于无监督的预测标签不具备真实类别0，1，2对应的顺序关系
# 通过将y_train和y_train_cluster建立映射关系，得到y_test
def map_clusters_to_labels(y_train, y_cluster,y_train_cluster):
    label_map = {}
    for cluster in np.unique(y_cluster):
        mask = y_train_cluster == cluster
        # 根据训练集中的聚类结果找到主标签
        most_common = mode(y_train[mask], keepdims=True).mode[0]
        label_map[cluster] = most_common
    return np.vectorize(label_map.get)(y_cluster)

if __name__ == "__main__":

    train_x,train_y = read_origin_data("../info/ranked_data_res_train.csv")
    test_x,test_y = read_origin_data("../info/ranked_data_res_test.csv")
    
    #归一化
    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    test_x_scaled = scaler.transform(test_x)
    joblib.dump(scaler, '../model/scaler')  # 保存归一化模型

    train(train_x_scaled,train_y,test_x_scaled,test_y)
    # train_noLabel(train_x_scaled,train_y,test_x_scaled,test_y)
    


