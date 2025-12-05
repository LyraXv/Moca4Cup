import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, make_scorer

# ========= 数据加载 ========= #
# 替换成你自己的路径
from ours.mlClassifiers import read_origin_data

train_x, train_y = read_origin_data("../info/ranked_data_res_train.csv")
test_x, test_y = read_origin_data("../info/ranked_data_res_test.csv")
valid_x, valid_y = read_origin_data("../info/ranked_data_res_valid.csv")

X_trainval = np.vstack([train_x, valid_x])
y_trainval = np.concatenate([train_y, valid_y])
X_test = test_x.values
y_test = test_y.values

# ========= 数据归一化 ========= #
scaler = StandardScaler()
X_trainval_scaled = scaler.fit_transform(X_trainval)
X_test_scaled = scaler.transform(X_test)

# ========= 评估指标 ========= #
scoring = {
    'accuracy': 'accuracy',
    'precision_macro': 'precision_macro',
    'recall_macro': 'recall_macro',
    'f1_macro': 'f1_macro',
    'roc_auc_ovr': make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr', average='macro')
}
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# ========= 模型与参数 ========= #
models = {
    # "Naive Bayes": (GaussianNB(), {
    #     'var_smoothing': [0, 1e-9, 1e-7, 1e-5, 1e-3]
    # }),
    # "Random Forest": (RandomForestClassifier(random_state=42), {
    #     'max_depth': [7, 9, 11, 13, 15],
    #     'n_estimators': [50, 100, 150, 200, 250]
    # }),
    # "Decision Tree": (DecisionTreeClassifier(random_state=42), {
    #     'max_depth': [7, 9, 11, 13, 15]
    # }),
    # "LightGBM": (LGBMClassifier(random_state=42), {
    #     'max_depth': [7, 9, 11, 13, 15],
    #     'n_estimators': [50, 100, 150, 200, 250],
    #     'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
    #     'verbose':[0]
    # }),
    'SVM': (SVC(), [{'kernel': ['linear'], 'C': [2 ** x for x in range(-10, 11)]},
                    {'kernel': ['rbf', 'sigmoid'], 'C': [2 ** x for x in range(-10, 11)],
                     'gamma': [2 ** x for x in range(-10, 11)] + ['scale']},
                    {'kernel': ['poly'], 'C': [2 ** x for x in range(-10, 11)],
                     'gamma': [2 ** x for x in range(-10, 11)] + ['scale'], 'degree': [2, 3, 4, 5]}]),
    'Logistic Regression': (LogisticRegression(), {'C': [x for x in range(1, 11)]}),

}

# ========= 模型训练与评估 ========= #
results = {}
best_models = {}

for name, (model, params) in models.items():
    print(f"\nTraining {name}...")
    grid = GridSearchCV(model, params, cv=cv, scoring='f1_macro', n_jobs=-1)
    grid.fit(X_trainval_scaled, y_trainval)
    best_model = grid.best_estimator_
    best_models[name] = best_model

    # 保存每组参数结果
    cv_results_df = pd.DataFrame(grid.cv_results_)
    cv_results_df.to_csv(f"{name}_grid_search_results.csv", index=False)
    print(f"Saved parameter results to {name}_grid_search_results.csv")

    # 交叉验证评估
    # cv_result = cross_validate(best_model, X_trainval_scaled, y_trainval, cv=cv, scoring=scoring)
    # results[name] = {k: np.mean(v) for k, v in cv_result.items()}

    # 测试集评估
    y_pred = best_model.predict(X_test_scaled)
    y_proba = best_model.predict_proba(X_test_scaled)

    results[name] = {
        "best_params": grid.best_params_,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "test_auc_macro": roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    }

# ========= 输出结果 ========= #
summary = pd.DataFrame({
    model: {
        'Accuracy (Test)': results[model]['classification_report']['accuracy'],
        'Precision_macro (Test)': results[model]['classification_report']['macro avg']['precision'],
        'Recall_macro (Test)': results[model]['classification_report']['macro avg']['recall'],
        'F1_macro (Test)': results[model]['classification_report']['macro avg']['f1-score'],
        'AUC_macro (Test)': results[model]['test_auc_macro']
    } for model in results
}).T
summary.to_csv("ML_classifiers_best_results_SVM_LR.csv") #存储最佳结果
print("\n=== Test Set Macro-Average Summary ===")
print(summary)