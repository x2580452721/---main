import sys
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, silhouette_score

# 获取项目根目录（假设 app.py 在 backend 文件夹下）
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import algorithms as algos
from datasets.iris import load_iris
from datasets.mnist_sample import load_mnist_sample
from datasets.regression_sample import load_regression_sample

# 初始化Flask应用
app = Flask(__name__)
CORS(app)

# 全局存储数据集和分割结果
datasets = {
    'iris': None,
    'mnist': None,
    'regression': None
}
# 存储最近一次分割结果（按 dataset_id）
dataset_splits = {
    # example: 'iris': { 'X_train':..., 'X_test':..., 'y_train':..., 'y_test':..., 'params': {...} }
}

def convert_numpy_types(obj):
    """递归将 NumPy 类型转换为 Python 原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            new_k = int(k) if isinstance(k, np.integer) else k
            new_dict[new_k] = convert_numpy_types(v)
        return new_dict
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def load_all_datasets():
    """加载所有数据集到内存"""
    datasets['iris'] = load_iris()            # (X, y, desc)
    datasets['mnist'] = load_mnist_sample()
    datasets['regression'] = load_regression_sample()

def calculate_metrics(y_true, y_pred, task_type='classification'):
    metrics = {}
    # 统一转换为 numpy 数组并一维化
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # 确保是数值（用于回归误差）
    # 对分类可保留整数标签比较
    if y_true.dtype.kind in 'if':
        y_true = y_true.astype(np.float64)
    if y_pred.dtype.kind in 'if':
        y_pred = y_pred.astype(np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"真实值和预测值形状不匹配: {y_true.shape} vs {y_pred.shape}")

    # MSE（对分类/回归都计算一个基线 MSE）
    try:
        mse = np.mean((y_true - y_pred) ** 2)
    except Exception:
        mse = None
    metrics['mse'] = float(mse) if mse is not None else None

    if task_type == 'classification':
        # 准确率（按标签完全匹配）
        try:
            accuracy = float(np.mean(y_true == y_pred))
        except Exception:
            accuracy = None
        metrics['accuracy'] = accuracy

        # Macro averaged precision/recall/f1
        try:
            classes = np.unique(y_true)
            precision_list, recall_list, f1_list = [], [], []
            for cls in classes:
                tp = np.sum((y_true == cls) & (y_pred == cls))
                fp = np.sum((y_true != cls) & (y_pred == cls))
                fn = np.sum((y_true == cls) & (y_pred != cls))
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)
            metrics['precision'] = float(np.mean(precision_list)) if precision_list else None
            metrics['recall'] = float(np.mean(recall_list)) if recall_list else None
            metrics['f1'] = float(np.mean(f1_list)) if f1_list else None
        except Exception:
            metrics['precision'] = metrics['recall'] = metrics['f1'] = None
    else:
        # regression specific metrics
        try:
            rmse = np.sqrt(mse) if mse is not None else None
            mae = float(np.mean(np.abs(y_true - y_pred)))
        except Exception:
            rmse = None
            mae = None
        metrics['rmse'] = float(rmse) if rmse is not None else None
        metrics['mae'] = float(mae) if mae is not None else None
        # 分类指标不可用
        metrics['accuracy'] = None
        metrics['precision'] = None
        metrics['recall'] = None
        metrics['f1'] = None

    return metrics

# ---------- 原有接口保持不变 / 小修复 ----------

@app.route('/api/algorithms', methods=['GET'])
def get_algorithms():
    algorithms = [
        {'id': 'decision_tree', 'name': '决策树', 'task_type': 'classification'},
        {'id': 'naive_bayes', 'name': '朴素贝叶斯', 'task_type': 'classification'},
        {'id': 'knn', 'name': 'K最近邻', 'task_type': 'both'},
        {'id': 'svm', 'name': '支持向量机', 'task_type': 'classification'},
        {'id': 'random_forest', 'name': '随机森林', 'task_type': 'classification'},
        {'id': 'linear_regression', 'name': '线性回归', 'task_type': 'regression'},
        {'id': 'logistic_regression', 'name': '逻辑回归', 'task_type': 'classification'},
        {'id': 'adaboost', 'name': 'AdaBoost', 'task_type': 'classification'},
        {'id': 'kmeans', 'name': 'K均值聚类', 'task_type': 'clustering'},
        {'id': 'em', 'name': 'EM算法', 'task_type': 'clustering'}
    ]
    return jsonify({'algorithms': algorithms})

@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    datasets_info = []
    for ds_id in ['iris', 'mnist', 'regression']:
        if datasets.get(ds_id) is None:
            desc = {'description': '未加载'}
            task_type = None
            name = ds_id
        else:
            _, _, desc = datasets[ds_id]
            if ds_id == 'regression':
                task_type = 'regression'
                name = '回归样本'
            else:
                task_type = 'classification'
                name = 'MNIST样本' if ds_id == 'mnist' else '鸢尾花数据集'
        datasets_info.append({
            'id': ds_id,
            'name': name,
            'task_type': task_type,
            'description': desc
        })
    return jsonify({'datasets': datasets_info})

@app.route('/api/dataset/<dataset_id>', methods=['GET'])
def get_dataset(dataset_id):
    if dataset_id not in datasets or datasets[dataset_id] is None:
        return jsonify({'error': '数据集不存在'}), 404
    X, y, desc = datasets[dataset_id]
    sample_size = int(request.args.get('sample_size', min(100, len(X))))
    sample_size = max(1, min(sample_size, len(X)))
    indices = np.random.choice(len(X), sample_size, replace=False)
    return jsonify({
        'samples': X[indices].tolist(),
        'labels': y[indices].tolist(),
        'description': desc
    })

# ---------- 新增接口：后端执行数据分割 ----------
@app.route('/api/split', methods=['POST'])
def split_dataset():
    """
    请求 JSON 示例:
    {
        "dataset": "iris",
        "test_size": 0.3,
        "random_state": 42,
        "stratify": true
    }
    """
    data = request.json
    if not data or 'dataset' not in data:
        return jsonify({'error': '缺少 dataset 参数'}), 400
    dataset_id = data['dataset']
    if dataset_id not in datasets or datasets[dataset_id] is None:
        return jsonify({'error': '数据集不存在'}), 404

    # 参数解析与校验
    test_size = data.get('test_size', 0.3)
    try:
        test_size = float(test_size)
        if not (0.0 < test_size < 1.0):
            raise ValueError()
    except Exception:
        return jsonify({'error': 'test_size 必须是 (0,1) 之间的数字'}), 400

    random_state = data.get('random_state', None)
    if random_state is not None:
        try:
            random_state = int(random_state)
        except Exception:
            return jsonify({'error': 'random_state 必须是整数或 null'}), 400

    stratify_flag = bool(data.get('stratify', True))

    X, y, desc = datasets[dataset_id]
    stratify_param = y if (stratify_flag and y is not None and len(np.unique(y)) > 1) else None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )
    except Exception as e:
        # 可能因为 stratify 导致错误（比如某个类样本太少），尝试不分层
        if stratify_param is not None:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=None
                )
            except Exception as e2:
                return jsonify({'error': f'分割失败: {str(e2)}'}), 500
        else:
            return jsonify({'error': f'分割失败: {str(e)}'}), 500

    # 保存分割结果到内存，方便后续训练使用
    dataset_splits[dataset_id] = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'params': {
            'test_size': test_size,
            'random_state': random_state,
            'stratify': bool(stratify_param is not None)
        }
    }

    response = {
        'dataset': dataset_id,
        'n_samples': len(X),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'params': dataset_splits[dataset_id]['params'],
        # 返回少量样本用于前端预览
        'train_preview': {
            'X': X_train[:5].tolist(),
            'y': y_train[:5].tolist()
        },
        'test_preview': {
            'X': X_test[:5].tolist(),
            'y': y_test[:5].tolist()
        }
    }
    response = convert_numpy_types(response)
    return jsonify(response)

# ---------- 修改 /api/train：支持前端传入分割参数（优先） ----------
@app.route('/api/train', methods=['POST'])
def train_model():
    data = request.json
    print(f'{data}')
    if not data or 'algorithm' not in data or 'dataset' not in data:
        return jsonify({'error': '缺少算法或数据集参数'}), 400
    algorithm_id = data['algorithm']
    dataset_id = data['dataset']
    if dataset_id not in datasets or datasets[dataset_id] is None:
        return jsonify({'error': '数据集不存在'}), 404

    # 如果前端传入分割参数，则立即执行分割并保存（覆盖旧的分割）
    # 支持： test_size, random_state, stratify
    test_size = data.get('test_size', None)
    random_state = data.get('random_state', None)
    stratify_flag = data.get('stratify', None)

    if test_size is not None:
        # validate
        try:
            ts = float(test_size)
            if not (0.0 < ts < 1.0):
                raise ValueError()
        except Exception:
            return jsonify({'error': 'test_size 必须是 (0,1) 之间的数字'}), 400

        if random_state is not None:
            try:
                rs = int(random_state)
            except Exception:
                return jsonify({'error': 'random_state 必须是整数或 null'}), 400
        else:
            rs = None

        # stratify default True if dataset labeled
        if stratify_flag is None:
            stratify_flag = True
        stratify_flag = bool(stratify_flag)

        X, y, _ = datasets[dataset_id]
        stratify_param = y if (stratify_flag and y is not None and len(np.unique(y)) > 1) else None

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=ts, random_state=rs, stratify=stratify_param
            )
        except Exception as e:
            # 可能 stratify 失败，尝试不 stratify
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=ts, random_state=rs, stratify=None
                )
            except Exception as e2:
                return jsonify({'error': f'分割失败: {str(e2)}'}), 500

        dataset_splits[dataset_id] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'params': {
                'test_size': ts,
                'random_state': rs,
                'stratify': bool(stratify_param is not None)
            }
        }
    else:
        # 否则优先使用已分割的数据
        split = dataset_splits.get(dataset_id, None)
        if split is not None:
            X_train, X_test, y_train, y_test = split['X_train'], split['X_test'], split['y_train'], split['y_test']
        else:
            # 默认分割
            X, y, _ = datasets[dataset_id]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 创建模型实例并确定任务类型
    try:
        if algorithm_id == 'decision_tree':
            model = algos.DecisionTree(max_depth=5)
            task_type = 'classification'
        elif algorithm_id == 'naive_bayes':
            model = algos.NaiveBayes()
            task_type = 'classification'
        elif algorithm_id == 'knn':
            task_type = 'regression' if dataset_id == 'regression' else 'classification'
            model = algos.KNN(k=5, task_type=task_type)
        elif algorithm_id == 'svm':
            model = algos.SVM()
            task_type = 'classification'
        elif algorithm_id == 'random_forest':
            model = algos.RandomForest(n_trees=10)
            task_type = 'classification'
        elif algorithm_id == 'linear_regression':
            model = algos.LinearRegression()
            task_type = 'regression'
        elif algorithm_id == 'logistic_regression':
            model = algos.LogisticRegression()
            task_type = 'classification'
        elif algorithm_id == 'adaboost':
            model = algos.AdaBoost(n_estimators=10)
            task_type = 'classification'
        elif algorithm_id == 'kmeans':
            # 若训练集有标签则用类别数作为k，否则使用 3（默认）
            try:
                k_val = len(np.unique(y_train)) if y_train is not None else 3
                model = algos.KMeans(k=k_val)
            except Exception:
                model = algos.KMeans(k=3)
            task_type = 'clustering'
        elif algorithm_id == 'em':
            try:
                comps = len(np.unique(y_train)) if y_train is not None else 2
                model = algos.EMAlgorithm(n_components=comps)
            except Exception:
                model = algos.EMAlgorithm(n_components=2)
            task_type = 'clustering'
        else:
            return jsonify({'error': '算法不存在'}), 404
    except Exception as e:
        return jsonify({'error': f'初始化算法失败: {str(e)}'}), 500

    try:
        if task_type == 'clustering':
            # 聚类：通常使用所有无标签数据来训练（这里仍用训练+测试合并以演示）
            model.train(np.vstack((X_train, X_test)))
            y_pred = model.predict(X_test)
            metrics = {}
            try:
                metrics['ari'] = float(adjusted_rand_score(y_test, y_pred)) if (y_test is not None and len(y_test) > 0) else None
            except Exception:
                metrics['ari'] = None
            try:
                metrics['silhouette'] = float(silhouette_score(X_test, y_pred)) if (len(X_test) > 1) else None
            except Exception:
                metrics['silhouette'] = None
        else:
            model.train(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = calculate_metrics(y_test, y_pred, task_type)
    except Exception as e:
        return jsonify({'error': f'训练模型失败: {str(e)}'}), 500

    # 可视化数据（若模型实现了该方法）
    visualization_data = None
    try:
        visualization_data = model.get_visualization_data()
    except Exception:
        visualization_data = None

    response = {
        'algorithm': algorithm_id,
        'dataset': dataset_id,
        'metrics': metrics,
        'visualization': visualization_data,
        # 返回当前使用的 split 参数，方便前端核对
        'used_split': dataset_splits.get(dataset_id, {}).get('params', None)
    }
    response = convert_numpy_types(response)
    return jsonify(response)

# ---------- /api/compare 改为支持传入分割参数（复用 train 中的逻辑） ----------
@app.route('/api/compare', methods=['POST'])
def compare_algorithms():
    data = request.json
    if not data or 'algorithms' not in data or 'dataset' not in data or 'metric' not in data:
        return jsonify({'error': '缺少参数'}), 400
    algorithm_ids = data['algorithms']
    dataset_id = data['dataset']
    metric = data['metric']
    if dataset_id not in datasets or datasets[dataset_id] is None:
        return jsonify({'error': '数据集不存在'}), 404

    # 如果前端在 compare 请求中提供分割参数，则先分割并保存
    test_size = data.get('test_size', None)
    random_state = data.get('random_state', None)
    stratify_flag = data.get('stratify', None)

    if test_size is not None:
        # perform split and save - reuse code similar to /api/train
        try:
            ts = float(test_size)
            if not (0.0 < ts < 1.0):
                raise ValueError()
        except Exception:
            return jsonify({'error': 'test_size 必须是 (0,1) 之间的数字'}), 400

        if random_state is not None:
            try:
                rs = int(random_state)
            except Exception:
                return jsonify({'error': 'random_state 必须是整数或 null'}), 400
        else:
            rs = None

        if stratify_flag is None:
            stratify_flag = True
        stratify_flag = bool(stratify_flag)

        X, y, _ = datasets[dataset_id]
        stratify_param = y if (stratify_flag and y is not None and len(np.unique(y)) > 1) else None

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=ts, random_state=rs, stratify=stratify_param
            )
        except Exception:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=ts, random_state=rs, stratify=None
                )
            except Exception as e:
                return jsonify({'error': f'分割失败: {str(e)}'}), 500

        dataset_splits[dataset_id] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'params': {
                'test_size': ts,
                'random_state': rs,
                'stratify': bool(stratify_param is not None)
            }
        }

    # 使用已保存的分割或默认分割
    split = dataset_splits.get(dataset_id, None)
    if split is not None:
        X_train, X_test, y_train, y_test = split['X_train'], split['X_test'], split['y_train'], split['y_test']
    else:
        X, y, _ = datasets[dataset_id]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    results = {}
    for algorithm_id in algorithm_ids:
        try:
            # (模型实例化逻辑与 train_model 中一致)
            if algorithm_id == 'decision_tree':
                model = algos.DecisionTree(max_depth=5)
                task_type = 'classification'
            elif algorithm_id == 'naive_bayes':
                model = algos.NaiveBayes()
                task_type = 'classification'
            elif algorithm_id == 'knn':
                task_type = 'regression' if dataset_id == 'regression' else 'classification'
                model = algos.KNN(k=5, task_type=task_type)
            elif algorithm_id == 'svm':
                model = algos.SVM()
                task_type = 'classification'
            elif algorithm_id == 'random_forest':
                model = algos.RandomForest(n_trees=10)
                task_type = 'classification'
            elif algorithm_id == 'linear_regression':
                if dataset_id != 'regression':
                    results[algorithm_id] = {'metric': None, 'error': '不适用于分类任务'}
                    continue
                model = algos.LinearRegression()
                task_type = 'regression'
            elif algorithm_id == 'logistic_regression':
                model = algos.LogisticRegression()
                task_type = 'classification'
            elif algorithm_id == 'adaboost':
                model = algos.AdaBoost(n_estimators=10)
                task_type = 'classification'
            elif algorithm_id == 'kmeans':
                model = algos.KMeans(k=len(np.unique(y_train)))
                task_type = 'clustering'
            elif algorithm_id == 'em':
                model = algos.EMAlgorithm(n_components=len(np.unique(y_train)))
                task_type = 'clustering'
            else:
                results[algorithm_id] = {'metric': None, 'error': '算法不存在'}
                continue

            if task_type == 'clustering':
                model.train(np.vstack((X_train, X_test)))
                y_pred = model.predict(X_test)
                metrics = {}
                try:
                    metrics['ari'] = float(adjusted_rand_score(y_test, y_pred)) if (y_test is not None and len(y_test) > 0) else None
                except Exception:
                    metrics['ari'] = None
                try:
                    metrics['silhouette'] = float(silhouette_score(X_test, y_pred)) if (len(X_test) > 1) else None
                except Exception:
                    metrics['silhouette'] = None
            else:
                model.train(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics = calculate_metrics(y_test, y_pred, task_type)

            metric_value = metrics.get(metric, None)
            results[algorithm_id] = {'metric': metric_value}
        except Exception as e:
            results[algorithm_id] = {'metric': None, 'error': str(e)}

    results = convert_numpy_types(results)
    return jsonify({'results': results, 'used_split': dataset_splits.get(dataset_id, {}).get('params', None)})

# ---------- 启动应用 ----------
if __name__ == '__main__':
    load_all_datasets()
    app.run(debug=True, host='0.0.0.0', port=5000)
