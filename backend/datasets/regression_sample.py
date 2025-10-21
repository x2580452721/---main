import numpy as np

def load_regression_sample():
    """
    加载回归样本数据集（线性关系）
    
    返回:
        X: 特征数据
        y: 目标值（连续值）
        description: 数据集描述
    """
    np.random.seed(42)
    n_samples = 1000

    # 特征生成
    area = np.random.uniform(50, 200, n_samples)
    age = np.random.uniform(0, 50, n_samples)
    rooms = np.random.randint(1, 6, n_samples)
    distance = np.random.uniform(1, 20, n_samples)

    X = np.column_stack((area, age, rooms, distance))

    # 目标值生成，纯线性关系
    y = (
        1.5 * area - 1.0 * age + 20 * rooms - 3 * distance +
        np.random.normal(0, 3, n_samples)  # 小噪声
    )

    y = np.maximum(y, 30)  # 保证房价为正

    description = {
        'name': '回归样本数据集',
        'samples': n_samples,
        'features': X.shape[1],
        'feature_names': ['房屋面积(平方米)', '房龄(年)', '房间数量', '距离市中心(公里)'],
        'target_name': '房价(万元)',
        'description': '包含房屋特征与房价，纯线性关系，适合线性回归模型拟合。'
    }

    return X, y, description
