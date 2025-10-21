import numpy as np

class SVM:
    """支持向量机算法实现（二分类）"""
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000, kernel='linear', gamma=0.5):
        """
        初始化SVM模型
        :param learning_rate: 学习率
        :param lambda_param: 正则化参数
        :param n_iters: 迭代次数
        :param kernel: 核函数 ('linear' 或 'rbf')
        :param gamma: RBF核参数
        """
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.kernel = kernel
        self.gamma = gamma
        self.w = None
        self.b = None
        self.support_vectors = None

    def _kernel_function(self, X1, X2):
        """核函数实现"""
        if self.kernel == 'linear':
            return np.dot(X1, X2)
        elif self.kernel == 'rbf':
            diff = X1 - X2
            return np.exp(-self.gamma * np.dot(diff, diff))
        else:
            raise ValueError("不支持的核函数，目前仅支持 'linear' 或 'rbf'")

    def train(self, X, y):
        """
        训练SVM模型
        :param X: 特征数据
        :param y: 标签数据（应为0和1或-1和1）
        """
        X = np.array(X)
        y = np.array(y)

        # 确保标签是-1和1
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape

        # ✅ 加入随机扰动让每次训练略有不同
        rng = np.random.default_rng()
        shuffle_idx = rng.permutation(n_samples)
        X = X[shuffle_idx]
        y_ = y_[shuffle_idx]

        # 初始化权重和偏置
        self.w = np.zeros(n_features)
        self.b = 0

        # 梯度下降优化
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # 对RBF核的支持
                if self.kernel == 'rbf':
                    kernel_sum = np.sum([self._kernel_function(x_i, X[j]) * y_[j] for j in range(n_samples)])
                    condition_value = kernel_sum - self.b
                else:
                    condition_value = np.dot(x_i, self.w) - self.b

                condition = y_[idx] * condition_value >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]

        # ✅ 识别支持向量（距离边界最近的点）
        support_vector_indices = []
        for idx, x_i in enumerate(X):
            if y_[idx] * (np.dot(x_i, self.w) - self.b) <= 1.001:
                support_vector_indices.append(idx)

        self.support_vectors = {
            'X': X[support_vector_indices],
            'y': y_[support_vector_indices]
        }

    def predict(self, X):
        """预测样本类别"""
        if self.w is None:
            raise RuntimeError("模型尚未训练，请先调用 train() 方法")
        X = np.array(X)

        if self.kernel == 'rbf':
            y_pred = []
            for x in X:
                kernel_sum = np.sum([self._kernel_function(x, sv) for sv in self.support_vectors['X']])
                y_pred.append(kernel_sum - self.b)
            y_pred = np.array(y_pred)
        else:
            y_pred = np.dot(X, self.w) - self.b

        return np.where(y_pred >= 0, 1, 0)

    def get_visualization_data(self):
        """获取SVM可视化数据"""
        if self.w is None:
            return None

        return {
            'weights': self.w.tolist(),
            'bias': float(self.b),
            'support_vectors': {
                'X': self.support_vectors['X'].tolist(),
                'y': self.support_vectors['y'].tolist()
            },
            'kernel': self.kernel
        }
