import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

class LinearRegression:
    """闭式解线性回归，支持多项式特征和标准化，兼容现有后端接口"""
    def __init__(self, poly_degree=1, normalize=True):
        self.poly_degree = poly_degree
        self.normalize = normalize

        self.poly = None
        self.scaler = None
        self.weights = None
        self.bias = None
        self.train_mse_history = []

    def train(self, X, y):
        """
        X: 特征矩阵 (n_samples, n_features)
        y: 目标向量 (n_samples,)
        """
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).flatten()
        n_samples, n_features = X.shape

        # 生成多项式特征
        if self.poly_degree > 1:
            self.poly = PolynomialFeatures(degree=self.poly_degree, include_bias=False)
            X = self.poly.fit_transform(X)

        # 特征标准化
        if self.normalize:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        # 闭式解: w = (X^T X)^-1 X^T y
        # 为了包含 bias, 添加一列全1
        X_bias = np.hstack([np.ones((n_samples, 1)), X])
        try:
            w_full = np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ y
        except np.linalg.LinAlgError:
            raise ValueError("矩阵不可逆，训练失败")

        self.bias = float(w_full[0])
        self.weights = w_full[1:]

        # 训练误差
        y_pred = self.predict(X)
        mse = float(np.mean((y - y_pred) ** 2))
        self.train_mse_history = [mse]

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        if self.poly is not None:
            X = self.poly.transform(X)
        if self.normalize and self.scaler is not None:
            X = self.scaler.transform(X)
        return np.dot(X, self.weights) + self.bias

    def get_visualization_data(self):
        return {
            'type': 'linear_regression',
            'coefficients': self.weights.tolist() if self.weights is not None else [],
            'intercept': float(self.bias) if self.bias is not None else 0.0,
            'poly_degree': int(self.poly_degree),
            'train_mse_history': [float(m) for m in self.train_mse_history],
            'normalize': self.normalize
        }
