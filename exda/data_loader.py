"""数据加载和处理模块"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_data_from_csv(file_path, target_column='Defective'):
    """
    从CSV文件加载数据
    """
    data = pd.read_csv(file_path)
    
    if target_column not in data.columns:
        target_column = data.columns[-1]
    
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    feature_names = X.columns.tolist()
    X = X.values
    y = y.values
    
    return X, y, feature_names

def create_small_dataset(X, y, reduction_ratio=0.5, random_state=42):
    """
    通过随机删除样本来创建小数据集
    """
    np.random.seed(random_state)
    
    n_samples = len(y)
    n_keep = int(n_samples * reduction_ratio)
    keep_indices = np.random.choice(n_samples, n_keep, replace=False)
    
    X_small = X[keep_indices]
    y_small = y[keep_indices]
    
    return X_small, y_small

