
"""ExDA模型相关类"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pointbiserialr
from tqdm import tqdm

class SelectiveVAE(nn.Module):
    """选择性变分自编码器（sVAE）"""
    
    def __init__(self, input_dim_low, input_dim_high, hidden_dim=64, latent_dim=32):
        super(SelectiveVAE, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim_low + input_dim_high, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )
        
        # 潜在空间的均值和方差
        self.fc_mu = nn.Linear(hidden_dim//2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim//2, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + input_dim_high, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim_low)
        )
        
    def encode(self, x_low, x_high):
        x = torch.cat([x_low, x_high], dim=1)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, x_high):
        x = torch.cat([z, x_high], dim=1)
        return self.decoder(x)
    
    def forward(self, x_low, x_high):
        mu, logvar = self.encode(x_low, x_high)
        z = self.reparameterize(mu, logvar)
        x_low_recon = self.decode(z, x_high)
        return x_low_recon, mu, logvar

class ExDA:
    """Explainability-Guided Data Augmentation 实现"""
    
    def __init__(self, lambda_param=0.7, k_top_features=0.25, latent_dim=32, hidden_dim=64, 
                 lr=1e-3, epochs=50, batch_size=16, augmentation_percentage=0.3, target_ratio=0.5):
        self.lambda_param = lambda_param
        self.k_top_features = k_top_features
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.augmentation_percentage = augmentation_percentage
        self.target_ratio = target_ratio
        
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.F_high = None
        self.F_low = None
        self.svae = None
        
    def compute_feature_importance(self, X, y):
        """计算混合特征重要性"""
        n_features = X.shape[1]
        statistical_importance = np.zeros(n_features)
        model_importance = np.zeros(n_features)
        
        # 1. 统计相关性（点二列相关）
        for i in range(n_features):
            corr, p_value = pointbiserialr(X[:, i], y)
            statistical_importance[i] = abs(corr) if not np.isnan(corr) else 0
        
        # 2. 模型重要性（逻辑回归系数）
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X, y)
        model_importance = np.abs(lr_model.coef_[0])
        
        # 3. 混合重要性
        hybrid_importance = (self.lambda_param * statistical_importance + 
                           (1 - self.lambda_param) * model_importance)
        
        return hybrid_importance, statistical_importance, model_importance
    
    def select_important_features(self, importance_scores):
        """选择高重要性特征"""
        n_features = len(importance_scores)
        
        if isinstance(self.k_top_features, float):
            k = int(n_features * self.k_top_features)
        else:
            k = min(self.k_top_features, n_features)
        
        k = max(1, k)
        
        high_idx = np.argsort(importance_scores)[-k:]
        low_idx = np.setdiff1d(np.arange(n_features), high_idx)
        
        return high_idx, low_idx
    
    def vae_loss(self, x_recon, x_original, mu, logvar):
        """VAE损失函数（ELBO）"""
        recon_loss = nn.MSELoss()(x_recon, x_original)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + 0.01 * kl_loss
    
    def fit(self, X, y):
        """训练ExDA模型"""
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        self.feature_importance, stat_imp, model_imp = self.compute_feature_importance(X_scaled, y)
        self.F_high, self.F_low = self.select_important_features(self.feature_importance)
        
        if len(self.F_low) == 0:
            return self
        
        self.svae = SelectiveVAE(
            input_dim_low=len(self.F_low),
            input_dim_high=len(self.F_high),
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim
        )
        
        optimizer = optim.Adam(self.svae.parameters(), lr=self.lr)
        
        X_high = X_tensor[:, self.F_high]
        X_low = X_tensor[:, self.F_low]
        
        dataset = TensorDataset(X_low, X_high)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.svae.train()
        # 使用tqdm显示训练进度
        for epoch in tqdm(range(self.epochs), desc="训练ExDA模型", leave=False):
            total_loss = 0
            for batch_X_low, batch_X_high in dataloader:
                optimizer.zero_grad()
                x_recon, mu, logvar = self.svae(batch_X_low, batch_X_high)
                loss = self.vae_loss(x_recon, batch_X_low, mu, logvar)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        return self
    
    def augment(self, X, y, n_times=1):
        """生成增强样本"""
        if self.svae is None or len(self.F_low) == 0:
            return X, y
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        positive_count = np.sum(y == 1)
        negative_count = np.sum(y == 0)
        current_ratio = positive_count / len(y) if len(y) > 0 else 0
        
        total_samples_to_generate = int(len(X) * self.augmentation_percentage)
        total_final = len(X) + total_samples_to_generate
        target_positive = int(total_final * self.target_ratio)
        target_negative = total_final - target_positive
        
        positive_to_generate = max(0, target_positive - positive_count)
        negative_to_generate = max(0, target_negative - negative_count)
        
        if positive_to_generate + negative_to_generate > total_samples_to_generate:
            scale_factor = total_samples_to_generate / (positive_to_generate + negative_to_generate)
            positive_to_generate = int(positive_to_generate * scale_factor)
            negative_to_generate = total_samples_to_generate - positive_to_generate
        
        X_augmented = []
        y_augmented = []
        
        self.svae.eval()
        with torch.no_grad():
            # 生成正样本
            if positive_to_generate > 0:
                positive_indices = np.where(y == 1)[0]
                if len(positive_indices) > 0:
                    base_generate = positive_to_generate // len(positive_indices)
                    remainder = positive_to_generate % len(positive_indices)
                    
                    # 使用tqdm显示生成进度
                    for i, idx in tqdm(enumerate(positive_indices), total=len(positive_indices), 
                                      desc="生成正样本", leave=False):
                        x_original = X_tensor[idx]
                        x_high = x_original[self.F_high].unsqueeze(0)
                        x_low_original = x_original[self.F_low].unsqueeze(0)
                        
                        generate_count = base_generate + (1 if i < remainder else 0)
                        
                        for _ in range(generate_count):
                            x_low_new, _, _ = self.svae(x_low_original, x_high)
                            x_new = torch.zeros_like(x_original)
                            x_new[self.F_high] = x_high.squeeze(0)
                            x_new[self.F_low] = x_low_new.squeeze(0)
                            
                            X_augmented.append(x_new.numpy())
                            y_augmented.append(1)
            
            # 生成负样本
            if negative_to_generate > 0:
                negative_indices = np.where(y == 0)[0]
                if len(negative_indices) > 0:
                    base_generate = negative_to_generate // len(negative_indices)
                    remainder = negative_to_generate % len(negative_indices)
                    
                    # 使用tqdm显示生成进度
                    for i, idx in tqdm(enumerate(negative_indices), total=len(negative_indices), 
                                      desc="生成负样本", leave=False):
                        x_original = X_tensor[idx]
                        x_high = x_original[self.F_high].unsqueeze(0)
                        x_low_original = x_original[self.F_low].unsqueeze(0)
                        
                        generate_count = base_generate + (1 if i < remainder else 0)
                        
                        for _ in range(generate_count):
                            x_low_new, _, _ = self.svae(x_low_original, x_high)
                            x_new = torch.zeros_like(x_original)
                            x_new[self.F_high] = x_high.squeeze(0)
                            x_new[self.F_low] = x_low_new.squeeze(0)
                            
                            X_augmented.append(x_new.numpy())
                            y_augmented.append(0)
        
        X_combined = np.vstack([X, np.array(X_augmented)])
        y_combined = np.hstack([y, np.array(y_augmented)])
        X_combined = self.scaler.inverse_transform(X_combined)
        
        return X_combined, y_combined
