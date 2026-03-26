# ExDA - Explainability-Guided Data Augmentation

A data augmentation method for Cross-Project Defect Prediction (CPDP) that preserves semantic anchors while generating synthetic samples through a Selective Variational Autoencoder (sVAE).

## Overview

ExDA addresses the challenge of limited labeled data in CPDP by:

1. **Hybrid Feature Importance**: Combines statistical correlation (point-biserial) and model-based importance (Logistic Regression coefficients) to identify semantic anchors
2. **Selective VAE**: Generates synthetic samples by preserving high-importance features while perturbing low-importance features
3. **Semantic Anchor Preservation**: Ensures generated samples retain critical defect-related characteristics

## Installation

```bash
pip install torch numpy pandas scikit-learn scipy tqdm
```

## Quick Start

```python
import numpy as np
from exda import ExDA
from sklearn.linear_model import LogisticRegression

# Your defect data
X = np.array([[...], [...], ...])  # Features
y = np.array([0, 1, 0, ...])       # Labels (1=defect)

# Initialize ExDA
exda = ExDA(
    lambda_param=0.7,          # Hybrid importance weight
    k_top_features=0.25,      # Top 25% features as semantic anchors
    latent_dim=32,             # VAE latent dimension
    epochs=50,                 # Training epochs
    augmentation_percentage=0.3
)

# Train and augment
exda.fit(X, y)
X_aug, y_aug = exda.augment(X, y)

# Train classifier
clf = LogisticRegression()
clf.fit(X_aug, y_aug)
```

## Datasets

The `datasets/` folder contains four CPDP datasets:

| Dataset | Projects | Modules | Features |
|---------|----------|---------|----------|
| AEEEM | EQ, JDT, Lucene, Mylyn, PDE | 4,371 | 61 |
| JIRA | Camel, Hive, Ivy, Pig, Prop, Xalan | 2,705 | 101 |
| NASA | CM1, JM1, KC1, KC3, MC1, MW1 | 15,682 | 39 |
| PROMISE | ant, camel, ivy, jedit, log4j, etc. | 7,457 | 65 |

Each CSV file has the last column as the target (`class`: 0=non-defective, 1=defective).

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `lambda_param` | Weight for hybrid importance (0-1) | 0.7 |
| `k_top_features` | Fraction of top features as anchors | 0.25 |
| `latent_dim` | VAE latent space dimension | 32 |
| `hidden_dim` | VAE hidden layer dimension | 64 |
| `epochs` | Training epochs for VAE | 50 |
| `augmentation_percentage` | Target augmentation ratio | 0.3 |
| `target_ratio` | Target positive class ratio | 0.5 |

## Citation

If you use ExDA in your research, please cite our paper:

```
@article{exda2024,
  title={ExDA: Explainability-Guided Data Augmentation for Cross-Project Defect Prediction},
  author={},
  journal={},
  year={2024}
}
```

## License

MIT License