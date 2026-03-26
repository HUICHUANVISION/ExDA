"""
ExDA - Explainability-Guided Data Augmentation

A data augmentation method for Cross-Project Defect Prediction (CPDP)
that uses feature importance to preserve semantic anchors while generating
synthetic samples through a Selective Variational Autoencoder (sVAE).

Main Components:
- exda_model.py: Core ExDA implementation with SelectiveVAE and hybrid feature importance
- data_loader.py: Dataset loading utilities for CPDP
- validation.py: Cross-project validation framework
- main.py: Example usage

Usage:
    from exda import ExDA
    exda = ExDA()
    exda.fit(X_train, y_train)
    X_aug, y_aug = exda.augment(X_train, y_train)
"""

from .exda_model import ExDA, SelectiveVAE
from .data_loader import load_dataset, CrossProjectDataLoader
from .validation import cross_project_evaluation

__version__ = "1.0.0"
__all__ = ["ExDA", "SelectiveVAE", "load_dataset", "CrossProjectDataLoader", "cross_project_evaluation"]