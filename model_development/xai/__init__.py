"""
XAI (Explainable AI) Module for Two-Stage Anomaly Detection System

This module provides comprehensive explainability capabilities for:
- Autoencoder-based anomaly detection
- Attack type classification
- Integrated two-stage explanations
"""

__version__ = "4.0.0"
__author__ = "XAI Integration Team"

from .foundation.data_analyzer import DataAnalyzer
from .foundation.feature_importance import FeatureImportanceAnalyzer
from .foundation.baseline_explainer import BaselineExplainer
from .autoencoder_explainer import AutoencoderExplainer
from .classifier_explainer import ClassifierExplainer
from .integrated_explainer import IntegratedExplainer

__all__ = [
    "DataAnalyzer",
    "FeatureImportanceAnalyzer", 
    "BaselineExplainer",
    "AutoencoderExplainer",
    "ClassifierExplainer",
    "IntegratedExplainer"
]
