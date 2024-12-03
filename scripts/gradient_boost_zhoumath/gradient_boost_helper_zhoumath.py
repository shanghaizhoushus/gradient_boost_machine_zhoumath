# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 11:37:02 2024

@author: zhoushus
"""

# Import packages
import sys
import os
import copy
import warnings
import numpy as np
from sklearn.metrics import roc_auc_score
script_dir = os.path.abspath(os.path.join(os.getcwd(), '../../scripts/decision_tree_zhoumath'))
sys.path.insert(0, script_dir)
script_dir = os.path.abspath(os.path.join(os.getcwd(), '../../scripts/random_forest_zhoumath'))
sys.path.insert(0, script_dir)
from random_forest_helper_zhoumath import EarlyStopperRF, FeatureImportancesRF

# Settings
warnings.filterwarnings("ignore", category=UserWarning)

class EarlyStopperGB(EarlyStopperRF):
    def __init__(self, val_data, val_labels, early_stop_rounds, verbose):
        """
        Early stopping mechanism for the gradient boosting model.
        :param val_data: The validation data to evaluate the model's performance.
        :param val_labels: The true labels for the validation data.
        :param early_stop_rounds: The number of rounds without improvement before stopping.
        :param verbose: Whether to print verbose output during early stopping.
        """
        super().__init__(val_data=val_data,
                         val_labels=val_labels,
                         early_stop_rounds=early_stop_rounds,
                         verbose=verbose)

    def _evaluate_early_stop(self, current_decision_tree, gradientboostzhoumath):
        """
        Evaluate the model's performance on the validation data and check if early stopping should be triggered.
        :param predictions: The predicted labels for the validation data.
        :return: Boolean indicating whether early stopping should be triggered.
        """
        self.current_trees += 1
        self.feature_importances_cache._renew_feature_importances(current_decision_tree)
        train_labels = gradientboostzhoumath.labels
        labels_preds = gradientboostzhoumath.tree_models_predictions
        val_labels_preds = gradientboostzhoumath.tree_models_predictions_val
        
        if gradientboostzhoumath.task == 'classification':
            train_metric = roc_auc_score(train_labels, labels_preds)
            val_metric = roc_auc_score(self.val_labels, val_labels_preds)
            if self.verbose:
                print(f'Current trees: {self.current_trees}, current train AUC: {train_metric:.3f}, current val AUC: {val_metric:.3f}')
        
        if gradientboostzhoumath.task == 'regression':
            train_metric = 1 - np.mean((train_labels - labels_preds) ** 2) / np.mean((train_labels - train_labels.mean()) ** 2)
            val_metric = 1 - np.mean((self.val_labels - val_labels_preds) ** 2) / np.mean((self.val_labels - self.val_labels.mean()) ** 2)
            if self.verbose:
                print(f'Current trees : {self.current_trees}, current train R2: {train_metric:.3f}, current val R2: {val_metric:.3f}')
        
        if val_metric > self.best_metric:
            self.best_metric = val_metric
            gradientboostzhoumath.best_tree_models = copy.deepcopy(gradientboostzhoumath.tree_models)
            gradientboostzhoumath.feature_importances._renew_cache(self.feature_importances_cache)
            self.feature_importances_cache = FeatureImportancesGB(self.val_data.shape[1])
            self.current_early_stop_rounds= 0
        else:
            self.current_early_stop_rounds += 1

        if self.current_early_stop_rounds >= self.early_stop_rounds:
            if self.verbose:
                print(f'Early stop triggered at num trees {self.current_trees}')
            return True
        
        return False

# FeatureImportancesGB Class
class FeatureImportancesGB(FeatureImportancesRF):
    def __init__(self, num_features):
        """
        Initialize the FeatureImportances for Random Forest.
        :param num_features: The number of features in the dataset.
        :return: None
        """
        super().__init__(num_features)
    