# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 21:53:52 2024

@author: zhoushus
"""
# Import packages
import sys
import os
import copy
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
script_dir = os.path.abspath(os.path.join(os.getcwd(), '../../scripts/decision_tree_zhoumath'))
sys.path.insert(0, script_dir)
from decision_tree_helper_zhoumath import EarlyStopper, FeatureImportances

# EarlyStopperRF Class
class EarlyStopperRF(EarlyStopper):
    def __init__(self, val_data, val_labels, early_stop_rounds, verbose):
        """
        Initialize the Early Stopper for Random Forest.
        :param val_data: The validation data used to evaluate the model during training.
        :param val_labels: The validation labels corresponding to the validation data.
        :param early_stop_rounds: The number of rounds without improvement before early stopping is triggered.
        :param verbose: Boolean flag to enable verbose logging for early stopping.
        :return: None
        """
        super().__init__(val_data=val_data,
                       val_labels=val_labels,
                       early_stop_rounds=early_stop_rounds,
                       verbose=verbose)
        self.current_trees = 0
        self.currert_early_stop_rounds = 0
        self.feature_importances_cache = FeatureImportancesRF(val_data.shape[1])

    def _evaluate_early_stop(self, randomforestzhoumath, current_decision_tree):
        """
        Evaluate whether early stopping should be triggered based on the current performance.
        :param randomforestzhoumath: The random forest model.
        :param current_decision_tree: The current decision tree in the random forest.
        :return: Boolean indicating whether early stopping was triggered.
        """
        self.current_trees += 1
        self.feature_importances_cache._renew_feature_importances(current_decision_tree)
        labels_pred = randomforestzhoumath.predict_proba(randomforestzhoumath.data)[:, 1]
        val_labels_pred = randomforestzhoumath.predict_proba(self.val_data)[:, 1]
        
        if randomforestzhoumath.task == 'classification':
            train_metric = roc_auc_score(randomforestzhoumath.labels, labels_pred)
            val_metric = roc_auc_score(self.val_labels, val_labels_pred)
            if self.verbose:
                print(f'Current trees: {self.current_trees}, current train AUC: {train_metric:.3f}, current val AUC: {val_metric:.3f}')
        
        if randomforestzhoumath.task == 'regression':
            train_metric = -np.mean((randomforestzhoumath.labels - labels_pred) ** 2)
            val_metric = -np.mean((self.val_labels - val_labels_pred) ** 2)
            if self.verbose:
                print(f'Current trees : {self.current_trees - 1}, current train MSE: {-train_metric:.3f}, current val MSE: {-val_metric:.3f}')
            
        if val_metric > self.best_metric:
            self.best_metric = val_metric
            randomforestzhoumath.best_tree_models = copy.deepcopy(randomforestzhoumath.tree_models)
            randomforestzhoumath.feature_importances._renew_cache(self.feature_importances_cache)
            self.feature_importances_cache = FeatureImportancesRF(self.val_data.shape[1])
            self.currert_early_stop_rounds = 0
        else:
            self.currert_early_stop_rounds += 1

        if self.currert_early_stop_rounds >= self.early_stop_rounds:
            if self.verbose:
                print(f'Early stop triggered at num trees {self.current_trees}')
            return True

        return False
            
# FeatureImportancesRF Class
class FeatureImportancesRF(FeatureImportances):
    def __init__(self, num_features):
        """
        Initialize the FeatureImportances for Random Forest.
        :param num_features: The number of features in the dataset.
        :return: None
        """
        super().__init__(num_features)
    
    def _renew_feature_importances(self, current_decision_tree):
        """
        Renew the feature importances based on the current decision tree.
        :param current_decision_tree: The current decision tree to update feature importances.
        :return: None
        """
        self.split_importance += current_decision_tree.feature_importances.split_importance
        self.gain_importance += current_decision_tree.feature_importances.gain_importance
    
    def get_feature_importances_df(self, col_names = None):
        """
        Get a pandas DataFrame of feature importances.
        :param col_names: The column names for the features (optional).
        :return: A pandas DataFrame of feature importances.
        """
        feature_importances_df = pd.DataFrame()
        feature_importances_df['split_importance'] = self.split_importance
        feature_importances_df['gain_importance'] = self.gain_importance
        
        if col_names is not None:
            feature_importances_df.index = col_names
            
        return feature_importances_df
