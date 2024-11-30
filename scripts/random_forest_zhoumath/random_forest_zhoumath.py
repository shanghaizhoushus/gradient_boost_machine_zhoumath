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
script_dir = os.path.abspath(os.path.join(os.getcwd(), '../../scripts/decision_tree_zhoumath'))
sys.path.insert(0, script_dir)
from decision_tree_zhoumath import DecisionTreeZhoumath
from random_forest_helper_zhoumath import EarlyStopperRF, FeatureImportancesRF

# Settings
warnings.filterwarnings("ignore", category=UserWarning)

# RandomForest class
class RandomForestZhoumath(DecisionTreeZhoumath):
    def __init__(self, num_base_trees, ensemble_column_rate, ensemble_sample_rate, task, split_criterion, search_method, max_depth=None,
                 pos_weight=1, random_column_rate=1,min_split_sample_rate=0, min_leaf_sample_rate=0, verbose_for_tree=False,
                 verbose_for_ensemble = False):
        """
        Initialize the Random Forest model.
        :param num_base_trees: The number of base decision trees in the random forest.
        :param ensemble_column_rate: The rate of features to randomly sample for each decision tree.
        :param ensemble_sample_rate: The rate of samples to randomly sample for each decision tree.
        :param split_criterion: The criterion used for splitting nodes (e.g., "gini" or "entropy").
        :param search_method: The method for searching the best split ("dfs" or others).
        :param max_depth: The maximum depth of the decision trees (optional).
        :param pos_weight: The weight for the positive class in the decision tree (default is 1).
        :param random_column_rate: The rate of columns to randomly choose for splitting nodes (default is 1).
        :param min_split_sample_rate: The minimum sample rate for splitting a node.
        :param min_leaf_sample_rate: The minimum sample rate for a leaf node.
        :param verbose_for_tree: Boolean flag to enable verbose logging for individual trees.
        :param verbose_for_ensemble: Boolean flag to enable verbose logging for the random forest.
        :return: None
        """
        super().__init__(task, split_criterion, search_method, max_depth, pos_weight,
                         random_column_rate, min_split_sample_rate, min_leaf_sample_rate)
        
        self.num_base_trees = num_base_trees
        self.ensemble_column_rate = ensemble_column_rate
        self.ensemble_sample_rate = ensemble_sample_rate
        self.verbose_for_tree = verbose_for_tree
        self.verbose_for_ensemble = verbose_for_ensemble
        self.tree_models = []
        self.best_tree_models = []
     
    def fit(self, data, labels, val_data=None, val_labels=None, early_stop_rounds_for_tree=None,
            early_stop_rounds_for_forest = None, random_state=42):
        """
        Train the random forest model.
        :param data: The training data, a 2D numpy array where each row is a sample.
        :param labels: The labels for the training data.
        :param val_data: The validation data (optional).
        :param val_labels: The labels for the validation data (optional).
        :param early_stop_rounds_for_tree: Early stop rounds for individual decision trees (optional).
        :param early_stop_rounds_for_forest: Early stop rounds for the whole random forest (optional).
        :param random_state: The random seed for reproducibility.
        :return: None
        """
        self.data = np.ascontiguousarray(data)
        self.labels = np.ascontiguousarray(labels)
        self.val_data = None
        self.val_labels = None
        self.num_samples = data.shape[0]
        self.num_features = data.shape[1]
        self.random_state = random_state
        self.feature_importances = FeatureImportancesRF(self.num_features)
        self.early_stop_rounds_for_tree = early_stop_rounds_for_tree
        self.early_stopper_rf = None
        
        if (val_data is not None) and (val_labels is not None) and (early_stop_rounds_for_forest is not None):
            self.val_data = np.ascontiguousarray(val_data)
            self.val_labels = np.ascontiguousarray(val_labels)
            self.early_stopper_rf = EarlyStopperRF(val_data=self.val_data,
                                                   val_labels=self.val_labels,
                                                   early_stop_rounds=early_stop_rounds_for_forest,
                                                   verbose=self.verbose_for_ensemble)
        
        for i in range(self.num_base_trees):
            early_stop_triggered = self._generate_trees(i)
            if early_stop_triggered:
                break
            
        self.data = None
        self.labels = None
        self.val_data = None
        self.val_labels = None
                
        if self.early_stopper_rf is not None:
            self.tree_models = copy.deepcopy(self.best_tree_models)
            return
        
        return
    
    def _generate_trees(self, i):
        """
        Generate a single decision tree for the random forest.
        :param i: The index of the current tree.
        :return: Boolean indicating whether early stopping was triggered.
        """
        np.random.seed(self.random_state + i)
        current_decision_tree = DecisionTreeZhoumath(task = self.task,
                                                     split_criterion=self.split_criterion,
                                                     search_method=self.search_method,
                                                     max_depth=self.max_depth,
                                                     pos_weight=self.pos_weight,
                                                     random_column_rate=self.random_column_rate,
                                                     min_split_sample_rate=self.min_split_sample_rate,
                                                     min_leaf_sample_rate=self.min_leaf_sample_rate,
                                                     verbose=self.verbose_for_tree)
        valid_samples_num = max(np.ceil(self.num_samples * self.ensemble_sample_rate), 2).astype(np.int32)
        valid_samples = np.sort(np.random.choice(np.arange(self.num_samples),
                                                 size=valid_samples_num, replace=False)).astype(np.int32)
        muted_features_num = np.floor(self.num_features *(1 -  self.ensemble_column_rate)).astype(np.int32)
        muted_features = np.sort(np.random.choice(np.arange(self.num_features),
                                                  size=muted_features_num, replace=False)).astype(np.int32)
        train_data = self.data[valid_samples, :].copy()
        train_data[:, muted_features] = -np.inf
        train_labels = self.labels[valid_samples].copy()
        current_decision_tree.fit(data=train_data,
                                  labels=train_labels,
                                  val_data=self.val_data,
                                  val_labels=self.val_labels,
                                  early_stop_rounds=self.early_stop_rounds_for_tree)
        self.tree_models.append(current_decision_tree)
        
        if self.early_stopper_rf is not None:
            early_stop_triggered = self.early_stopper_rf._evaluate_early_stop(self, current_decision_tree)
            if early_stop_triggered:
                self.tree_models = copy.deepcopy(self.best_tree_models)
                return True
        else:
            self.feature_importances._renew_feature_importances(current_decision_tree)
        
        return False
            
    def predict_proba(self, data):
        """
        Predict the probabilities for the given data.
        :param data: The input data for prediction.
        :return: A 2D numpy array of predicted probabilities.
        """
        tree_predictions = np.zeros((data.shape[0], len(self.tree_models)))
        
        for i in range(len(self.tree_models)):
            tree_predictions[:, i] = self.tree_models[i].predict_proba(data)
            
        tree_prediction = tree_predictions.mean(axis = 1)
        return tree_prediction
