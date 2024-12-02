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
from random_forest_zhoumath import RandomForestZhoumath
from decision_tree_zhoumath import DecisionTreeZhoumath, DecisionTreeZhoumathLosloss
from random_forest_helper_zhoumath import EarlyStopperRF, FeatureImportancesRF

# Settings
warnings.filterwarnings("ignore", category=UserWarning)

# GradientBoost class
class GradientBoostZhoumath(RandomForestZhoumath):
    def __init__(self, learning_rate, num_base_trees, ensemble_column_rate, ensemble_sample_rate, task, split_criterion,
                 search_method, max_depth=None, pos_weight=1, random_column_rate=1,min_split_sample_rate=0,
                 min_leaf_sample_rate=0, verbose_for_tree=False, verbose_for_ensemble = False):
        """
        Initialize the Gradient Boosting model, based on Random Forest.
        :param learning_rate: The learning rate (shrinkage) for the gradient boosting process.
        :param num_base_trees: The number of base decision trees in the ensemble.
        :param ensemble_column_rate: The fraction of features to sample for each tree.
        :param ensemble_sample_rate: The fraction of samples to sample for each tree.
        :param task: The task type, either 'classification' or 'regression'.
        :param split_criterion: The splitting criterion used for the decision tree (e.g., 'gini', 'entropy', 'mse').
        :param search_method: The method for finding the best split ('dfs' or other methods).
        :param max_depth: The maximum depth of each tree (optional).
        :param pos_weight: The weight for the positive class in case of imbalanced classes (default is 1).
        :param random_column_rate: The rate of features randomly chosen for splitting (default is 1).
        :param min_split_sample_rate: The minimum fraction of samples required to split a node.
        :param min_leaf_sample_rate: The minimum fraction of samples required at a leaf node.
        :param verbose_for_tree: If True, enables verbose logging for individual decision trees.
        :param verbose_for_ensemble: If True, enables verbose logging for the ensemble training process.
        :return: None
        """
        super().__init__(num_base_trees, ensemble_column_rate, ensemble_sample_rate, task, split_criterion, search_method,
                         max_depth, pos_weight, random_column_rate, min_split_sample_rate, min_leaf_sample_rate,
                         verbose_for_tree, verbose_for_ensemble)
        
        self.learning_rate = learning_rate
     
    def fit(self, data, labels, val_data=None, val_labels=None, early_stop_rounds_for_tree=None,
            early_stop_rounds_for_forest = None, random_state=42):
        """
        Train the Gradient Boosting model using an ensemble of decision trees.
        :param data: The training data, a 2D numpy array where each row represents a sample.
        :param labels: The corresponding labels for the training data.
        :param val_data: The validation data (optional).
        :param val_labels: The validation labels (optional).
        :param early_stop_rounds_for_tree: Early stop rounds for individual trees (optional).
        :param early_stop_rounds_for_forest: Early stop rounds for the whole forest (optional).
        :param random_state: The random seed for reproducibility.
        :return: None
        """
        self.data = np.ascontiguousarray(data)
        self.labels = np.ascontiguousarray(labels)
        self.num_samples = data.shape[0]
        self.num_features = data.shape[1]
        self.random_state = random_state
        self.feature_importances = FeatureImportancesGB(self.num_features)
        self.early_stop_rounds_for_tree = early_stop_rounds_for_tree
        self.early_stopper_gb = None
        self.labels_mean = np.mean(self.labels)
        gbdt_mse_condition = self.task == 'regression' or (self.task == 'classification' and self.split_criterion == 'mse')
        gbdt_logloss_condition = self.task == 'classification' and self.split_criterion == 'logloss'
        
        if gbdt_mse_condition:
            self.fit_mse(val_data, val_labels, early_stop_rounds_for_forest)
        elif gbdt_logloss_condition:
            self.fit_logloss(val_data, val_labels, early_stop_rounds_for_forest)
        else:
            raise ValueError("MSE is the only supported loss function for regression,"
                             "and MSE and Entropy(logloss) are supported for classification.")
            
        self.data = None
        self.labels = None
        self.val_data = None
        self.val_labels = None
        self.tree_models_predictions = None
        self.tree_models_predictions_val = None
        
    def fit_mse(self, val_data, val_labels, early_stop_rounds_for_forest):    
        self.tree_models_predictions = self.labels_mean * np.ones(self.labels.shape)
        
        if (val_data is not None) and (val_labels is not None) and early_stop_rounds_for_forest is not None:
            self.val_data = np.ascontiguousarray(val_data)
            self.val_labels = np.ascontiguousarray(val_labels)
            self.labels_mean_val = np.mean(self.labels)
            self.tree_models_predictions_val = self.labels_mean_val * np.ones(self.val_labels.shape)
            self.early_stopper_gb = EarlyStopperGB(val_data=val_data,
                                                   val_labels=val_labels,
                                                   early_stop_rounds=early_stop_rounds_for_forest,
                                                   verbose=self.verbose_for_ensemble)
            
        for i in range(self.num_base_trees):
            labels_residual = self.labels - self.tree_models_predictions
            
            if self.early_stopper_gb is not None:
                labels_residual_val = self.val_labels - self.tree_models_predictions_val
            
            early_stop_triggered = self._generate_trees_mse(i, labels_residual, labels_residual_val)
            
            if early_stop_triggered:
                break
        
        if self.early_stopper_gb is not None:
            self.tree_models = copy.deepcopy(self.best_tree_models)
            return
    
        return
    
    def fit_logloss(self, val_data, val_labels, early_stop_rounds_for_forest):    
        self.tree_models_predictions = self.labels_mean * np.ones(self.labels.shape)
        self.tree_models_logodds = np.log(self.labels_mean / (1 - self.labels_mean)) * np.ones(self.labels.shape)
        
        if (val_data is not None) and (val_labels is not None) and early_stop_rounds_for_forest is not None:
            self.val_data = np.ascontiguousarray(val_data)
            self.val_labels = np.ascontiguousarray(val_labels)
            self.tree_models_predictions_val = self.labels_mean_val * np.ones(self.val_labels.shape)
            self.tree_models_logodds_val = np.log(self.labels_mean_val / (1 - self.labels_mean_val)) * np.ones(self.val_labels.shape)
            self.early_stopper_gb = EarlyStopperGB(val_data=val_data,
                                                   val_labels=val_labels,
                                                   early_stop_rounds=early_stop_rounds_for_forest,
                                                   verbose=self.verbose_for_ensemble)
            
        for i in range(self.num_base_trees):
            labels_residual = self.labels - self.tree_models_predictions
            labels_hessian = self.tree_models_predictions * (1 - self.tree_models_predictions)
            
            if self.early_stopper_gb is not None:
                labels_residual_val = self.val_labels - self.tree_models_predictions_val
                labels_hessian_val = self.tree_models_predictions_val * (1 - self.tree_models_predictions_val)
            
            early_stop_triggered = self._generate_trees_logloss(i, labels_residual, labels_residual_val,
                                                                labels_hessian, labels_hessian_val)
            
            if early_stop_triggered:
                break
        
        if self.early_stopper_gb is not None:
            self.tree_models = copy.deepcopy(self.best_tree_models)
            return
            
        return
   
    def _generate_trees_mse(self, i, train_residual, val_residual):
        """
        Generate a single decision tree to be added to the gradient boosting ensemble.
        :param i: The index of the current tree being trained.
        :param labels_residual: The residuals for the training data (difference between actual and predicted values).
        :param labels_residual_val: The residuals for the validation data (optional).
        :return: Boolean indicating whether early stopping was triggered (True/False).
        """
        np.random.seed(self.random_state + i)
        current_decision_tree = DecisionTreeZhoumath(task='regression',
                                                     split_criterion='mse',
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
        train_residual = train_residual[valid_samples].copy()
        current_decision_tree.fit(data=train_data,
                                  labels=train_residual,
                                  val_data=self.val_data,
                                  val_labels=train_residual,
                                  early_stop_rounds=self.early_stop_rounds_for_tree)
        self.tree_models.append(current_decision_tree)
        data_prediciton = current_decision_tree.predict_proba(self.data)
        self.tree_models_predictions += self.learning_rate * data_prediciton
        
        if self.early_stopper_gb is not None:
            val_data_prediciton = current_decision_tree.predict_proba(self.val_data)
            self.tree_models_predictions_val += self.learning_rate * val_data_prediciton
            early_stop_triggered = self.early_stopper_gb._evaluate_early_stop(current_decision_tree, self)
            
            if early_stop_triggered:
                self.tree_models = copy.deepcopy(self.best_tree_models)
                return True
        else:
            self.feature_importances._renew_feature_importances(current_decision_tree)
            
        return False
    
    def _generate_trees_logloss(self, i, train_residual, val_residual, train_hessians, val_hessians):
        """
        Generate a single decision tree to be added to the gradient boosting ensemble.
        :param i: The index of the current tree being trained.
        :param labels_residual: The residuals for the training data (difference between actual and predicted values).
        :param labels_residual_val: The residuals for the validation data (optional).
        :return: Boolean indicating whether early stopping was triggered (True/False).
        """
        np.random.seed(self.random_state + i)
        current_decision_tree = DecisionTreeZhoumathLosloss(search_method=self.search_method,
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
        train_residual = train_residual[valid_samples].copy()
        train_hessians = train_hessians[valid_samples]
        current_decision_tree.fit(data=train_data,
                                  labels=train_residual,
                                  hessians = train_hessians,
                                  val_data=self.val_data,
                                  val_labels=val_residual,
                                  val_hessians=val_hessians,
                                  early_stop_rounds=self.early_stop_rounds_for_tree)
        self.tree_models.append(current_decision_tree)
        data_logodds = current_decision_tree.predict_proba(self.data)
        self.tree_models_logodds += self.learning_rate * data_logodds
        self.tree_models_predictions = 1 / (1 + np.exp(-self.tree_models_logodds))
        
        if self.early_stopper_gb is not None:
            val_data_logodds = current_decision_tree.predict_proba(self.val_data)
            self.tree_models_logodds_val += self.learning_rate * val_data_logodds
            self.tree_models_predictions_val = 1 / (1 + np.exp(-self.tree_models_logodds_val))
            early_stop_triggered = self.early_stopper_gb._evaluate_early_stop(current_decision_tree, self)
            
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
        tree_predictions = np.zeros((data.shape[0], len(self.tree_models) + 1))
        tree_predictions[:,0] = self.labels_mean * np.ones(data.shape[0])
        
        for i in range(len(self.tree_models)):
            tree_predictions[:, i+1] = self.tree_models[i].predict_proba(data) * self.learning_rate
            
        tree_prediction = tree_predictions.sum(axis = 1)
        return tree_prediction

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
    