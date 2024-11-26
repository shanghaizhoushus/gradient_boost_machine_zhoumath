# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 09:31:27 2024

@author: zhoushus
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from numba import njit

# TreeNode class
class TreeNode:
    def __init__(self, feature=None, threshold=None, prob=None, left=None, right=None, null_direction=None):
        """
        Initialize a TreeNode.
        :param feature: Feature index used for splitting.
        :param threshold: Threshold value for splitting.
        :param prob: Probability distribution for the node.
        :param left: Left child node.
        :param right: Right child node.
        :param null_direction: Direction for handling null values.
        """
        self.feature = feature
        self.threshold = threshold
        self.prob = prob
        self.left = left
        self.right = right
        self.null_direction = null_direction

# CollectionNode class
class CollectionNode:
    def __init__(self, row_indices=None, depth=None, parent_index=None, node_index=None, child_direction=None, parent_indices= None):
        """
        Initialize a CollectionNode.
        :param row_indices: Indices of rows in the current node.
        :param depth: Depth of the node in the tree.
        :param parent_sorted_indices: Sorted indices of parent node.
        :param parent_null_indices: Null value indices of parent node.
        :param parent_index: Index of the parent node.
        :param node_index: Index of the current node.
        :param child_direction: Direction ('left' or 'right') from the parent.
        """
        self.row_indices = row_indices
        self.depth = depth
        self.parent_index = parent_index
        self.node_index = node_index
        self.child_direction = child_direction
        self.parent_indices = parent_indices

# BestStatus Class
class BestStatus:
    def __init__(self, best_feature=None, best_index=0, best_metric=0, best_threshold=None, best_null_direction=None):
        """
        Initialize BestStatus.
        :param best_feature: Feature index with the best split.
        :param best_index: Index of the best split.
        :param best_metric: Metric value of the best split.
        :param best_threshold: Threshold value for the best split.
        :param best_null_direction: Direction for handling null values in the best split.
        """
        self.best_feature = best_feature
        self.best_index = best_index
        self.best_metric = best_metric
        self.best_threshold = best_threshold
        self.best_null_direction = best_null_direction

    def _renew_best_status(self, metrics, filtered_sorted_indices, data):
        """
        Update the best status based on metrics.
        :param metrics: Metric values for each feature and threshold.
        :param filtered_sorted_indices: Filtered sorted indices for each feature.
        :param data: Feature data.
        """
        self.best_metric = np.max(metrics)
        self.best_index, self.best_feature = np.unravel_index(metrics.argmax(), metrics.shape)
        self.left_indices, self.right_indices = BestStatus._get_left_right_indices(filtered_sorted_indices,
                                                                                   self.best_index,
                                                                                   self.best_feature)
        sorted_best_feature_data = data[:, self.best_feature][filtered_sorted_indices[:, self.best_feature]]
        sorted_best_feature_data = np.ascontiguousarray(sorted_best_feature_data)
        self.best_threshold = np.round((sorted_best_feature_data[self.best_index] +
                                        sorted_best_feature_data[self.best_index + 1]) / 2, 7)
        
    @staticmethod
    @njit
    def _get_left_right_indices(full_sorted_indices, best_index, best_feature):
        """
        Get indices for the left and right splits.
        :param full_sorted_indices: Sorted indices of all features.
        :param best_index: Index of the best split point.
        :param best_feature: Index of the best feature to split.
        :return: Left and right split indices.
        """
        left_indices = full_sorted_indices[:(best_index + 1), best_feature]
        right_indices = full_sorted_indices[(best_index + 1):, best_feature]
        return np.ascontiguousarray(left_indices), np.ascontiguousarray(right_indices)

    def _renew_best_status_null(self, metrics, sorted_data, feature_index, null_direction):
        """
        Update the best status considering null values.
        :param metrics: Metric values for each feature and threshold.
        :param filtered_sorted_indices: Filtered sorted indices for each feature.
        :param sorted_data: Sorted feature data.
        :param feature_index: Feature index being considered.
        :param null_direction: Direction for handling null values ('left' or 'right').
        """
        metrics_max = metrics.max()
        thresholds = (sorted_data[:-1] + sorted_data[1:]) / 2

        if metrics_max > 0 and metrics_max > self.best_metric:
            self.best_metric = metrics_max
            self.best_feature = feature_index
            self.best_index = metrics.argmax()
            self.best_threshold = thresholds[self.best_index]
            self.best_null_direction = null_direction

    def _finalize_best_status_null(self, parent_indices, selected_labels):
        """
        Finalize the best status considering null values.
        :param filtered_sorted_indices: Filtered sorted indices for each feature.
        :param filtered_null_indices: Null value indices for each feature.
        :param selected_labels: Labels for the selected data.
        :return: True if no valid split is found, otherwise False.
        """
        null_indices = parent_indices.parent_null_indices[self.best_feature]
        null_shape = null_indices.shape[0]

        nonecase1 = ((self.best_index <= null_shape) and (self.best_null_direction == 'left'))
        nonecase2 = ((self.best_index >= selected_labels.shape[0] - null_shape) and (self.best_null_direction == 'right'))

        if nonecase1 or nonecase2 or self.best_threshold <= 0:
            self.best_feature = None
            self.best_threshold = None
            self.best_metric = None
            self.best_null_direction = None
            self.left_indices = None
            self.right_indices = None
            return True

        if self.best_null_direction == 'left':
            left_sorted_indices = parent_indices.parent_sorted_indices[self.best_feature][:(self.best_index - null_shape + 1)]
            self.left_indices = np.concatenate([left_sorted_indices, null_indices])
            self.right_indices = parent_indices.parent_sorted_indices[self.best_feature][(self.best_index - null_shape + 1):]
        else:
            self.left_indices = parent_indices.parent_sorted_indices[self.best_feature][:(self.best_index + 1)]
            self.right_indices = np.concatenate([parent_indices.parent_sorted_indices[self.best_feature][(self.best_index + 1):], null_indices])
            
        return False

class EarlyStopper:
    def __init__(self, val_data, val_labels, early_stop_rounds, current_max_depth=0, currert_early_stop_rounds=0):
        """
        Initialize EarlyStopper.
        :param val_data: Validation feature data.
        :param val_labels: Validation labels.
        :param early_stop_rounds: Number of rounds for early stopping.
        :param current_max_depth: Current maximum depth of the tree.
        :param currert_early_stop_rounds: Current number of early stop rounds without improvement.
        """
        self.val_data = val_data
        self.val_labels = val_labels
        self.early_stop_rounds = early_stop_rounds
        self.best_auc = 0
        self.current_max_depth = current_max_depth
        self.currert_early_stop_rounds = currert_early_stop_rounds

    def _evaluate_early_stop(self, decisiontreezhoumath, current_node, tree):
        """
        Evaluate whether to trigger early stopping.
        :param decisiontreezhoumath: Instance of the decision tree.
        :param current_node: Current node being processed.
        :param tree: Current tree.
        :return: True if early stopping should be triggered, otherwise False.
        """
        labels_pred = decisiontreezhoumath.predict_proba(decisiontreezhoumath.data, tree)[:, 1]
        val_labels_pred = decisiontreezhoumath.predict_proba(self.val_data, tree)[:, 1]
        train_auc = roc_auc_score(decisiontreezhoumath.labels, labels_pred)
        val_auc = roc_auc_score(self.val_labels, val_labels_pred)
        print(f'Current depth: {self.current_max_depth}, current train AUC: {train_auc:.3f}, current val AUC: {val_auc:.3f}')
        self.current_max_depth = current_node.depth

        if val_auc > self.best_auc:
            self.best_auc = val_auc
            decisiontreezhoumath.best_tree = tree.copy()
            self.currert_early_stop_rounds = 0
        else:
            self.currert_early_stop_rounds += 1

        if self.currert_early_stop_rounds >= self.early_stop_rounds:
            print(f'Early stop triggered at depth {self.current_max_depth - 1}')
            return True

        return False

class ParentIndices:
    def __init__(self, parent_sorted_indices = None, parent_null_indices = None):
        """
        Initialize ParentIndices to manage parent node indices for sorted and null values.
        :param parent_sorted_indices: Sorted indices of the parent node.
        :param parent_null_indices: Null value indices of the parent node.
        """
        self.parent_sorted_indices = parent_sorted_indices
        self.parent_null_indices = parent_null_indices
        
    def _filter_sorted_indices(self, row_indices):
        """
        Retrieve the sorted indices for the given rows.
        :param current_node: Current node being processed.
        :return: Filtered sorted indices for all features.
        """
        sorted_indices_flattened = ParentIndices._flatten_sorted_indices(self.parent_sorted_indices)
        filtered_sorted_indices_flattened = sorted_indices_flattened[np.in1d(sorted_indices_flattened, row_indices)]
        filtered_sorted_indices = ParentIndices._unflatten_sorted_indices(self.parent_sorted_indices,filtered_sorted_indices_flattened)
        filtered_indices = ParentIndices(parent_sorted_indices=np.ascontiguousarray(filtered_sorted_indices))
        return filtered_indices
    
    @staticmethod
    @njit
    def _flatten_sorted_indices(parent_sorted_indices):
        """
        Flatten the sorted indices for easier processing.
        :param parent_sorted_indices: Sorted indices of the parent node.
        :return: Flattened sorted indices.
        """
        parent_sorted_indices_t = parent_sorted_indices.T
        parent_sorted_indices_t = np.ascontiguousarray(parent_sorted_indices_t)
        return parent_sorted_indices_t.reshape(-1)

    @staticmethod
    @njit
    def _unflatten_sorted_indices(parent_sorted_indices, filtered_sorted_indices_flattened):
        """
        Unflatten the sorted indices to match the original shape.
        :param parent_sorted_indices: Original sorted indices.
        :param filtered_sorted_indices_flattened: Flattened sorted indices to unflatten.
        :return: Unflattened sorted indices.
        """
        filtered_sorted_indices_flattened = np.ascontiguousarray(filtered_sorted_indices_flattened)
        return filtered_sorted_indices_flattened.reshape((parent_sorted_indices.shape[1], -1)).T

    def _init_root_indices_null(self, data):
        """
        Initialize root indices for handling missing values.
        :param data: Feature data.
        """
        self.parent_null_indices = np.array([np.where(np.isnan(data[:, i]))[0] for i in range(data.shape[1])], dtype=object)
        null_quantities = np.array([self.parent_null_indices[i].shape[0] for i in range(self.parent_null_indices.shape[0])])
        data_no_nan = np.where(np.isnan(data), np.inf, data)
        root_sorted_indices_with_null = np.argsort(data_no_nan, axis=0)
        self.parent_sorted_indices = np.array([
            root_sorted_indices_with_null[:-null_quantities[i], i] for i in range(self.parent_null_indices.shape[0])
        ], dtype=object)

    def _filter_sorted_indices_null(self, row_indices):
        """
        Retrieve the sorted indices for the given rows, considering missing values.
        :param row_indices: Indices of rows in the current node.
        :return: Filtered sorted and null indices for all features.
        """
        filtered_sorted_indices = []
        filtered_null_indices = []

        for i in range(self.parent_sorted_indices.shape[0]):
            parent_sorted_indices_i = self.parent_sorted_indices[i]
            filtered_sorted_indices_i = parent_sorted_indices_i[np.in1d(parent_sorted_indices_i, row_indices)]
            filtered_sorted_indices.append(np.ascontiguousarray(filtered_sorted_indices_i))
            parent_null_indices_i = self.parent_null_indices[i]
            filtered_null_indices_i = parent_null_indices_i[np.in1d(parent_null_indices_i, row_indices)]
            filtered_null_indices.append(np.ascontiguousarray(filtered_null_indices_i))
            
        filtered_indices = ParentIndices(parent_sorted_indices=np.array(filtered_sorted_indices, dtype=object),
                                         parent_null_indices=np.array(filtered_null_indices, dtype=object))

        return filtered_indices
    
class FeatureImportances:
    def __init__(self, num_features):
        """
        Initialize FeatureImportances to track feature split and gain importances.
        :param num_features: Number of features in the dataset.
        """
        self.split_importance = np.zeros(num_features)
        self.gain_importance = np.zeros(num_features)
    
    def _renew_feature_importances(self, beststatus):
        """
        Update feature importance values based on the current best split.
        :param beststatus: Current best split status.
        """
        if beststatus.best_feature is not None:
            self.split_importance[beststatus.best_feature] += 1
            self.gain_importance[beststatus.best_feature] += beststatus.best_metric
    
    def get_feature_importances_df(self, col_names = None):
        """
        Get feature importances as a DataFrame.
        :param col_names: List of column names for the features.
        :return: DataFrame containing split and gain importances for each feature.
        """
        feature_importances_df = pd.DataFrame()
        feature_importances_df['split_importance'] = self.split_importance
        feature_importances_df['gain_importance'] = self.gain_importance
        
        if col_names is not None:
            feature_importances_df.index = col_names
            
        return feature_importances_df
