# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:09:39 2024

@author: zhoushus
"""

# Import packages
import warnings
import numpy as np
from numba import njit
from decision_tree_zhoumath import DecisionTreeZhoumath
from decision_tree_helper_zhoumath import CollectionNode, ParentIndices, BestStatus

# Settings
warnings.filterwarnings("ignore", category=UserWarning)

# DecisionTreeWithNullZhoumath extends DecisionTreeZhoumath
class DecisionTreeWithNullZhoumath(DecisionTreeZhoumath):    
    def _init_root_collection_node(self):
        """
        Initialize the root collection node for the decision tree.
        :return: Initialized root collection node.
        """
        root_indices = ParentIndices()
        root_indices._init_root_indices_null(self.data)
        root_collection_node = CollectionNode(depth=0,
                                              row_indices=np.arange(self.labels.shape[0]),
                                              parent_indices=root_indices)

        return root_collection_node
    

    def _choose_best_split(self, current_node):
        """
        Determine the best split for the current node, considering missing values.
        :param current_node: Current node being processed.
        :return: Best split status, filtered sorted indices, and filtered null indices.
        """
        
        filtered_indices, selected_labels, base_metric = self._init_best_split(current_node, self.pos_weight, True)
        intrinsic_value = None
        
        if self.split_criterion == 'entropy_gain_ratio':
            intrinsic_value = DecisionTreeZhoumath._calculate_intrinsic_value(selected_labels)
            
        num_features = self.data.shape[1]
        current_best_status = BestStatus()
        
        if self.random_column_rate < 1:
            muted_column_nums = np.floor(num_features * (1 - self.random_column_rate)).astype(np.int32)
            muted_columns = np.random.choice(np.arange(num_features), size=muted_column_nums, replace=False).astype(np.int32)
        
        current_best_status = self._iterate_features(num_features, current_best_status, filtered_indices, base_metric, intrinsic_value, muted_columns)
        no_best_split = current_best_status._finalize_best_status_null(filtered_indices, selected_labels)

        if no_best_split:
            filtered_indices = ParentIndices()
            
        self.feature_importances._renew_feature_importances(current_best_status)

        return current_best_status, filtered_indices
    
    def _iterate_features(self, num_features, current_best_status, filtered_indices, base_metric, intrinsic_value, muted_columns):
        """
        Iterate over features to determine the best split considering missing values.
        :param num_features: Number of features in the dataset.
        :param current_best_status: Current best split status.
        :param filtered_indices: Filtered indices containing sorted and null indices.
        :param base_metric: Base metric before the split.
        :param intrinsic_value: Intrinsic value for entropy gain ratio.
        :return: Updated best split status.
        """
        for feature_index in range(num_features):
            if feature_index in muted_columns:
                continue
            
            for null_direction in ['left', 'right']:
                sorted_indices = filtered_indices.parent_sorted_indices[feature_index]
                null_indices = filtered_indices.parent_null_indices[feature_index]
                if null_direction == 'left':
                    sorted_indices = np.concatenate([null_indices, sorted_indices])
                    sorted_data = np.ascontiguousarray(self.data[sorted_indices, feature_index])
                    sorted_data = np.nan_to_num(sorted_data, nan=-np.inf)
                else:
                    sorted_indices = np.concatenate([sorted_indices, null_indices])
                    sorted_data = np.ascontiguousarray(self.data[sorted_indices, feature_index])
                    sorted_data = np.nan_to_num(sorted_data, nan=np.inf)
                
                sorted_labels = np.ascontiguousarray(self.labels[sorted_indices])
                
                if self.split_criterion == 'mse':
                    if self.pos_weight != 1 and self.task == 'classification':
                        sorted_labels[sorted_labels == 1] = self.pos_weight
                    metrics = DecisionTreeZhoumath._calculate_metrics_mse(sorted_labels, base_metric)
                else:
                    metrics = DecisionTreeZhoumath._calculate_metrics(sorted_labels, base_metric, self.pos_weight, self.split_criterion)
                    
                if self.split_criterion == 'entropy_gain_ratio':
                    intrinsic_value = intrinsic_value.reshape(-1)
                    metrics = metrics / intrinsic_value
                    
                current_best_status._renew_best_status_null(metrics, sorted_data, feature_index, null_direction)
                
        return current_best_status

    @staticmethod
    @njit
    def _calculate_metrics(sorted_labels, base_metric, pos_weight, split_criterion):
        """
        Calculate information gain for potential split thresholds considering missing values.
        :param sorted_labels: Labels sorted by the feature values.
        :param base_metric: Base metric before the split.
        :param pos_weight: Weight for positive class.
        :param gini: Indicator for Gini impurity.
        :return: Information gain for each potential threshold.
        """
        num_rows = sorted_labels.shape[0]
        left_cumsum = np.cumsum(sorted_labels)[:-1]
        left_cumsum = np.ascontiguousarray(left_cumsum)
        right_cumsum = sorted_labels.sum() - left_cumsum
        left_one_rate = left_cumsum / np.arange(1, num_rows)
        right_one_rate = right_cumsum / np.arange(num_rows - 1, 0, -1)
        left_zero_rate = 1 - left_one_rate
        right_zero_rate = 1 - right_one_rate
        left_zero_rate = left_zero_rate / (left_zero_rate + pos_weight * left_one_rate)
        right_zero_rate = right_zero_rate / (right_zero_rate + pos_weight * right_one_rate)
        left_one_rate = 1 - left_zero_rate
        right_one_rate = 1 - right_zero_rate
        left_probs = (np.arange(1, num_rows) / num_rows)
        right_probs = (np.arange(num_rows - 1, 0, -1) / num_rows)

        if split_criterion == 'entropy_gain' or split_criterion == 'entropy_gain_ratio':
            left_entropy = -left_zero_rate * np.log2(left_zero_rate + 1e-9) - pos_weight *  left_one_rate * np.log2(left_one_rate + 1e-9)
            right_entropy = -right_zero_rate * np.log2(right_zero_rate + 1e-9) - pos_weight * right_one_rate * np.log2(right_one_rate + 1e-9)
            weighted_entropy = left_probs * left_entropy + right_probs * right_entropy
            return base_metric - weighted_entropy
        elif split_criterion == 'gini':
            left_gini = 1 - left_zero_rate ** 2 - left_one_rate ** 2
            right_gini = 1 - right_zero_rate ** 2 - right_one_rate ** 2
            weighted_gini = left_probs * left_gini + right_probs * right_gini
            return base_metric - weighted_gini
        
    @staticmethod
    @njit
    def _calculate_metrics_mse(sorted_labels, base_metric, pos_weight):
        """
        Calculate the information gain (for entropy) or Gini impurity (for Gini) 
        for each potential split threshold.
        The method computes the weighted impurity (either Gini or entropy) after 
        splitting the data based on each possible threshold.
        :param sorted_labels: Labels sorted by the feature values.
        :param base_metric: The base metric (Gini or entropy) before the split.
        :param pos_weight: Weight for the positive class.
        :param gini: Indicator for Gini impurity (if 0, calculates Gini; if 1, calculates entropy).
        :return: Information gain or Gini impurity reduction for each potential split threshold.
        """
        if pos_weight != 1:
            sorted_labels[sorted_labels == 1] = pos_weight
            
        num_rows = sorted_labels.shape[0]
        left_cumsum = np.cumsum(sorted_labels)[:-1]
        left_cumsum = np.ascontiguousarray(left_cumsum)
        left_square_cumsum = np.cumsum(sorted_labels ** 2)[:-1]
        left_cumsum_square = left_cumsum ** 2
        left_se = left_square_cumsum - left_cumsum_square / np.arange(1, num_rows)
        right_cumsum = np.sum(sorted_labels) - left_cumsum
        right_square_cumsum = np.sum(np.square(sorted_labels)) - left_square_cumsum
        right_cumsum_square = right_cumsum ** 2
        right_se = right_square_cumsum - right_cumsum_square / np.arange(num_rows-1, 0, -1)
        weighted_se = right_se + left_se
        weighted_mse = weighted_se / (num_rows)
        return base_metric - weighted_mse
