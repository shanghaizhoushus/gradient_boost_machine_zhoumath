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
    def fit(self, data, labels, val_data, val_labels, early_stop_rounds, random_state=42):
        """
        Train the decision tree.
        :param data: Feature data.
        :param labels: Labels.
        :param val_data: Validation feature data.
        :param val_labels: Validation labels.
        :param early_stop_rounds: Number of rounds for early stopping.
        :param random_state: Random seed for reproducibility.
        """
        self._fitting(data, labels, val_data, val_labels, early_stop_rounds, random_state)
    
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
        current_best_status = BestStatus()
        num_features = self.data.shape[1]

        if current_node.depth == 0:
            filtered_indices = current_node.parent_indices
        else:
            filtered_indices = current_node.parent_indices._filter_sorted_indices_null(current_node.row_indices)
        
        selected_labels = np.ascontiguousarray(self.labels[current_node.row_indices])
        base_metric = DecisionTreeZhoumath._calculate_base_metric(selected_labels, self.pos_weight, self.gini)
        
        intrinsic_value = None
        if self.split_criterion == 'entropy_gain_ratio':
            intrinsic_value = DecisionTreeZhoumath._calculate_intrinsic_value(selected_labels)
            
        current_best_status = self._iterate_features(num_features, current_best_status, filtered_indices, base_metric, intrinsic_value)
        no_best_split = current_best_status._finalize_best_status_null(filtered_indices, selected_labels)

        if no_best_split:
            filtered_indices = ParentIndices()
            
        self.feature_importances._renew_feature_importances(current_best_status)

        return current_best_status, filtered_indices
    
    def _iterate_features(self, num_features, current_best_status, filtered_indices, base_metric, intrinsic_value):
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
                metrics = DecisionTreeWithNullZhoumath._calculate_metrics(sorted_labels, base_metric, self.pos_weight, self.gini)

                if self.split_criterion == 'entropy_gain_ratio':
                    intrinsic_value = intrinsic_value.reshape(-1)
                    metrics = metrics / intrinsic_value
                    
                current_best_status._renew_best_status_null(metrics, sorted_data, feature_index, null_direction)
                
        return current_best_status

    @staticmethod
    @njit
    def _calculate_metrics(sorted_labels, base_metric, pos_weight, gini):
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

        if gini == 0:
            left_entropy = -left_zero_rate * np.log2(left_zero_rate + 1e-9) - pos_weight *  left_one_rate * np.log2(left_one_rate + 1e-9)
            right_entropy = -right_zero_rate * np.log2(right_zero_rate + 1e-9) - pos_weight * right_one_rate * np.log2(right_one_rate + 1e-9)
            weighted_entropy = left_probs * left_entropy + right_probs * right_entropy
            return base_metric - weighted_entropy
        else:
            left_gini = 1 - left_zero_rate ** 2 - left_one_rate ** 2
            right_gini = 1 - right_zero_rate ** 2 - right_one_rate ** 2
            weighted_gini = left_probs * left_gini + right_probs * right_gini
            return base_metric - weighted_gini
