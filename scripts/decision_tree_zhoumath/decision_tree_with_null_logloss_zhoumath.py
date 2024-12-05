# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:09:39 2024

@author: zhoushus
"""

# Import packages
import warnings
import numpy as np
from numba import njit
from decision_tree_logloss_zhoumath import DecisionTreeLoglossZhoumath
from decision_tree_helper_zhoumath import CollectionNode, ParentIndices, BestStatus

# Settings
warnings.filterwarnings("ignore", category=UserWarning)

# DecisionTreeWithNullZhoumath extends DecisionTreeZhoumath
class DecisionTreeWithNullLoglossZhoumath(DecisionTreeLoglossZhoumath):    
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
        filtered_indices, selected_labels, selected_hessians, base_metric = self._init_best_split(current_node, True)
        
        if self.lambda_l1 > 0:
            sum_labels_l1 = np.max([np.abs(np.sum(selected_labels)) - self.lambda_l1, 0])
            base_metric = base_metric * (sum_labels_l1 ** 2 / (np.sum(selected_labels) ** 2))

        num_features = self.data.shape[1]
        current_best_status = BestStatus()
        
        if self.random_column_rate < 1:
            muted_column_nums = np.floor(num_features * (1 - self.random_column_rate)).astype(np.int32)
            muted_columns = np.random.choice(np.arange(num_features), size=muted_column_nums, replace=False).astype(np.int32)
        
        current_best_status = self._iterate_features(num_features, current_best_status, filtered_indices, base_metric, muted_columns)
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
                sorted_hessians = np.ascontiguousarray(self.hessians[sorted_indices])
                
                if self.lambda_l1 > 0:
                    metrics = DecisionTreeWithNullLoglossZhoumath._calculate_metrics(sorted_labels, sorted_hessians,
                                                                                     base_metric, self.lambda_l2)
                else:
                    metrics = DecisionTreeWithNullLoglossZhoumath._calculate_metrics_l1(sorted_labels, sorted_hessians,
                                                                                        base_metric, self.lambda_l1,
                                                                                        self.lambda_l2)
                    
                current_best_status._renew_best_status_null(metrics, sorted_data, feature_index, null_direction)
                
        return current_best_status

    @staticmethod
    @njit
    def _calculate_metrics(sorted_labels, sorted_hessians, base_metric, lambda_l2):
        """
        Calculate information gain for potential split thresholds considering missing values.
        :param sorted_labels: Labels sorted by the feature values.
        :param base_metric: Base metric before the split.
        :param pos_weight: Weight for positive class.
        :param gini: Indicator for Gini impurity.
        :return: Information gain for each potential threshold.
        """
        left_cumsum = np.cumsum(sorted_labels)[:-1]
        left_cumsum = np.ascontiguousarray(left_cumsum)
        right_cumsum = sorted_labels.sum() - left_cumsum
        left_cumsum_square = np.square(left_cumsum)
        right_cumsum_square = np.square(right_cumsum)
        left_hessians_cumsum = np.cumsum(sorted_hessians)[:-1]
        right_hessians_cumsum = np.sum(sorted_hessians) - left_hessians_cumsum
        left_loss = -0.5 * (left_cumsum_square / (left_hessians_cumsum + lambda_l2))
        right_loss = -0.5 * (right_cumsum_square / (right_hessians_cumsum + lambda_l2))
        sum_loss = left_loss + right_loss
        return base_metric - sum_loss
    
    @staticmethod
    def _calculate_metrics_l1(sorted_labels, sorted_hessians, base_metric, lambda_l1, lambda_l2):
        """
        Calculate information gain for potential split thresholds considering missing values.
        :param sorted_labels: Labels sorted by the feature values.
        :param base_metric: Base metric before the split.
        :param pos_weight: Weight for positive class.
        :param gini: Indicator for Gini impurity.
        :return: Information gain for each potential threshold.
        """
        left_cumsum = np.cumsum(sorted_labels)[:-1]
        left_cumsum = np.ascontiguousarray(left_cumsum)
        left_cumsum_l1 = np.max([np.abs(left_cumsum) - lambda_l1, np.zeros(left_cumsum.shape)], axis = 0)
        right_cumsum = sorted_labels.sum() - left_cumsum
        right_cumsum_l1 = np.max([np.abs(right_cumsum) - lambda_l1, np.zeros(right_cumsum.shape)], axis = 0)
        left_cumsum_square = np.square(left_cumsum_l1)
        right_cumsum_square = np.square(right_cumsum_l1)
        left_hessians_cumsum = np.cumsum(sorted_hessians)[:-1]
        right_hessians_cumsum = np.sum(sorted_hessians) - left_hessians_cumsum
        left_loss = -0.5 * (left_cumsum_square / (left_hessians_cumsum + lambda_l2))
        right_loss = -0.5 * (right_cumsum_square / (right_hessians_cumsum + lambda_l2))
        sum_loss = left_loss + right_loss
        return base_metric - sum_loss

