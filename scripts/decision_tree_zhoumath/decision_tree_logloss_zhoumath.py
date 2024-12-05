# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 22:49:51 2024

@author: zhoushus
"""

# Import packages
import warnings
import numpy as np
from collections import deque
from numba import njit
from decision_tree_zhoumath import DecisionTreeZhoumath

# Settings
warnings.filterwarnings("ignore", category=UserWarning)

# DecisionTree class
class DecisionTreeLoglossZhoumath(DecisionTreeZhoumath):
    def __init__(self, task, split_criterion, search_method, max_depth=None, pos_weight=1,
                 random_column_rate=1, min_split_sample_rate=0, min_leaf_sample_rate=0,
                 lambda_l1 = 0, lambda_l2=0, verbose=True):
        """
        Initialize the decision tree.
        :param split_criterion: Criterion for splitting ("entropy_gain", "entropy_gain_ratio", or "gini").
        :param search_method: Search method to split the tree ("bfs" or "dfs").
        :param max_depth: Maximum depth of the tree.
        :param pos_weight: Weight for positive class.
        """
        super().__init__(task, split_criterion, search_method, max_depth, pos_weight, random_column_rate,
                     min_split_sample_rate, min_leaf_sample_rate, verbose)
        
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
    
    def fit(self, data, labels, hessians, val_data=None, val_labels=None, val_hessians=None,
            early_stop_rounds=1, random_state=42):
        """
        Train the decision tree.
        :param data: Feature data.
        :param labels: Labels.
        :param hessians: Hessian values for gradient boosting.
        :param val_data: Validation feature data for early stopping.
        :param val_labels: Validation labels for early stopping.
        :param early_stop_rounds: Number of rounds for early stopping without improvement.
        :param random_state: Random seed for reproducibility.
        :return: None. The tree is trained in-place and stored in `self.tree`.
        """
        from decision_tree_with_null_logloss_zhoumath import DecisionTreeWithNullLoglossZhoumath

        if np.any(np.equal(data, np.nan)):
            tree_with_null = DecisionTreeWithNullLoglossZhoumath(
                split_criterion=self.split_criterion,
                search_method=self.search_method,
                max_depth=self.max_depth
            )
            tree_with_null.fitting(data, labels, hessians, val_data, val_labels, val_hessians, early_stop_rounds, random_state)
            self.tree = tree_with_null.tree
            self.feature_importances = tree_with_null.feature_importances
        
        else:
            self._fitting(data, labels, hessians, val_data, val_labels, val_hessians, early_stop_rounds, random_state)
        
    def _fitting(self, data, labels, hessians, val_data, val_labels, val_hessians, early_stop_rounds, random_state):
        """
        Fit the decision tree by processing the training data and optionally enabling early stopping.
        :param data: Training feature data.
        :param labels: Training labels.
        :param hessians: Hessians for gradient boosting.
        :param val_data: Validation feature data (used for early stopping).
        :param val_labels: Validation labels (used for early stopping).
        :param early_stop_rounds: Number of rounds without improvement before early stopping.
        :param random_state: Random seed for reproducibility.
        """
        from decision_tree_helper_zhoumath import FeatureImportances, EarlyStopperLogloss, CategorialModule
        
        data = np.ascontiguousarray(data)
        data = DecisionTreeZhoumath._add_perturbation(data, random_state)
        labels = np.ascontiguousarray(labels)
        hessians = np.ascontiguousarray(hessians)
        self.categorial_module = CategorialModule(np.zeros((2,2)))
        early_stopper = None
        
        if early_stop_rounds and self.search_method != 'bfs':
            raise ValueError("Early Stopping requires 'bfs' as the search method.")
        
        if (val_data is not None) and (val_labels is not None) and (early_stop_rounds is not None):
            if self.verbose:
                print("Early stop mode is opened. Search method can only be BFS.")
                
            val_data = np.ascontiguousarray(val_data)
            val_labels = np.ascontiguousarray(val_labels)
            val_hessians = np.ascontiguousarray(val_hessians)
            early_stopper = EarlyStopperLogloss(val_data=val_data,
                                                val_labels=val_labels,
                                                early_stop_rounds=early_stop_rounds,
                                                verbose=self.verbose)
            self.search_method = 'bfs'
            self.best_tree = []
        else:
            early_stopper = None
            
        self.data = data
        self.labels = labels
        self.hessians = hessians
        self.feature_importances = FeatureImportances(data.shape[1])
        self.feature_importances_cache = FeatureImportances(data.shape[1])
        self.tree = self._build_tree(early_stopper)
        self.data = None
        self.labels = None
        self.hessians = None
        
        if early_stopper is None:
            self.feature_importances = self.feature_importances_cache
            
    @staticmethod
    def _add_perturbation(data, random_state):
        """
        Add a small perturbation to the data to avoid numerical issues.
        :param data: Feature data.
        :param random_state: Random seed for reproducibility.
        :return: Perturbed data.
        """
        np.random.seed(random_state)
        perturbation = np.random.uniform(-1, 1, size=data.shape) * 1e-7
        return data + perturbation

    def _build_tree(self, early_stopper):
        """
        Recursively build the decision tree by evaluating splits at each node.
        The tree is constructed in a depth-first or breadth-first manner depending 
        on the search method.
        :param early_stopper: Instance of EarlyStopper for early stopping (optional).
        :return: A decision tree, represented as a list of tree nodes (TreeNode or CollectionNode).
        """
        from decision_tree_helper_zhoumath import CollectionNode, TreeNode
        
        root_collection_node = self._init_root_collection_node()

        if self.search_method == 'dfs':
            collection = [root_collection_node]
            pop_method = collection.pop
        else:
            collection = deque([root_collection_node])
            pop_method = collection.popleft

        tree = []

        while collection:
            current_node = pop_method()
            current_node.node_index = len(tree)
            labels = np.ascontiguousarray(self.labels[current_node.row_indices])
            hessians = np.ascontiguousarray(self.hessians[current_node.row_indices])

            if early_stopper is not None and current_node.depth > early_stopper.current_max_depth:
                early_stop_triggered = early_stopper._evaluate_early_stop(self, current_node, tree)
                
                if early_stop_triggered:
                    return self.best_tree.copy()
                
            min_split_sample_rate_condition = current_node.row_indices.shape[0] < (self.labels.shape[0] * self.min_split_sample_rate)
            
            if (self.max_depth is not None and current_node.depth >= self.max_depth) or np.unique(labels).size == 1 or min_split_sample_rate_condition:
                DecisionTreeLoglossZhoumath._finallize_node(labels, hessians, self.lambda_l1, self.lambda_l2, tree, current_node)
                continue

            current_best_status, filtered_indices = self._choose_best_split(current_node)
            min_right_leaf_sample_rate_condition = (current_best_status.right_indices.shape[0]) < (self.labels.shape[0] * self.min_leaf_sample_rate)
            min_left_leaf_sample_rate_condition = (current_best_status.left_indices.shape[0]) < (self.labels.shape[0] * self.min_leaf_sample_rate)
            min_leaf_sample_rate_condition = min_right_leaf_sample_rate_condition or min_left_leaf_sample_rate_condition

            if current_best_status.best_feature is None or current_best_status.best_metric <= 0 or min_leaf_sample_rate_condition:
                DecisionTreeLoglossZhoumath._finallize_node(labels, hessians, self.lambda_l1, self.lambda_l2, tree, current_node)
                continue

            current_tree_node = TreeNode(feature=current_best_status.best_feature,
                                         threshold=current_best_status.best_threshold,
                                         null_direction=current_best_status.best_null_direction)

            if early_stopper is not None:
                sum_labels = np.sign(np.sum(labels)) * np.max([np.abs(np.sum(labels)) - self.lambda_l1, 0])
                current_tree_node.prob = sum_labels / (np.sum(hessians) + self.lambda_l2)
                
            DecisionTreeZhoumath._add_node_to_tree(tree, current_node, current_tree_node)
            right_node = CollectionNode(row_indices=current_best_status.right_indices,
                                        depth=current_node.depth + 1,
                                        parent_indices=filtered_indices,
                                        parent_index=current_node.node_index,
                                        child_direction="right")
            left_node = CollectionNode(row_indices=current_best_status.left_indices,
                                       depth=current_node.depth + 1,
                                       parent_indices=filtered_indices,
                                       parent_index=current_node.node_index,
                                       child_direction="left")
            collection.append(right_node)
            collection.append(left_node)

        if early_stopper is not None:
            current_node.depth += 1
            early_stop_triggered = early_stopper._evaluate_early_stop(self, current_node, tree)
            if early_stop_triggered:
                return self.best_tree.copy()

        return tree if early_stopper is None else self.best_tree.copy()
    
    @staticmethod
    def _finallize_node(labels, hessians, lambda_l1, lambda_l2, tree, current_node):
        """
        Finalize a node by calculating its probability and adding it to the tree.
        :param labels: Labels for the current node.
        :param hessians: Hessians for gradient boosting.
        :param tree: The decision tree (list of nodes).
        :param current_node: The current collection node being processed.
        """
        from decision_tree_helper_zhoumath import TreeNode
        
        sum_labels = np.sign(np.sum(labels)) * np.max([np.abs(np.sum(labels)) - lambda_l1, 0])
        current_tree_node = TreeNode(prob = sum_labels / np.sum(hessians) + lambda_l2)
        DecisionTreeZhoumath._add_node_to_tree(tree, current_node, current_tree_node)

    def _choose_best_split(self, current_node):
        """
        Evaluate and determine the best feature and threshold to split the current node.
        This involves calculating metrics for each feature and threshold, and selecting 
        the one that maximizes the chosen split criterion (e.g., log loss reduction).
        If the `random_column_rate` is less than 1, a random subset of features is considered 
        for the split.
        :param current_node: The current collection node being processed.
        :return: Best split status (containing best feature and threshold) and filtered indices.
        """
        from decision_tree_helper_zhoumath import BestStatus
        
        filtered_indices, filtered_sorted_labels, filtered_sorted_hessians, base_metric = self._init_best_split(current_node, False)
        
        if self.lambda_l1 > 0:
            labels = filtered_sorted_labels[:, 0]
            sum_labels_l1 = np.max([np.abs(np.sum(labels)) - self.lambda_l1, 0])
            base_metric = base_metric * (sum_labels_l1 ** 2 / (np.sum(labels) ** 2))
            metrics = DecisionTreeLoglossZhoumath._calculate_metrics_logloss_l1(filtered_sorted_labels, filtered_sorted_hessians,
                                                                                base_metric, self.lambda_l1, self.lambda_l2)
            
        else:
            metrics = DecisionTreeLoglossZhoumath._calculate_metrics_logloss(filtered_sorted_labels, filtered_sorted_hessians,
                                                                             base_metric, self.lambda_l2)

        num_features = filtered_sorted_labels.shape[1]
        
        if self.random_column_rate < 1:
            muted_column_nums = np.floor(num_features * (1 - self.random_column_rate)).astype(np.int32)
            muted_columns = np.random.choice(np.arange(num_features), size=muted_column_nums, replace=False).astype(np.int32)
            metrics[:,muted_columns] = -np.inf
            
        current_best_status = BestStatus()
        current_best_status._renew_best_status(metrics, filtered_indices.parent_sorted_indices, self.data)
        self.feature_importances_cache._renew_feature_importances(current_best_status)
        return current_best_status, filtered_indices
    
    def _init_best_split(self, current_node, isnull):
        """
        Initialize the best split search by preparing the filtered indices and sorted labels.
        :param current_node: The current collection node being processed. Contains row indices to filter data.
        :param isnull: Boolean indicating whether to consider null values during the split.
        :return: Filtered indices and sorted labels for the best split search, as well as the base metric for comparison.
        """
        if current_node.depth == 0:
            filtered_indices = current_node.parent_indices
        else:
            filtered_indices = current_node.parent_indices._filter_sorted_indices(current_node.row_indices, isnull)
        
        filtered_indices = current_node.parent_indices._filter_sorted_indices(current_node.row_indices, isnull)
        filtered_sorted_labels = np.ascontiguousarray(self.labels[filtered_indices.parent_sorted_indices])
        filtered_sorted_hessians = np.ascontiguousarray(self.hessians[filtered_indices.parent_sorted_indices])
        base_metric = DecisionTreeLoglossZhoumath._calculate_base_metric(filtered_sorted_labels[:, 0], filtered_sorted_hessians[:, 0], self.lambda_l2)
        return filtered_indices, filtered_sorted_labels, filtered_sorted_hessians, base_metric
        
    @staticmethod
    @njit
    def _calculate_base_metric(labels, hessians, lambda_l2):
        """
        Calculate the base metric for the given labels and hessians.
        :param labels: Labels.
        :param hessians: Hessians for gradient boosting.
        :return: Base metric value.
        """
        base_metric = -0.5 * ((np.sum(labels) ** 2)/(np.sum(hessians) + lambda_l2))
        return base_metric
        
    @staticmethod
    @njit
    def _calculate_metrics_logloss(sorted_labels, sorted_hessians, base_metric, lambda_l2):
        """
        Calculate the log loss reduction for each potential split threshold.
        The method computes the weighted log loss after splitting the data based on each possible threshold.
        :param sorted_labels: Labels sorted by the feature values.
        :param sorted_hessians: Hessians sorted by the feature values.
        :param base_metric: The base metric before the split.
        :return: Log loss reduction for each potential split threshold.
        """
        sorted_labels_t = sorted_labels.T
        sorted_labels_t = np.ascontiguousarray(sorted_labels_t)
        sorted_labels_total_cumsum = sorted_labels_t.reshape(-1).cumsum().reshape(sorted_labels_t.shape).T
        sorted_labels_total_cumsum = np.ascontiguousarray(sorted_labels_total_cumsum)
        left_cumsum = (sorted_labels_total_cumsum - np.concatenate((np.array([0], dtype=np.int64),
                                                                    sorted_labels_total_cumsum[-1, :-1])))[:-1, :]
        left_cumsum_square = np.square(left_cumsum)
        right_cumsum = np.sum(sorted_labels[:, 0]) - left_cumsum
        right_cumsum_square = np.square(right_cumsum)
        sorted_hessians_t = sorted_hessians.T
        sorted_hessians_t = np.ascontiguousarray(sorted_hessians_t)
        sorted_hessians_total_cumsum = sorted_hessians_t.reshape(-1).cumsum().reshape(sorted_hessians_t.shape).T
        sorted_hessians_total_cumsum = np.ascontiguousarray(sorted_hessians_total_cumsum)
        left_hessians_cumsum = (sorted_hessians_total_cumsum - np.concatenate((np.array([0], dtype=np.int64),
                                                                               sorted_hessians_total_cumsum[-1, :-1])))[:-1, :]
        right_hessians_cumsum = np.sum(sorted_hessians[:, 0]) - left_hessians_cumsum
        left_loss = -0.5 * (left_cumsum_square / (left_hessians_cumsum + lambda_l2))
        right_loss = -0.5 * (right_cumsum_square / (right_hessians_cumsum + lambda_l2))
        sum_loss = left_loss + right_loss
        return base_metric - sum_loss
    
    @staticmethod
    def _calculate_metrics_logloss_l1(sorted_labels, sorted_hessians, base_metric, lambda_l1, lambda_l2):
        """
        Calculate the log loss reduction for each potential split threshold.
        The method computes the weighted log loss after splitting the data based on each possible threshold.
        :param sorted_labels: Labels sorted by the feature values.
        :param sorted_hessians: Hessians sorted by the feature values.
        :param base_metric: The base metric before the split.
        :return: Log loss reduction for each potential split threshold.
        """
        sorted_labels_t = sorted_labels.T
        sorted_labels_t = np.ascontiguousarray(sorted_labels_t)
        sorted_labels_total_cumsum = sorted_labels_t.reshape(-1).cumsum().reshape(sorted_labels_t.shape).T
        sorted_labels_total_cumsum = np.ascontiguousarray(sorted_labels_total_cumsum)
        left_cumsum = (sorted_labels_total_cumsum - np.concatenate((np.array([0], dtype=np.int64),
                                                                    sorted_labels_total_cumsum[-1, :-1])))[:-1, :]
        left_cumsum_l1 = np.max([np.abs(left_cumsum) - lambda_l1, np.zeros(left_cumsum.shape)], axis = 0)
        left_cumsum_square = np.square(left_cumsum_l1)
        right_cumsum = np.sum(sorted_labels[:, 0]) - left_cumsum
        right_cumsum_l1 = np.max([np.abs(right_cumsum) - lambda_l1, np.zeros(right_cumsum.shape)], axis = 0)
        right_cumsum_square = np.square(right_cumsum_l1)
        sorted_hessians_t = sorted_hessians.T
        sorted_hessians_t = np.ascontiguousarray(sorted_hessians_t)
        sorted_hessians_total_cumsum = sorted_hessians_t.reshape(-1).cumsum().reshape(sorted_hessians_t.shape).T
        sorted_hessians_total_cumsum = np.ascontiguousarray(sorted_hessians_total_cumsum)
        left_hessians_cumsum = (sorted_hessians_total_cumsum - np.concatenate((np.array([0], dtype=np.int64),
                                                                               sorted_hessians_total_cumsum[-1, :-1])))[:-1, :]
        right_hessians_cumsum = np.sum(sorted_hessians[:, 0]) - left_hessians_cumsum
        left_loss = -0.5 * (left_cumsum_square / (left_hessians_cumsum + lambda_l2))
        right_loss = -0.5 * (right_cumsum_square / (right_hessians_cumsum + lambda_l2))
        sum_loss = left_loss + right_loss
        return base_metric - sum_loss
