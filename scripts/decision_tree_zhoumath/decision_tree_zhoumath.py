# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:09:39 2024

@author: zhoushus
"""

# Import packages
import warnings
import numpy as np
from collections import deque
from numba import njit
import pickle

# Settings
warnings.filterwarnings("ignore", category=UserWarning)

# DecisionTree class
class DecisionTreeZhoumath:
    def __init__(self, split_criterion, search_method, max_depth=None, pos_weight=1, random_column_rate=1,
                 min_split_sample_rate=0, min_leaf_sample_rate=0, verbose=True):
        """
        Initialize the decision tree.
        :param split_criterion: Criterion for splitting ("entropy_gain", "entropy_gain_ratio", or "gini").
        :param search_method: Search method to split the tree ("bfs" or "dfs").
        :param max_depth: Maximum depth of the tree.
        :param pos_weight: Weight for positive class.
        """
        valid_split_criteria = {"entropy_gain", "entropy_gain_ratio", "gini"}
        valid_search_methods = {"bfs", "dfs"}

        if split_criterion not in valid_split_criteria:
            raise ValueError(f"Invalid split criterion: {split_criterion}. Choose from {valid_split_criteria}.")

        if search_method not in valid_search_methods:
            raise ValueError(f"Invalid search method: {search_method}. Choose from {valid_search_methods}.")

        self.split_criterion = split_criterion
        self.search_method = search_method
        self.max_depth = max_depth
        self.pos_weight = pos_weight
        self.random_column_rate = random_column_rate
        self.min_split_sample_rate = min_split_sample_rate
        self.min_leaf_sample_rate = min_leaf_sample_rate
        self.tree = None
        self.gini = 0
        self.verbose = verbose
        
        if self.split_criterion == 'gini':
            self.gini = 1

    def fit(self, data, labels, val_data=None, val_labels=None, early_stop_rounds=1, random_state=42):
        """
        Train the decision tree.
        :param data: Feature data.
        :param labels: Labels.
        :param val_data: Validation feature data.
        :param val_labels: Validation labels.
        :param early_stop_rounds: Number of rounds for early stopping.
        :param random_state: Random seed for reproducibility.
        :return: True if the tree is successfully trained.
        """
        from decision_tree_with_null_zhoumath import DecisionTreeWithNullZhoumath

        if np.any(np.equal(data, np.nan)):
            tree_with_null = DecisionTreeWithNullZhoumath(
                split_criterion=self.split_criterion,
                search_method=self.search_method,
                max_depth=self.max_depth
            )
            tree_with_null.fitting(data, labels, val_data, val_labels, early_stop_rounds, random_state)
            self.tree = tree_with_null.tree
            self.feature_importances = tree_with_null.feature_importances
        
        else:
            self._fitting(data, labels, val_data, val_labels, early_stop_rounds, random_state)
        
    def _fitting(self, data, labels, val_data, val_labels, early_stop_rounds, random_state):
        """
        Train the decision tree model.
        :param data: Training feature data.
        :param labels: Training labels.
        :param val_data: Validation feature data for early stopping.
        :param val_labels: Validation labels for early stopping.
        :param early_stop_rounds: Number of early stop rounds without improvement before stopping.
        :param random_state: Random seed for reproducibility.
        """
        from decision_tree_helper_zhoumath import EarlyStopper, FeatureImportances, CatgorialModule
        
        data = np.ascontiguousarray(data)
        labels = np.ascontiguousarray(labels)
        self.categorial_module = CatgorialModule(data)
        
        if np.any(self.categorial_module.is_cat_feature):
            data = self.categorial_module._prepossess_catgorial(data, labels, random_state)
            
            if early_stop_rounds:
                val_data = self.categorial_module._prepossess_catgorial_val(val_data)
        else:
            data = DecisionTreeZhoumath._add_perturbation(data, random_state)
        
        if early_stop_rounds and self.search_method != 'bfs':
            raise ValueError("Early Stopping requires 'bfs' as the search method.")

        if (val_data is not None) and (val_labels is not None) and early_stop_rounds is not None:
            if self.verbose:
                print("Early stop mode is opened. Search method can only be BFS.")
                
            val_data = np.ascontiguousarray(val_data)
            val_labels = np.ascontiguousarray(val_labels)
            early_stopper = EarlyStopper(val_data=val_data,
                                         val_labels=val_labels,
                                         early_stop_rounds=early_stop_rounds,
                                         verbose=self.verbose)
            self.search_method = 'bfs'
            self.best_tree = []
        else:
            early_stopper = None

        labels = np.ascontiguousarray(labels.astype(np.int32))
        self.data = data
        self.labels = labels
        self.feature_importances = FeatureImportances(data.shape[1])
        self.feature_importances_cache = FeatureImportances(data.shape[1])
        self.tree = self._build_tree(early_stopper)
        self.data = None
        self.labels = None
        
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
        Recursively build the decision tree.
        :param early_stopper: Instance of EarlyStopper for early stopping.
        :return: A decision tree represented as a list of nodes.
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

            if early_stopper is not None and current_node.depth > early_stopper.current_max_depth:
                early_stop_triggered = early_stopper._evaluate_early_stop(self, current_node, tree)
                
                if early_stop_triggered:
                    return self.best_tree.copy()
                
            min_split_sample_rate_condition = current_node.row_indices.shape[0] < (self.labels.shape[0] * self.min_split_sample_rate)
            
            if (self.max_depth is not None and current_node.depth >= self.max_depth) or np.unique(labels).size == 1 or min_split_sample_rate_condition:
                DecisionTreeZhoumath._finallize_node(labels, tree, current_node)
                continue

            current_best_status, filtered_indices = self._choose_best_split(current_node)
            
            min_right_leaf_sample_rate_condition = (current_best_status.right_indices.shape[0]) < (self.labels.shape[0] * self.min_leaf_sample_rate)
            min_left_leaf_sample_rate_condition = (current_best_status.left_indices.shape[0]) < (self.labels.shape[0] * self.min_leaf_sample_rate)
            min_leaf_sample_rate_condition = min_right_leaf_sample_rate_condition or min_left_leaf_sample_rate_condition

            if current_best_status.best_feature is None or current_best_status.best_metric <= 0 or min_leaf_sample_rate_condition:
                DecisionTreeZhoumath._finallize_node(labels, tree, current_node)
                continue

            current_tree_node = TreeNode(feature=current_best_status.best_feature,
                                         threshold=current_best_status.best_threshold,
                                         null_direction=current_best_status.best_null_direction)

            if early_stopper is not None:
                probabilities = DecisionTreeZhoumath._calculate_probabilities(labels)
                current_tree_node.prob = probabilities

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
    
    def _init_root_collection_node(self):
        """
        Initialize the root collection node for building the tree.
        :return: Initialized root collection node with depth 0 and root indices.
        """
        from decision_tree_helper_zhoumath import CollectionNode, ParentIndices
        
        root_indices = ParentIndices(parent_sorted_indices=np.ascontiguousarray(np.argsort(self.data, axis=0)))
        root_collection_node = CollectionNode(depth=0,
                                              row_indices=np.arange(self.labels.shape[0]),
                                              parent_indices=root_indices)
        return root_collection_node
    
    @staticmethod
    def _finallize_node(labels,tree, current_node):
        from decision_tree_helper_zhoumath import TreeNode
        
        probabilities = DecisionTreeZhoumath._calculate_probabilities(labels)
        current_tree_node = TreeNode(prob=probabilities)
        DecisionTreeZhoumath._add_node_to_tree(tree, current_node, current_tree_node)

    @staticmethod
    @njit
    def _calculate_probabilities(labels):
        """
        Calculate class probabilities for the given labels.
        :param labels: Labels for the current node.
        :return: A probability distribution as a NumPy array.
        """
        mean = np.mean(labels)
        return np.array([1 - mean, mean])

    @staticmethod
    def _add_node_to_tree(tree, current_node, current_tree_node):
        """
        Add a new node to the decision tree.
        :param tree: List representing the current tree structure.
        :param current_node: Current collection node being processed.
        :param current_tree_node: TreeNode representing the new node to add to the tree.
        """
        tree.append(current_tree_node)
        if current_node.parent_index is not None:
            parent_tree_node = tree[current_node.parent_index]

            if current_node.child_direction == 'left':
                parent_tree_node.left = current_node.node_index
            else:
                parent_tree_node.right = current_node.node_index

    def _choose_best_split(self, current_node):
        """
        Determine the best feature and threshold to split the current node.
        :param current_node: The current collection node being processed.
        :return: Best split status and filtered sorted indices for the node.
        """
        from decision_tree_helper_zhoumath import BestStatus
        
        filtered_indices, filtered_sorted_labels, base_metric = self._init_best_split(current_node, False)
        metrics = DecisionTreeZhoumath._calculate_metrics(filtered_sorted_labels, base_metric, self.pos_weight, self.gini)

        if self.split_criterion == 'entropy_gain_ratio':
            intrinsic_value = DecisionTreeZhoumath._calculate_intrinsic_value(filtered_sorted_labels[:, 0])
            metrics = metrics / intrinsic_value
            
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

        if current_node.depth == 0:
            filtered_indices = current_node.parent_indices
        else:
            filtered_indices = current_node.parent_indices._filter_sorted_indices(current_node.row_indices, isnull)
        
        filtered_indices = current_node.parent_indices._filter_sorted_indices(current_node.row_indices, isnull)
        filtered_sorted_labels = np.ascontiguousarray(self.labels[filtered_indices.parent_sorted_indices])
        base_metric = DecisionTreeZhoumath._calculate_base_metric(filtered_sorted_labels[:, 0], self.pos_weight, self.gini)
        return filtered_indices, filtered_sorted_labels, base_metric

    @staticmethod
    @njit
    def _calculate_base_metric(labels, pos_weight, gini):
        """
        Calculate the base metric for the given labels.
        :param labels: Labels.
        :param pos_weight: Weight for positive class.
        :param gini: Indicator for Gini impurity.
        :return: Base metric value.
        """
        one_rate = labels.mean()
        zero_rate = 1 - one_rate
        zero_rate = zero_rate + 1e-9
        one_rate = one_rate + 1e-9
        zero_rate = zero_rate / (zero_rate + pos_weight * one_rate)
        one_rate = 1 - zero_rate

        if gini == 0:
            return -zero_rate * np.log2(zero_rate + 1e-9) - pos_weight * one_rate * np.log2(one_rate + 1e-9)
        else:
            return 1 - zero_rate ** 2 - one_rate ** 2

    @staticmethod
    @njit
    def _calculate_metrics(sorted_labels, base_metric, pos_weight, gini):
        """
        Calculate information gain for potential split thresholds.
        :param sorted_labels: Labels sorted by the feature values.
        :param base_metric: Base metric before the split.
        :param pos_weight: Weight for positive class.
        :param gini: Indicator for Gini impurity.
        :return: Information gain for each potential threshold.
        """
        num_rows = sorted_labels.shape[0]
        sorted_labels_t = sorted_labels.T
        sorted_labels_t = np.ascontiguousarray(sorted_labels_t)
        sorted_labels_total_cumsum = sorted_labels_t.reshape(-1).cumsum().reshape(sorted_labels_t.shape).T
        sorted_labels_total_cumsum = np.ascontiguousarray(sorted_labels_total_cumsum)
        left_cumsum = (sorted_labels_total_cumsum - np.concatenate((np.array([0], dtype=np.int64),
                                                                    sorted_labels_total_cumsum[-1, :-1])))[:-1, :]
        left_cumsum = np.ascontiguousarray(left_cumsum)
        right_cumsum = np.sum(sorted_labels[:, 0]) - left_cumsum
        left_one_rate = left_cumsum / np.arange(1, num_rows).reshape(-1, 1)
        right_one_rate = right_cumsum / np.arange(num_rows - 1, 0, -1).reshape(-1, 1)
        left_zero_rate = np.ones(left_one_rate.shape) - left_one_rate
        right_zero_rate = np.ones(right_one_rate.shape) - right_one_rate
        left_zero_rate = left_zero_rate / (left_zero_rate + pos_weight * left_one_rate)
        right_zero_rate = right_zero_rate / (right_zero_rate + pos_weight * right_one_rate)
        left_one_rate = 1 - left_zero_rate
        right_one_rate = 1 - right_zero_rate
        left_probs = (np.arange(1, num_rows) / num_rows).reshape(-1, 1)
        right_probs = (np.arange(num_rows - 1, 0, -1) / num_rows).reshape(-1, 1)

        if gini == 0:
            left_entropy = -left_zero_rate * np.log2(left_zero_rate + 1e-9) - pos_weight * left_one_rate * np.log2(left_one_rate + 1e-9)
            right_entropy = -right_zero_rate * np.log2(right_zero_rate + 1e-9) - pos_weight * right_one_rate * np.log2(right_one_rate + 1e-9)
            weighted_entropy = left_probs * left_entropy + right_probs * right_entropy
            return base_metric - weighted_entropy
        else:
            left_gini = 1 - left_zero_rate ** 2 - left_one_rate ** 2
            right_gini = 1 - right_zero_rate ** 2 - right_one_rate ** 2
            weighted_gini = left_probs * left_gini + right_probs * right_gini
            return base_metric - weighted_gini

    @staticmethod
    @njit
    def _calculate_intrinsic_value(labels):
        """
        Calculate the intrinsic value for a given split.
        :param labels: Labels.
        :return: Intrinsic value.
        """
        num_rows = labels.shape[0]
        left_prob = (np.arange(1, num_rows) / num_rows).reshape(-1, 1)
        right_prob = (np.arange(num_rows - 1, 0, -1) / num_rows).reshape(-1, 1)
        intrinsic_value = -left_prob * np.log2(left_prob + 1e-9) - right_prob * np.log2(right_prob + 1e-9)
        return intrinsic_value.reshape(-1, 1)

    def predict_proba(self, data, tree=None):
        """
        Predict probabilities for a batch of samples.
        :param data: Feature data.
        :param tree: Decision tree to use for prediction.
        :return: Probability predictions as a 2D array.
        """
        from decision_tree_helper_zhoumath import Predictor
        
        if tree is None:
            tree = self.tree.copy()
            data = np.ascontiguousarray(data)
            
            if np.any(self.categorial_module.is_cat_feature):
                data = self.categorial_module._prepossess_catgorial_val(data)
                    
        data = np.ascontiguousarray(data)
        predictor = Predictor(data, tree)
        return predictor._get_prediction()

    def replace_features_with_column_names(self, column_names):
        """
        Replace feature indices in the tree with column names.
        :param column_names: List of column names.
        """
        for i, node in enumerate(self.tree):
            if node.feature is not None and type(node.feature) != str:
                node.feature = column_names[node.feature]
                print(f'Node {i}, feature {node.feature}, threshold {node.threshold} left {node.left}, right {node.right}.')

    def to_pkl(self, filename):
        """
        Save the current model instance to a pickle file.
        :param filename: Path to the file where the model will be saved.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)
