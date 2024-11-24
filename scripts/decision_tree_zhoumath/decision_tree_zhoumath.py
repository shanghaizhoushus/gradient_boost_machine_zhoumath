# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:09:39 2024

@author: zhoushus
"""

# Import packages
import warnings
import numpy as np
from collections import deque
from sklearn.metrics import roc_auc_score
from numba import njit
import pickle

# Settings
warnings.filterwarnings("ignore", category=UserWarning)

# TreeNode class(Under Construction)
class TreeNode:
    def __init__(self, feature=None, threshold=None, prob=None, left=None, right=None, null_direction=None):
        self.feature = feature
        self.threshold = threshold
        self.prob = prob
        self.left = left
        self.right = right
        self.null_direction = null_direction

# DecisionTree class
class DecisionTreeZhoumath:
    
    def __init__(self, split_criterion, search_method, max_depth=None, pos_weight = 1):
        """
        Initialize the decision tree.
        :param split_criterion: Criterion for splitting ("gain" or "gain_ratio").
        :param search_method: Search method to split the tree ("bfs" or "dfs").
        :param max_depth: Maximum depth of the tree.
        """
        valid_split_criteria = {"gain", "gain_ratio"}
        valid_search_methods = {"bfs", "dfs"}
        
        if split_criterion not in valid_split_criteria:
            raise ValueError(f"Invalid split criterion: {split_criterion}. Choose from {valid_split_criteria}.")
        
        if search_method not in valid_search_methods:
            raise ValueError(f"Invalid search method: {search_method}. Choose from {valid_search_methods}.")
        
        self.split_criterion = split_criterion
        self.search_method = search_method
        self.max_depth = max_depth
        self.pos_weight = pos_weight
        self.tree = None
    
    def fit(self, data, labels, val_data = None, val_labels = None, early_stop_rounds = 1, random_state=42):
        """
        Train the decision tree.
        :param data: Feature data.
        :param labels: Labels.
        :param random_state: Random seed for reproducibility.
        """
        if self.early_stop and self.search_method != 'bfs':
            raise ValueError("Early Stopping requires 'bfs' as the search method.")
            
        if np.any(np.isnan(data)):
            tree_with_null = DecisionTreeWithNullZhoumath(
                split_criterion=self.split_criterion,
                search_method=self.search_method,
                max_depth=self.max_depth
            )
            tree_with_null.fit(data, labels, val_data, val_labels, early_stop_rounds, random_state)
            self.tree = tree_with_null.tree
            
            return 0
            
        self.early_stop = False
        
        if (val_data is not None) and (val_data is not None):
            print("Early stop mode is opened. Search method can only be BFS.")
            self.early_stop = True
            self.val_data = val_data
            self.val_labels = val_labels
            self.search_method = 'bfs'
            self.early_stop_rounds = early_stop_rounds
            self.best_auc = 0
            self.best_tree = []
        
        data = np.ascontiguousarray(data)
        data = DecisionTreeZhoumath._add_perturbation(data, random_state)
        labels = np.ascontiguousarray(labels.astype(np.int32))
        self.data = data
        self.labels = labels
        self.tree = self._build_tree()
            
        if self.early_stop == True:
            self.tree = self.best_tree
            
        self.data = None
        self.labels = None
        
        return 0
    
    
    @staticmethod
    @njit
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
    
    def _build_tree(self):
        """
        Recursively build the decision tree.
        :return: A decision tree represented as a list of nodes.
        """
        if self.search_method == 'dfs':
            collection = [{
                "depth": 0,
                "row_indices": np.arange(self.labels.shape[0]),
                "parent_sorted_indices": np.argsort(self.data, axis=0)
            }]
            pop_method = collection.pop
        else:
            collection = deque([{
                "depth": 0,
                "row_indices": np.arange(self.labels.shape[0]),
                "parent_sorted_indices": np.ascontiguousarray(np.argsort(self.data, axis=0))
            }])
            pop_method = collection.popleft
        
        tree = []
        
        if self.early_stop == True:
            current_max_depth = 0
            currert_early_stop_rounds = 0
        
        while collection:
            node = pop_method()
            node_index = len(tree)
            depth, row_indices, parent_sorted_indices = node["depth"], node["row_indices"], node["parent_sorted_indices"]
            labels = np.ascontiguousarray(self.labels[row_indices])
            parent_index = node.get("parent_index", None)
            child_direction = node.get("child_direction", None)
            
            if self.early_stop == True and depth > current_max_depth:
                current_max_depth, currert_early_stop_rounds = self._evaluate_early_stop(tree, depth, current_max_depth, currert_early_stop_rounds)
                
                if currert_early_stop_rounds >= self.early_stop_rounds:
                    return tree
            
            if (self.max_depth is not None and depth >= self.max_depth) or np.unique(labels).size == 1:
                probabilities = DecisionTreeZhoumath._calculate_probabilities(labels)
                leaf = {"prob": probabilities}
                self._add_node_to_tree(tree, parent_index, node_index, child_direction, leaf)
                continue
            
            best_feature, best_threshold, best_metric, left_row_indices, right_row_indices, filtered_sorted_indices = self._choose_best_split(
                row_indices, parent_sorted_indices, depth
            )
            
            if best_feature is None or best_metric <= 0:
                probabilities = DecisionTreeZhoumath._calculate_probabilities(labels)
                leaf = {"prob": probabilities}
                self._add_node_to_tree(tree, parent_index, node_index, child_direction, leaf)
                continue
            
            current_node = {"feature": best_feature, "threshold": best_threshold}
            
            if self.early_stop == True:
                probabilities = DecisionTreeZhoumath._calculate_probabilities(labels)
                leaf = {"prob": probabilities}
                current_node["prob"] = probabilities
            
            
            self._add_node_to_tree(tree, parent_index, node_index, child_direction, current_node)
            
            collection.append({
                "row_indices": right_row_indices,
                "depth": depth + 1,
                "parent_sorted_indices": filtered_sorted_indices,
                "parent_index": node_index,
                "child_direction": "right"
            })
            collection.append({
                "row_indices": left_row_indices,
                "depth": depth + 1,
                "parent_sorted_indices": filtered_sorted_indices,
                "parent_index": node_index,
                "child_direction": "left"
            })
            
        if self.early_stop == True:
            current_max_depth, currert_early_stop_rounds = self._evaluate_early_stop(tree, depth, current_max_depth, currert_early_stop_rounds)
            
            if currert_early_stop_rounds >= self.early_stop_rounds:
                return tree
        
        return tree
    
    def _evaluate_early_stop(self, tree, depth, current_max_depth, currert_early_stop_rounds):
        labels_pred = self.predict_proba(self.data, tree)[:, 1]
        val_labels_pred = self.predict_proba(self.val_data, tree)[:, 1]
        train_auc = roc_auc_score(self.labels, labels_pred)
        val_auc = roc_auc_score(self.val_labels, val_labels_pred)
        print(f'Current depth: {current_max_depth}, current train AUC: {train_auc:.3f}, current val AUC: {val_auc:.3f}')
        current_max_depth = depth
        
        if val_auc > self.best_auc:
            currert_early_stop_rounds = 0
            self.best_auc = val_auc
            self.best_tree = tree
        else:
            currert_early_stop_rounds += 1
            
        return current_max_depth, currert_early_stop_rounds
    
    @staticmethod
    @njit
    def _calculate_probabilities(labels):
        """
        Calculate the probability of each class in the given labels.
        :param labels: Labels.
        :return: A probability array.
        """
        mean = np.mean(labels)
        return np.array([1 - mean, mean])
    
    @staticmethod
    def _add_node_to_tree(tree, parent_index, node_index, child_direction, node):
        """
        Helper function to add a node to the tree structure.
        :param tree: The tree list.
        :param parent_index: Index of the parent node in the tree.
        :param child_direction: "left" or "right" indicating the child branch.
        :param node: Node to be added (dict).
        """
        tree.append(node)
        if parent_index is not None:
            tree[parent_index][child_direction] = node_index
    
    def _choose_best_split(self, row_indices, parent_sorted_indices, depth):
        """
        Determine the best split for the current node.
        :param row_indices: Row indices for the current node.
        :param parent_sorted_indices: Sorted indices of the parent node.
        :param depth: Depth of the current node.
        :return: Best feature, threshold, metric, left and right row indices, and filtered sorted indices.
        """
        best_feature = None
        best_index = 0
        best_metric = 0
        best_threshold = None
         
        if depth == 0:
            filtered_sorted_indices = parent_sorted_indices     
        else:
            filtered_sorted_indices = DecisionTreeZhoumath._filter_sorted_indices(row_indices, parent_sorted_indices)
            
        selected_labels = np.ascontiguousarray(self.labels[row_indices])
        base_entropy = DecisionTreeZhoumath._calculate_entropy(selected_labels, self.pos_weight)
        
        filtered_sorted_labels = np.ascontiguousarray(self.labels[filtered_sorted_indices])
        metrics = DecisionTreeZhoumath._calculate_metrics(filtered_sorted_labels, base_entropy, self.pos_weight)
        
        if self.split_criterion == 'gain_ratio':
            intrinsic_value = DecisionTreeZhoumath._calculate_intrinsic_value(selected_labels)
            metrics = metrics / intrinsic_value
    
        best_metric = np.max(metrics)
        best_index, best_feature = np.unravel_index(metrics.argmax(), metrics.shape)
        left_indices, right_indices = DecisionTreeZhoumath._get_left_right_indices(filtered_sorted_indices, best_index, best_feature)
        sorted_best_feature_data = self.data[:, best_feature][filtered_sorted_indices[:, best_feature]]
        sorted_best_feature_data = np.ascontiguousarray(sorted_best_feature_data)
        best_threshold = np.round((sorted_best_feature_data[best_index] + sorted_best_feature_data[best_index + 1]) / 2, 7)
        
        if best_threshold <= 0:
            return None, None, None, None, None, None
        
        return best_feature, best_threshold, best_metric, left_indices, right_indices, filtered_sorted_indices
    
    @staticmethod
    def _filter_sorted_indices(row_indices, parent_sorted_indices):
        """
        Retrieve the sorted indices for the given rows.
        :param row_indices: Row indices to filter.
        :param parent_sorted_indices: Sorted indices of the parent node.
        :return: Filtered sorted indices for all features.
        """
        sorted_indices_flattened = DecisionTreeZhoumath._flatten_sorted_indices(parent_sorted_indices)
        filtered_sorted_indices_flattened = sorted_indices_flattened[np.in1d(sorted_indices_flattened, row_indices)]
        return DecisionTreeZhoumath._unflatten_sorted_indices(parent_sorted_indices, filtered_sorted_indices_flattened)
    
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
    
    @staticmethod
    @njit
    def _calculate_entropy(labels, pos_weight):
        """
        Calculate the base entropy for the given labels.
        :param labels: Labels.
        :return: Base entropy value.
        """
        one_rate = labels.mean()
        zero_rate = 1 - one_rate
        return -zero_rate * np.log2(zero_rate + 1e-9) - pos_weight * one_rate * np.log2(one_rate + 1e-9)
    
    @staticmethod
    @njit
    def _calculate_metrics(sorted_labels, base_entropy, pos_weight):
        """
        Calculate information gain for potential split thresholds.
        :param sorted_labels: Labels sorted by the feature values.
        :param base_entropy: Base entropy before the split.
        :return: Information gain for each potential threshold.
        """
        num_rows = sorted_labels.shape[0]
        sorted_labels_t = sorted_labels.T
        sorted_labels_t = np.ascontiguousarray(sorted_labels_t)
        sorted_labels_total_cumsum = sorted_labels_t.reshape(-1).cumsum().reshape(sorted_labels_t.shape).T
        sorted_labels_total_cumsum = np.ascontiguousarray(sorted_labels_total_cumsum)
        left_cumsum = (sorted_labels_total_cumsum - np.concatenate((np.array([0], dtype=np.int64), sorted_labels_total_cumsum[-1, :-1])))[:-1, :]
        left_cumsum = np.ascontiguousarray(left_cumsum)
        right_cumsum = np.sum(sorted_labels[:, 0]) - left_cumsum
        left_one_rate = left_cumsum / np.arange(1, num_rows).reshape(-1, 1)
        right_one_rate = right_cumsum / np.arange(num_rows - 1, 0, -1).reshape(-1, 1)
        left_zero_rate = np.ones(left_one_rate.shape) - left_one_rate
        right_zero_rate = np.ones(right_one_rate.shape) - right_one_rate
        left_entropy = -left_zero_rate * np.log2(left_zero_rate + 1e-9) - pos_weight * left_one_rate * np.log2(left_one_rate + 1e-9)
        right_entropy = -right_zero_rate * np.log2(right_zero_rate + 1e-9) - pos_weight * right_one_rate * np.log2(right_one_rate + 1e-9)
        left_probs = (np.arange(1, num_rows) / num_rows).reshape(-1, 1)
        right_probs = (np.arange(num_rows - 1, 0, -1) / num_rows).reshape(-1, 1)
        weighted_entropy = left_probs * left_entropy + right_probs * right_entropy
        return base_entropy - weighted_entropy
    
    @staticmethod
    @njit
    def _calculate_intrinsic_value(labels):
        """
        Calculate the intrinsic value for a given split.
        :param labels: Labels.
        :return: Intrinsic Value (IV).
        """
        num_rows = labels.shape[0]
        left_prob = (np.arange(1, num_rows) / num_rows).reshape(-1, 1)
        right_prob = (np.arange(num_rows - 1, 0, -1) / num_rows).reshape(-1, 1)
        intrinsic_value = -left_prob * np.log2(left_prob + 1e-9) - right_prob * np.log2(right_prob + 1e-9)
        return intrinsic_value.reshape(-1, 1)
    
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
    
    def predict_proba(self, data, tree = None):
        """
        Predict probabilities for a batch of samples.
        :param data: Feature data.
        :return: Probability predictions as a 2D array.
        """
        if tree is None:
            tree = self.tree

        X = np.ascontiguousarray(data)
        indices = np.arange(X.shape[0], dtype=int)
        current_node = np.zeros((X.shape[0]), dtype=int)
        probabilities = np.empty((X.shape[0]))
        
        for i in range(len(tree)):
            node = tree[i]
            feature = node.get('feature', None)
            threshold = node.get('threshold', None)
            left = node.get('left', None)
            right = node.get('right', None)
            prob = node.get('prob', None)
            null_direction = node.get('null_direction', 'left')
            DecisionTreeZhoumath._traverse_next_node(feature, threshold, left, right, prob, X,
                                                     indices, current_node, probabilities, null_direction, i)
        
        return np.vstack([1 - probabilities, probabilities]).T
    
    @staticmethod
    def _traverse_next_node(feature, threshold, left, right, prob, X, indices, current_node, probabilities, null_direction, i):
        """
        Navigate to the next node in the decision tree.
        :param feature: Feature to split.
        :param threshold: Threshold value for the split.
        :param left: Index of the left child node.
        :param right: Index of the right child node.
        :param prob: Probability if the node is a leaf.
        :param X: Feature data.
        :param indices: Indices of the samples.
        :param current_node: Current node for each sample.
        :param probabilities: Probabilities for each sample.
        :param i: Index of the current node.
        """
        if feature is not None:
            index = indices[current_node == i]
            feature_values = X[index, feature]
            
            if null_direction == 'left':
                left_condition = (feature_values <= threshold) | np.isnan(feature_values)
            else:
                left_condition = (feature_values <= threshold)
                
                
            if left is not None and right is not None:
                current_node[index[left_condition]] = left
                current_node[index[~left_condition]] = right
        
        if prob is not None:
            probabilities[indices[current_node == i]] = prob[1]
    
    def replace_features_with_column_names(self, column_names):
        """
        Replace feature indices in the tree with column names.
        :param column_names: List of column names.
        """
        for node in self.tree:
            if "feature" in node and isinstance(node["feature"], int):
                node["feature"] = column_names[node["feature"]]
                
    def to_pkl(self, filename):
        """
        Save the current model instance to a pickle file.
        :param filename: Path to the file where the model will be saved.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

#DecisionTreeWithNullZhoumath extends DecisionTreeZhoumath
class DecisionTreeWithNullZhoumath(DecisionTreeZhoumath):
    
    def fit(self, data, labels, val_data, val_labels, early_stop_rounds, random_state=42):
        """
        Train the decision tree.
        :param data: Feature data.
        :param labels: Labels.
        """
        self.early_stop = False
        
        if (val_data is not None) and (val_data is not None):
            print("Early stop mode is opened. Search method can only be BFS.")
            self.early_stop = True
            self.val_data = val_data
            self.val_labels = val_labels
            self.search_method = 'bfs'
            self.early_stop_rounds = early_stop_rounds
            self.best_auc = 0
            self.best_tree = []
        
        data = np.ascontiguousarray(data)
        data = DecisionTreeZhoumath._add_perturbation(data, random_state)
        labels = np.ascontiguousarray(labels.astype(np.int32))
        self.data = data
        self.labels = labels
        self.tree = self._build_tree()
            
        if self.early_stop == True:
            self.tree = self.best_tree
            
        self.data = None
        self.labels = None
        
    
    def _build_tree(self):
        """
        Recursively build the decision tree.
        :return: A decision tree represented as a list of nodes.
        """
        data = self.data
        root_sorted_indices = np.empty(shape=data.shape, dtype=np.int32)
        root_null_indices = np.array([np.where(np.isnan(data[:, i]))[0] for i in range(data.shape[1])], dtype=object)
        null_quantities = np.array([root_null_indices[i].shape[0] for i in range(root_null_indices.shape[0])])
        root_sorted_indices_with_null = np.argsort(self.data, axis=0)
        root_sorted_indices = np.array([
            root_sorted_indices_with_null[:-null_quantities[i], i] for i in range(root_null_indices.shape[0])
        ], dtype=object)
        
        if self.search_method == 'dfs':
            collection = [{
                "depth": 0,
                "row_indices": np.arange(self.labels.shape[0]),
                "parent_sorted_indices": root_sorted_indices,
                "parent_null_indices": root_null_indices
            }]
            pop_method = collection.pop
        else:
            collection = deque([{
                "depth": 0,
                "row_indices": np.arange(self.labels.shape[0]),
                "parent_sorted_indices": root_sorted_indices,
                "parent_null_indices": root_null_indices
            }])
            pop_method = collection.popleft
            
        tree = []
        
        if self.early_stop == True:
            current_max_depth = 0
            currert_early_stop_rounds = 0
        
        while collection:
            node = pop_method()
            node_index = len(tree)
            depth, row_indices, parent_sorted_indices, parent_null_indices = (
                node["depth"], node["row_indices"], node["parent_sorted_indices"], node["parent_null_indices"]
            )
            labels = np.ascontiguousarray(self.labels[row_indices])
            parent_index = node.get("parent_index", None)
            child_direction = node.get("child_direction", None)
            
            if self.early_stop == True and depth > current_max_depth:
                print(self.best_auc)
                current_max_depth, currert_early_stop_rounds = self._evaluate_early_stop(tree, depth, current_max_depth, currert_early_stop_rounds)
                
                if currert_early_stop_rounds >= self.early_stop_rounds:
                    return tree
            
            if (self.max_depth is not None and depth >= self.max_depth) or np.unique(labels).size == 1:
                probabilities = DecisionTreeZhoumath._calculate_probabilities(labels)
                leaf = {"prob": probabilities}
                self._add_node_to_tree(tree, parent_index, node_index, child_direction, leaf)
                continue
            
            best_feature, best_threshold, best_metric, best_null_direction, left_row_indices, right_row_indices, filtered_sorted_indices, filtered_null_indices = self._choose_best_split(
                row_indices, parent_sorted_indices, depth, parent_null_indices=parent_null_indices
            )
            
            if best_feature is None or best_metric <= 0:
                probabilities = DecisionTreeZhoumath._calculate_probabilities(labels)
                leaf = {"prob": probabilities}
                self._add_node_to_tree(tree, parent_index, node_index, child_direction, leaf)
                continue
            
            current_node = {"feature": best_feature, "threshold": best_threshold, "null_direction": best_null_direction}
            
            if self.early_stop == True:
                probabilities = DecisionTreeZhoumath._calculate_probabilities(labels)
                leaf = {"prob": probabilities}
                current_node["prob"] = probabilities
            
            self._add_node_to_tree(tree, parent_index, node_index, child_direction, current_node)
            
            collection.append({
                "row_indices": right_row_indices,
                "depth": depth + 1,
                "parent_sorted_indices": filtered_sorted_indices,
                "filter_null_indices": filtered_null_indices,
                "parent_index": node_index,
                "child_direction": "right",
                "parent_null_indices": root_null_indices
            })
            collection.append({
                "row_indices": left_row_indices,
                "depth": depth + 1,
                "parent_sorted_indices": filtered_sorted_indices,
                "filter_null_indices": filtered_null_indices,
                "parent_index": node_index,
                "child_direction": "left",
                "parent_null_indices": root_null_indices
            })
            
        if self.early_stop == True:
            current_max_depth, currert_early_stop_rounds = self._evaluate_early_stop(tree, depth, current_max_depth, currert_early_stop_rounds)
            
            if currert_early_stop_rounds >= self.early_stop_rounds:
                return tree
        
        return tree
    
    def _choose_best_split(self, row_indices, parent_sorted_indices, depth, **kwargs):
        """
        Determine the best split for the current node, considering missing values.
        :param row_indices: Row indices for the current node.
        :param parent_sorted_indices: Sorted indices of the parent node.
        :param depth: Depth of the current node.
        :param kwargs: Additional keyword arguments.
        :return: Best feature, threshold, metric, null direction, left and right row indices, filtered sorted indices, and filtered null indices.
        """
        parent_null_indices = kwargs.get('parent_null_indices', None)
        best_feature = None
        best_index = 0
        best_metric = 0
        best_threshold = None
        best_null_direction = None
        num_features = self.data.shape[1]
        
        if depth == 0:
            filtered_sorted_indices = parent_sorted_indices
            filtered_null_indices = parent_null_indices
        else:
            filtered_sorted_indices, filtered_null_indices = DecisionTreeWithNullZhoumath._filter_sorted_indices(
                row_indices, parent_sorted_indices, parent_null_indices
            )
            
        selected_labels = np.ascontiguousarray(self.labels[row_indices])
        base_entropy = DecisionTreeZhoumath._calculate_entropy(selected_labels, self.pos_weight)
        
        if self.split_criterion == 'gain_ratio':
            intrinsic_value = DecisionTreeZhoumath._calculate_intrinsic_value(selected_labels)
        
        for feature_index in range(num_features):
            for null_direction in ['left', 'right']:
                sorted_indices = filtered_sorted_indices[feature_index]
                null_indices = filtered_null_indices[feature_index]
                
                if null_direction == 'left':
                    sorted_indices = np.concatenate([null_indices, sorted_indices])
                    sorted_data = np.ascontiguousarray(self.data[sorted_indices, feature_index])
                    sorted_data = np.nan_to_num(sorted_data, nan=-np.inf)
                else:
                    sorted_indices = np.concatenate([sorted_indices, null_indices])
                    sorted_data = np.ascontiguousarray(self.data[sorted_indices, feature_index])
                    sorted_data = np.nan_to_num(sorted_data, nan=np.inf)
                
                sorted_labels = np.ascontiguousarray(self.labels[sorted_indices])
                metrics = DecisionTreeWithNullZhoumath._calculate_metrics(sorted_labels, base_entropy, self.pos_weight)
                
                if self.split_criterion == 'gain_ratio':
                    intrinsic_value = intrinsic_value.reshape(-1)
                    metrics = metrics / intrinsic_value
            
                metrics_max = metrics.max()
                thresholds = (sorted_data[:-1] + sorted_data[1:]) / 2
            
                if metrics_max > 0 and metrics_max > best_metric:
                    best_metric = metrics_max
                    best_feature = feature_index
                    best_index = metrics.argmax()
                    best_threshold = thresholds[best_index]
                    best_null_direction = null_direction
        
        null_indices = filtered_null_indices[best_feature]
        null_shape = null_indices.shape[0]
        
        if ((best_index <= null_shape) and (best_null_direction == 'left')) or best_threshold <= 0 or ((best_index >= selected_labels.shape[0] - null_shape) and (best_null_direction == 'right')):
            return None, None, None, None, None, None, None, None
        
        if best_null_direction == 'left':
            left_sorted_indices = filtered_sorted_indices[best_feature][:(best_index - null_shape + 1)]
            left_indices = np.concatenate([left_sorted_indices, null_indices])
            right_indices = filtered_sorted_indices[best_feature][(best_index - null_shape + 1):]
        else:
            left_indices = filtered_sorted_indices[best_feature][:(best_index + 1)]
            right_indices = np.concatenate([filtered_sorted_indices[best_feature][(best_index + 1):], null_indices])
            
        return (
            best_feature, best_threshold, best_metric, best_null_direction,
            np.ascontiguousarray(left_indices), np.ascontiguousarray(right_indices),
            filtered_sorted_indices, filtered_null_indices
        )
    
    @staticmethod
    def _filter_sorted_indices(row_indices, parent_sorted_indices, parent_null_indices):
        """
        Retrieve the sorted indices for the given rows, considering missing values.
        :param row_indices: Row indices to filter.
        :param parent_sorted_indices: Sorted indices of the parent node.
        :param parent_null_indices: Null value indices of the parent node.
        :return: Filtered sorted and null indices for all features.
        """
        filtered_sorted_indices = []
        filtered_null_indices = []
        
        for i in range(parent_sorted_indices.shape[0]):
            parent_sorted_indices_i = parent_sorted_indices[i]
            filtered_sorted_indices_i = parent_sorted_indices_i[np.in1d(parent_sorted_indices_i, row_indices)]
            filtered_sorted_indices.append(np.ascontiguousarray(filtered_sorted_indices_i))
            
            parent_null_indices_i = parent_null_indices[i]
            filtered_null_indices_i = parent_null_indices_i[np.in1d(parent_null_indices_i, row_indices)]
            filtered_null_indices.append(np.ascontiguousarray(filtered_null_indices_i))
        
        return np.array(filtered_sorted_indices, dtype=object), np.array(filtered_null_indices, dtype=object)
    
    
    @staticmethod
    @njit
    def _calculate_metrics(sorted_labels, base_entropy, pos_weight):
        """
        Calculate information gain for potential split thresholds.
        :param sorted_labels: Labels sorted by the feature values.
        :param base_entropy: Base entropy before the split.
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
        left_probs = (np.arange(1, num_rows) / num_rows)
        right_probs = (np.arange(num_rows - 1, 0, -1) / num_rows)
        left_entropy = -left_zero_rate * np.log2(left_zero_rate + 1e-9) - pos_weight *  left_one_rate * np.log2(left_one_rate + 1e-9)
        right_entropy = -right_zero_rate * np.log2(right_zero_rate + 1e-9) - pos_weight * right_one_rate * np.log2(right_one_rate + 1e-9)
        weighted_entropy = left_probs * left_entropy + right_probs * right_entropy
        return base_entropy - weighted_entropy
    
