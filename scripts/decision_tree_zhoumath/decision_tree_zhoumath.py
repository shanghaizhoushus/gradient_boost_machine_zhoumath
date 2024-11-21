# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:09:39 2024

@author: zhoushus
"""


#import packages
import warnings
import numpy as np
from collections import deque
from numba import njit


# Setting
warnings.filterwarnings("ignore", category=UserWarning)


# DecisionTree class
class DecisionTreeZhoumath:
    
    
    def __init__(self, split_criterion, search_method, max_depth=None):
        """
        Initialize the decision tree.
        :param max_depth: Maximum depth of the tree.
        :param search_method: Search method to split the tree.
        :param split_criterion: Criterion for splitting ("gain" or "gain_ratio").
        """
        valid_split_criteria = {"gain", "gain_ratio"}
        
        if split_criterion not in valid_split_criteria:
            raise ValueError(f"Invalid split criterion: {split_criterion}. Choose from {valid_split_criteria}.")
        
        valid_search_method = {"bfs", "dfs"}
        
        if search_method not in valid_search_method:
            raise ValueError(f"Invalid search method: {search_method}. Choose from {valid_search_method}.")
        
        self.split_criterion = split_criterion
        self.search_method = search_method
        self.max_depth = max_depth
        self.tree = None
    
    
    def fit(self, data, labels):
        """
        Train the decision tree.
        :param data: Feature data.
        :param labels: Labels.
        """
        data = np.ascontiguousarray(data)
        data = DecisionTreeZhoumath._get_perturbated_data(data)
        labels = np.ascontiguousarray(labels)
        sorted_indices = np.ascontiguousarray(np.argsort(data, axis=0))
        self.data = data
        self.labels = labels
        self.sorted_indices = sorted_indices
        self.tree = self.build_tree()
    
    
    @staticmethod
    @njit
    def _get_perturbated_data(data):
        perturbation = np.random.uniform(-1e-9, 1e-9, size=data.shape)
        data = data + perturbation
        data = np.ascontiguousarray(data)
        return data
    
    
    def build_tree(self):
        """
        Recursively build the decision tree.
        :return: A decision tree dictionary.
        """
        
        if self.search_method == 'dfs':
            collection = [{"depth": 0, "row_indices": np.arange(self.labels.shape[0])}]
        
        if self.search_method == 'bfs':
            collection = deque([{"depth": 0, "row_indices": np.arange(self.labels.shape[0])}])
        
        tree = []
        
        while collection:
            
            if self.search_method == 'dfs':
                node = collection.pop()
            
            if self.search_method == 'bfs':
                node = collection.popleft()
            
            node_index = len(tree)
            row_indices, depth = node["row_indices"], node["depth"]
            labels = self.labels[row_indices]
            parent_index = node.get("parent_index", None)
            child_direction = node.get("child_direction", None)
            
            if (self.max_depth is not None and depth >= self.max_depth) or (np.mean(labels) == 1 or np.mean(labels) == 0) or labels.shape[0] == 1:
                probs = DecisionTreeZhoumath._get_probs(labels)
                leaf = {"prob": probs}
                DecisionTreeZhoumath._add_node_to_tree(tree, parent_index, node_index,
                                                       child_direction, leaf)
                continue
            
            best_feature, best_threshold, best_metric, left_row_indices, right_row_indices = self.choose_best_split(row_indices)
            
            if best_feature is None or best_metric <= 0:
                probs = DecisionTreeZhoumath._get_probs(labels)
                leaf = {"prob": probs}
                DecisionTreeZhoumath._add_node_to_tree(tree, parent_index, node_index,
                                                       child_direction, leaf)
                continue
            
            current_node = {"feature": best_feature, "threshold": best_threshold}
            DecisionTreeZhoumath._add_node_to_tree(tree, parent_index, node_index,
                                                   child_direction, current_node)
            collection.append({"row_indices": right_row_indices, "depth": depth + 1,
                               "parent_index": node_index,"child_direction": "right"})
            collection.append({"row_indices": left_row_indices, "depth": depth + 1,
                               "parent_index": node_index,"child_direction": "left"})
        
        return tree
    
    
    @staticmethod
    @njit
    def _get_probs(labels):
        """
        Get the probability of a label list.
        :param labels: Labels.
        :return: A probability dictionary.
        """
        mean = np.mean(labels)
        probs = np.array([1 - mean, mean])
        return probs
    
    
    @staticmethod
    def _add_node_to_tree(tree, parent_index, node_index, child_direction, node):
        """
        Helper function to add a node to the tree structure.
        :param tree: The tree dictionary.
        :param node_list: A list of all nodes in the tree.
        :param parent_index: Index of the parent node in node_list.
        :param child_direction: "left" or "right" indicating the child branch.
        :param node: Node to be added (dict).
        """
        tree.append(node)
        
        if parent_index is not None:
            tree[parent_index][child_direction] = node_index
        
    
    
    def choose_best_split(self, row_indices):
        """
        Choose the best feature and threshold for splitting.
        :param row_indices: Row indices.
        :return: Best feature index, threshold, metric, and indices for both sides.
        """
        best_feature = None
        best_index = 0
        best_metric = 0
        best_threshold = None
        full_sorted_indices = self.get_sorted_indices(row_indices)
        selected_labels = self.labels[row_indices]
        base_entropy, left_labels_cumcount, left_prob, right_labels_cumcount, right_prob = DecisionTreeZhoumath._get_base_entropy_cumcount_prob(selected_labels)
        
        if self.split_criterion == 'gain_ratio':
            intrinsic_value = DecisionTreeZhoumath._calculate_intrinsic_value(left_prob, right_prob)
        
        for feature_index in range(self.data.shape[1]):
            sorted_indices = full_sorted_indices[:,feature_index]
            sorted_data = self.data[sorted_indices, feature_index]
            sorted_labels = self.labels[sorted_indices]
            thresholds, metrices = DecisionTreeZhoumath._calculate_thresholds_metrices(sorted_data, sorted_labels,
                                                                                       left_labels_cumcount, right_labels_cumcount,
                                                                                       left_prob, right_prob,
                                                                                       base_entropy)
            
            if self.split_criterion == 'gain_ratio':
                metrices = metrices / intrinsic_value
        	
            metrices_max = metrices.max()
        	
            if metrices_max > 0 and metrices_max > best_metric:
                best_metric = metrices_max
                best_feature = feature_index
                best_index = metrices.argmax()
                best_threshold = thresholds[best_index]
        
        
        left_indices, right_indices = DecisionTreeZhoumath._get_left_indices_right_indices(full_sorted_indices, best_index, best_feature)
        return best_feature, best_threshold, best_metric, left_indices, right_indices
    
    
    def get_sorted_indices(self, row_indices):
        """
        Retrieve the sorted indices for the given rows.
        :param row_indices: Row indices to sort.
        :return: Sorted indices for all features.
        """
        sorted_indices_flattened = self.sorted_indices.ravel(order='F')
        full_sorted_indices = sorted_indices_flattened[np.isin(sorted_indices_flattened, row_indices)].reshape((-1, self.sorted_indices.shape[1]), order='F')
        return full_sorted_indices
    
    
    @staticmethod
    @njit
    def _get_base_entropy_cumcount_prob(selected_labels):
        selected_labels_one_rate = selected_labels.mean()
        selected_labels_zero_rate = 1 - selected_labels_one_rate
        base_entropy = - selected_labels_zero_rate * np.log2(selected_labels_zero_rate) - selected_labels_one_rate * np.log2(selected_labels_one_rate)
        left_labels_cumcount = np.arange(1, selected_labels.shape[0])
        left_prob = left_labels_cumcount / selected_labels.shape[0]
        right_labels_cumcount = selected_labels.shape[0] - left_labels_cumcount
        right_prob = 1 - left_prob
        return base_entropy, left_labels_cumcount, left_prob, right_labels_cumcount, right_prob
    
    
    @staticmethod
    @njit
    def _calculate_thresholds_metrices(sorted_data, sorted_labels, left_labels_cumcount, right_labels_cumcount, left_prob, right_prob, base_entropy):
        """
        Calculate weighted entropy for two label distributions.
        :param left_sorted_labels_mean: Mean values for the left split.
        :param right_sorted_labels_mean: Mean values for the right split.
        :param left_prob: Probability weight for the left split.
        :param right_prob: Probability weight for the right split.
        :return: Weighted entropy value.
        """
        thresholds = (sorted_data[:-1] + sorted_data[1:]) / 2
        left_sorted_labels_cumsum = np.cumsum(sorted_labels)[:-1]
        left_sorted_labels_mean = left_sorted_labels_cumsum / left_labels_cumcount
        right_sorted_labels_cumsum = sorted_labels.sum() - left_sorted_labels_cumsum
        right_sorted_labels_mean = right_sorted_labels_cumsum / right_labels_cumcount
        left_sorted_labels_one_rate = left_sorted_labels_mean
        left_sorted_labels_zero_rate = 1 - left_sorted_labels_mean
        right_sorted_labels_one_rate = right_sorted_labels_mean
        right_sorted_labels_zero_rate = 1 - right_sorted_labels_mean
        left_entropy = - left_sorted_labels_zero_rate * np.log2(left_sorted_labels_zero_rate + 1e-9) - left_sorted_labels_one_rate * np.log2(left_sorted_labels_one_rate + 1e-9)
        right_entropy = - right_sorted_labels_zero_rate * np.log2(right_sorted_labels_zero_rate + 1e-9) - right_sorted_labels_one_rate * np.log2(right_sorted_labels_one_rate + 1e-9)
        weighted_entropy = left_prob * left_entropy + right_prob * right_entropy
        info_gain = base_entropy - weighted_entropy
        return thresholds, info_gain
    
    
    @staticmethod
    @njit 
    def _calculate_intrinsic_value(left_prob, right_prob):
        """
        Calculate the intrinsic value for a given split.
        :param left_prob: Probability of the left subset.
        :param right_prob: Probability of the right subset.
        :return: Intrinsic Value (IV).
        """
        intrinsic_value = - left_prob * np.log2(left_prob) - right_prob * np.log2(right_prob)
        return intrinsic_value
    
    
    @staticmethod
    @njit    
    def _get_left_indices_right_indices(full_sorted_indices, best_index, best_feature):
        left_indices = full_sorted_indices[:(best_index+1), best_feature]
        right_indices = full_sorted_indices[(best_index+1):, best_feature]
        return left_indices, right_indices
    
    
    def predict_proba(self, data):
        """
        Predict probabilities for a batch of samples.
        :param data: Feature data.
        :return: Probability predictions as a DataFrame.
        """
        X = np.array(data)
        indices = np.arange(X.shape[0], dtype = int)
        current_node = np.zeros((X.shape[0]), dtype = int)
        probs = np.zeros((X.shape[0]))
        
        for i in range(len(self.tree)):
            node = self.tree[i]
            feature = node.get('feature', None)
            threshold = node.get('threshold', None)
            left = node.get('left', None)
            right = node.get('right', None)
            prob = node.get('prob', None)
            DecisionTreeZhoumath._goto_next_node(feature, threshold, left, right, prob,
                                                 X, indices, current_node, probs, i)
        
        probs_stacked = np.vstack([1- probs, probs])
        return probs_stacked
    
    
    @staticmethod
    @njit
    def _goto_next_node(feature, threshold, left, right, prob, X, indices, current_node, probs, i):
        if feature is not None:
            index = indices[current_node == i]
            filter_X = X[index, feature]
            smaller = filter_X <= threshold
            current_node[index[smaller]] = left
            current_node[index[~smaller]] = right
        
        if prob is not None:
            probs[indices[current_node == i]] = prob[1]
        
    
    
    def replace_features_with_column_names(self, column_names):
        """
        Replace feature indices in the tree with column names.
        :param column_names: List of column names.
        """
        
        for node in self.tree:
            
            if "feature" in node:
                node["feature"] = column_names[node["feature"]]
            
        
    
    
