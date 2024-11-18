# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 12:50:56 2024

@author: zhoushus
"""


#import packages
import numpy as np
import pandas as pd
from collections import deque
import warnings


# Setting
warnings.filterwarnings("ignore", category=UserWarning)


# DecisionTree class
class DecisionTreeZhoumath:
    def __init__(self, split_criterion, search_method, max_depth=None):
        """
        Initialize the decision tree.
        :param max_depth: Maximum depth of the tree.
        :param search_method: Search mathod to split the tree
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
        
    def calculate_intrinsic_value(self, left_labels, right_labels):
        """
        Calculate the intrinsic value for a given split.
        :param left_labels: Labels in the left subset.
        :param right_labels: Labels in the right subset.
        :return: Intrinsic Value (IV).
        """
        total_count = len(left_labels) + len(right_labels)
        prob_left = len(left_labels) / total_count
        prob_right = len(right_labels) / total_count
        intrinsic_value = 0.0
    
        if prob_left > 0:
            intrinsic_value -= prob_left * np.log2(prob_left)
            
        if prob_right > 0:
            intrinsic_value -= prob_right * np.log2(prob_right)
    
        return intrinsic_value

    def calculate_entropy(self, labels):
        """
        Calculate entropy for a given label distribution.
        :param labels: Array of labels.
        :return: Entropy value.
        """
        probabilities = labels / np.sum(labels)
        non_zero_probs = probabilities[probabilities > 0]
        entropy = -np.sum(non_zero_probs * np.log2(non_zero_probs))
        return entropy

    def choose_best_split(self, data, labels):
        """
        Choose the best feature and threshold for splitting.
        :param data: Feature data.
        :param labels: Labels.
        :return: Best feature index, threshold, and information gain.
        """
        num_samples, num_features = data.shape
        base_entropy = self.calculate_entropy(labels)
        best_metric = -1
        best_feature = None
        best_threshold = None

        for feature_index in range(num_features):
            sorted_data = self.sorted_data[feature_index]
            sorted_labels = self.sorted_labels[feature_index]
            
            for i in range(1, num_samples):
                
                if sorted_data[i] == sorted_data[i - 1]:
                    continue
                
                threshold = (sorted_data[i] + sorted_data[i - 1]) / 2
                left_labels = sorted_labels[:i]
                right_labels = sorted_labels[i:]
                prob_left = len(left_labels) / len(labels)
                prob_right = len(right_labels) / len(labels)
                new_entropy = prob_left * self.calculate_entropy(left_labels) + prob_right * self.calculate_entropy(right_labels)
                info_gain = base_entropy - new_entropy

                if self.split_criterion == "gain_ratio":
                    intrinsic_value = self.calculate_intrinsic_value(left_labels, right_labels)
                    
                    if intrinsic_value > 0:
                        metric = info_gain / intrinsic_value
                    else:
                        metric = 0
                        
                elif self.split_criterion == "gain":
                    metric = info_gain
                else:
                    raise ValueError(f"Unsupported split criterion: {self.split_criterion}")

                if metric > best_metric:
                    best_metric = metric
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold, best_metric
    
    @staticmethod
    def _get_prob_dic(labels):
        """
        Get the probability of a label list.
        :param labels: Labels.
        :return: A probability dictionary.
        """
        
        if len(labels) == 0:
            return {"prob": [0.5, 0.5]}
        
        mean = np.mean(labels)
        return {"prob": [1 - mean, mean]}
    
    def _add_node_to_tree(self, tree, node_list, parent_index, child_direction, node):
        """
        Helper function to add a node to the tree structure.
        :param tree: The tree dictionary.
        :param node_list: A list of all nodes in the tree.
        :param parent_index: Index of the parent node in node_list.
        :param child_direction: "left" or "right" indicating the child branch.
        :param node: Node to be added (dict).
        """
        
        if parent_index is None:
            tree.update(node)
        else:
            parent = node_list[parent_index]
            
            if child_direction == "left":
                parent["left"] = node
            elif child_direction == "right":
                parent["right"] = node
                
        node_list.append(node)
        
    def build_tree_bfs(self, data, labels):
        """
        Build the decision tree using breadth-first search.
        :param data: Feature data (numpy array).
        :param labels: Labels (numpy array).
        :return: A decision tree dictionary.
        """
        queue = deque([{"data": data, "labels": labels, "depth": 0}])
        tree = {}
        node_list = []
    
        while queue:
            node = queue.popleft()
            data, labels, depth = node["data"], node["labels"], node["depth"]
            parent_index = node.get("parent_index", None)
            child_direction = node.get("child_direction", None)
    
            if len(set(labels)) == 1:
                leaf = {"prob": [1.0, 0.0]} if labels[0] == 0 else {"prob": [0.0, 1.0]}
                self._add_node_to_tree(tree, node_list, parent_index, child_direction, leaf)
                continue
    
            if self.max_depth is not None and depth >= self.max_depth:
                leaf = self._get_prob_dic(labels)
                self._add_node_to_tree(tree, node_list, parent_index, child_direction, leaf)
                continue
    
            best_feature, best_threshold, best_info_gain = self.choose_best_split(data, labels)
            
            if best_feature is None or best_info_gain <= 0:
                leaf = self._get_prob_dic(labels)
                self._add_node_to_tree(tree, node_list, parent_index, child_direction, leaf)
                continue
    
            current_node = {"feature": best_feature, "threshold": best_threshold}
            node_index = len(node_list)
            self._add_node_to_tree(tree, node_list, parent_index, child_direction, current_node)
            left_mask = data[:, best_feature] <= best_threshold
            right_mask = ~left_mask
            queue.append({"data": data[left_mask],
                          "labels": labels[left_mask],
                          "depth": depth + 1,
                          "parent_index": node_index,
                          "child_direction": "left"}
                         )
            queue.append({"data": data[right_mask],
                          "labels": labels[right_mask],
                          "depth": depth + 1,
                          "parent_index": node_index,
                          "child_direction": "right"}
                         )
    
        return tree
    
    def build_tree_dfs(self, data, labels):
        """
        Recursively build the decision tree.
        :param data: Feature data.
        :param labels: Labels.
        :param depth: Current depth of the tree.
        :return: A decision tree dictionary.
        """
        stack = [{"data": data, "labels": labels, "depth": 0}]
        tree = {}
    
        while stack:
            node = stack.pop()
            data, labels, depth = node["data"], node["labels"], node["depth"]
            
            if len(set(labels)) == 1 or (self.max_depth is not None and depth >= self.max_depth):
                leaf = self._get_prob_dic(labels)
                tree.update(leaf)
                continue
    
            best_feature, best_threshold, best_info_gain = self.choose_best_split(data, labels)
            if best_feature is None or best_info_gain <= 0:
                leaf = self._get_prob_dic(labels)
                tree.update(leaf)
                continue
    
            left_mask = data[:, best_feature] <= best_threshold
            right_mask = ~left_mask
            tree = {
                "feature": best_feature,
                "threshold": best_threshold,
                "left": self.build_tree_dfs(data[left_mask],
                                            labels[left_mask],
                                            depth + 1),
                "right": self.build_tree_dfs(data[right_mask],
                                            labels[right_mask],
                                             depth + 1)
            }
            return tree

    
    def fit(self, data, labels):
        """
        :param search_method: Search mathod to split the tree
        Train the decision tree.
        :param data: Feature data.
        :param labels: Labels.
        """
        
        self.sorted_indices = {i: np.argsort(data[:, i]) for i in range(data.shape[1])}
        self.sorted_data = {i: data[self.sorted_indices[i], i] for i in range(data.shape[1])}
        self.sorted_labels = {i: labels[self.sorted_indices[i]] for i in range(data.shape[1])}
        
        if self.search_method == 'bfs':
            self.tree = self.build_tree_bfs(data, labels)
        elif self.search_method == 'dfs':
            self.tree = self.build_tree_dfs_iterative(data, labels)

    def predict_proba_single(self, tree, sample):
        """
        Predict probabilities for a single sample.
        :param tree: Decision tree.
        :param sample: Feature vector.
        :return: Probability distribution.
        """
        
        if "prob" in tree:
            return tree["prob"]

        feature = tree["feature"]
        threshold = tree["threshold"]

        if sample[feature] <= threshold:
            return self.predict_proba_single(tree["left"], sample)
        else:
            return self.predict_proba_single(tree["right"], sample)

    def predict_proba(self, data):
        """
        Predict probabilities for a batch of samples.
        :param data: Feature data.
        :return: Probability predictions as a DataFrame.
        """
        pred_np = np.array([self.predict_proba_single(self.tree, sample) for sample in data])
        pred_pd = pd.DataFrame(pred_np, columns=[0, 1])
        return pred_pd

    def replace_features_with_column_names(self, column_names):
        """
        Replace feature indices in the tree with column names.
        :param column_names: List of column names.
        """
        self._replace_recursively(self.tree, column_names)

    def _replace_recursively(self, tree, column_names):
        """
        Recursively replace feature indices with column names in the tree.
        :param tree: Decision tree dictionary.
        :param column_names: List of column names.
        """
        
        if "feature" in tree:
            tree["feature"] = column_names[tree["feature"]]

            if "left" in tree:
                self._replace_recursively(tree["left"], column_names)
                
            if "right" in tree:
                self._replace_recursively(tree["right"], column_names)

