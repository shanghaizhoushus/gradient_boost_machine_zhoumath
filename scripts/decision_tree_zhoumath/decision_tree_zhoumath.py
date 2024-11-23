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
        :param split_criterion: Criterion for splitting ("gain" or "gain_ratio").
        :param search_method: Search method to split the tree ("bfs" or "dfs").
        :param max_depth: Maximum depth of the tree.
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
    
    
    def fit(self, data, labels, random_state=20221126):
        """
        Train the decision tree.
        :param data: Feature data.
        :param labels: Labels.
        """
        data = np.ascontiguousarray(data)
        data = DecisionTreeZhoumath._get_perturbated_data(data, random_state)
        labels = np.ascontiguousarray(labels.astype(np.int32))
        self.data = data
        self.labels = labels
        self.tree = self.build_tree(random_state)
    
    
    @staticmethod
    @njit
    def _get_perturbated_data(data, random_state):
        """
        Add a small perturbation to the data to avoid numerical issues.
        :param data: Feature data.
        :param random_state: Random seed for reproducibility.
        :return: Perturbated data.
        """
        np.random.seed(random_state)
        perturbation = np.random.uniform(-1, 1, size=data.shape) * 1e-7
        data = data + perturbation
        return data
    
    
    def build_tree(self, random_state):
        """
        Recursively build the decision tree.
        :return: A decision tree dictionary.
        """
        
        if self.search_method == 'dfs':
            collection = [{"depth": 0, "row_indices": np.arange(self.labels.shape[0]),
                           "parent_sorted_indices": np.argsort(self.data, axis=0)}]
            popmethod = collection.pop
        
        if self.search_method == 'bfs':
            collection = deque([{"depth": 0, "row_indices": np.arange(self.labels.shape[0]), "parent_sorted_indices": np.ascontiguousarray(np.argsort(self.data, axis=0))}])
            popmethod = collection.popleft
            
        tree = []
        
        while collection:
            
            node = popmethod()
            node_index = len(tree)
            depth, row_indices, parent_sorted_indices = node["depth"], node["row_indices"], node["parent_sorted_indices"]
            labels = np.ascontiguousarray(self.labels[row_indices])
            parent_index = node.get("parent_index", None)
            child_direction = node.get("child_direction", None)
            
            if (self.max_depth is not None and depth >= self.max_depth) or np.unique(labels).size == 1:
                probs = DecisionTreeZhoumath._get_probs(labels)
                leaf = {"prob": probs}
                DecisionTreeZhoumath._add_node_to_tree(tree, parent_index, node_index,
                                                       child_direction, leaf)
                continue
            
            best_feature, best_threshold, best_metric, left_row_indices, right_row_indices, filter_sorted_indices = self.choose_best_split(row_indices, parent_sorted_indices, depth)
            
            if best_feature is None or best_metric <= 0:
                probs = DecisionTreeZhoumath._get_probs(labels)
                leaf = {"prob": probs}
                DecisionTreeZhoumath._add_node_to_tree(tree, parent_index, node_index,
                                                       child_direction, leaf)
                continue
            
            current_node = {"feature": best_feature, "threshold": best_threshold}
            DecisionTreeZhoumath._add_node_to_tree(tree, parent_index, node_index,
                                                   child_direction, current_node)
            collection.append({"row_indices": right_row_indices, "depth": depth + 1, "parent_sorted_indices": filter_sorted_indices,
                               "parent_index": node_index,"child_direction": "right"})
            collection.append({"row_indices": left_row_indices, "depth": depth + 1, "parent_sorted_indices": filter_sorted_indices,
                               "parent_index": node_index,"child_direction": "left"})
        
        return tree
    
    
    @staticmethod
    @njit
    def _get_probs(labels):
        """
        Get the probability of a label list.
        :param labels: Labels.
        :return: A probability array.
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
        
    
    
    def choose_best_split(self, row_indices, parent_sorted_indices, depth):
        """
        Helper function to add a node to the tree structure.
        :param tree: The tree list.
        :param parent_index: Index of the parent node in the tree.
        :param node_index: Index of the current node.
        :param child_direction: "left" or "right" indicating the child branch.
        :param node: Node to be added (dict).
        """
        best_feature = None
        best_index = 0
        best_metric = 0
        best_threshold = None
         
        if depth == 0:
            filter_sorted_indices = parent_sorted_indices     
        else:
            filter_sorted_indices = DecisionTreeZhoumath._get_filter_sorted_indices(row_indices, parent_sorted_indices)
            
        selected_labels = np.ascontiguousarray(self.labels[row_indices])
        base_entropy, left_labels_cumcount, left_prob, right_labels_cumcount, right_prob = DecisionTreeZhoumath._get_base_entropy_cumcount_prob(selected_labels)
        
        filter_sorted_labels = np.ascontiguousarray(self.labels[filter_sorted_indices])
        metrices = DecisionTreeZhoumath._calculate_metrices(filter_sorted_labels, left_labels_cumcount,
                                                            right_labels_cumcount, left_prob,
                                                            right_prob, base_entropy)
        
        if self.split_criterion == 'gain_ratio':
            intrinsic_value = DecisionTreeZhoumath._calculate_intrinsic_value(left_prob, right_prob)
            metrices = metrices / intrinsic_value
    
        best_metric = np.max(metrices)
        best_index, best_feature = np.unravel_index(metrices.argmax(), metrices.shape)
        left_indices, right_indices = DecisionTreeZhoumath._get_left_indices_right_indices(filter_sorted_indices, best_index, best_feature)
        sorted_best_feature_data = self.data[:, best_feature][filter_sorted_indices[:, best_feature]]
        sorted_best_feature_data = np.ascontiguousarray(sorted_best_feature_data)
        best_threshold = (sorted_best_feature_data[best_index] + sorted_best_feature_data[best_index+1]) / 2
        return best_feature, best_threshold, best_metric, left_indices, right_indices, filter_sorted_indices
    
    
    @staticmethod
    def _get_filter_sorted_indices(row_indices, parent_sorted_indices):
        """
        Retrieve the sorted indices for the given rows.
        :param row_indices: Row indices to filter.
        :param parent_sorted_indices: Sorted indices of the parent node.
        :return: Sorted indices for all features.
        """
        parent_sorted_indices_t = parent_sorted_indices.T
        parent_sorted_indices_t = np.ascontiguousarray(parent_sorted_indices_t)
        sorted_indices_flattened = parent_sorted_indices_t.reshape(-1)
        filter_sorted_indices_flattened = sorted_indices_flattened[np.in1d(sorted_indices_flattened, row_indices)]
        filter_sorted_indices_flattened = np.ascontiguousarray(filter_sorted_indices_flattened)
        filter_sorted_indices_t = filter_sorted_indices_flattened.reshape((parent_sorted_indices.shape[1],-1))
        filter_sorted_indices = filter_sorted_indices_t.T
        filter_sorted_indices = np.ascontiguousarray(filter_sorted_indices)
        return filter_sorted_indices
    
    
    @staticmethod
    @njit
    def _get_base_entropy_cumcount_prob(selected_labels):
        """
        Calculate the base entropy and cumulative counts for selected labels.
        :param selected_labels: Labels.
        :return: Base entropy, left cumulative count, left probability, right cumulative count, right probability.
        """
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
    def _calculate_metrices(filter_sorted_labels, left_labels_cumcount, right_labels_cumcount, left_prob, right_prob, base_entropy):
        """
        Calculate information gain for potential split thresholds.
        :param sorted_data: Sorted feature data.
        :param sorted_labels: Sorted labels.
        :param left_labels_cumcount: Cumulative count of labels in the left split.
        :param right_labels_cumcount: Cumulative count of labels in the right split.
        :param left_prob: Probability of the left split.
        :param right_prob: Probability of the right split.
        :param base_entropy: Base entropy before the split.
        :return: Thresholds and information gain for each threshold.
        """
        filter_sorted_labels_t = filter_sorted_labels.T
        filter_sorted_labels_t = np.ascontiguousarray(filter_sorted_labels_t)
        filter_sorted_labels_t_flatten = filter_sorted_labels_t.reshape(-1)
        filter_sorted_labels_t_total_cumsum_flatteen = filter_sorted_labels_t_flatten.cumsum()
        filter_sorted_labels_t_total_cumsum = filter_sorted_labels_t_total_cumsum_flatteen.reshape(filter_sorted_labels_t.shape)
        filter_sorted_labels_total_cumsum = filter_sorted_labels_t_total_cumsum.T
        filter_sorted_labels_total_cumsum = np.ascontiguousarray(filter_sorted_labels_total_cumsum)
        filter_sorted_labels_left_sum = np.concatenate((np.array([0], dtype = np.int64), filter_sorted_labels_total_cumsum[-1,:-1]))
        filter_sorted_labels_cumsum = filter_sorted_labels_total_cumsum - filter_sorted_labels_left_sum
        left_sorted_labels_cumsum = filter_sorted_labels_cumsum[:-1,:]
        left_sorted_labels_cumsum = np.ascontiguousarray(left_sorted_labels_cumsum)
        right_sorted_labels_cumsum = np.sum(filter_sorted_labels[:,0]) - left_sorted_labels_cumsum
        left_sorted_labels_one_rate = left_sorted_labels_cumsum / left_labels_cumcount.reshape(-1,1)
        right_sorted_labels_one_rate = right_sorted_labels_cumsum / right_labels_cumcount.reshape(-1,1)
        left_sorted_labels_zero_rate = 1 - left_sorted_labels_one_rate
        right_sorted_labels_zero_rate = 1 - right_sorted_labels_one_rate
        left_entropy = -left_sorted_labels_zero_rate * np.log2(left_sorted_labels_zero_rate + 1e-9) - left_sorted_labels_one_rate * np.log2(left_sorted_labels_one_rate + 1e-9)
        right_entropy = -right_sorted_labels_zero_rate * np.log2(right_sorted_labels_zero_rate + 1e-9) - right_sorted_labels_one_rate * np.log2(right_sorted_labels_one_rate + 1e-9)
        weighted_entropy = left_prob.reshape(-1,1) * left_entropy + right_prob.reshape(-1,1) * right_entropy
        metrices = base_entropy - weighted_entropy
        return metrices
    
    
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
        intrinsic_value = intrinsic_value.reshape(-1,1)
        return intrinsic_value
    
    
    @staticmethod
    @njit    
    def _get_left_indices_right_indices(full_sorted_indices, best_index, best_feature):
        """
        Get indices for the left and right splits.
        :param full_sorted_indices: Sorted indices of all features.
        :param best_index: Index of the best split point.
        :param best_feature: Index of the best feature to split.
        :return: Left and right split indices.
        """
        left_indices = full_sorted_indices[:(best_index+1), best_feature]
        left_indices = np.ascontiguousarray(left_indices)
        right_indices = full_sorted_indices[(best_index+1):, best_feature]
        right_indices = np.ascontiguousarray(right_indices)
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
        probs = np.empty((X.shape[0]))
        
        for i in range(len(self.tree)):
            node = self.tree[i]
            feature = node.get('feature', None)
            threshold = node.get('threshold', None)
            left = node.get('left', None)
            right = node.get('right', None)
            prob = node.get('prob', None)
            DecisionTreeZhoumath._goto_next_node(feature, threshold, left, right, prob,
                                                 X, indices, current_node, probs, i)
        
        probs_stacked = np.vstack([1- probs, probs]).T
        return probs_stacked
    
    
    @staticmethod
    @njit
    def _goto_next_node(feature, threshold, left, right, prob, X, indices, current_node, probs, i):
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
        :param probs: Probabilities for each sample.
        :param i: Index of the current node.
        """
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
            
    
    