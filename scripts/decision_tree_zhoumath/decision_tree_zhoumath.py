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
    def __init__(self, row_indices=None, depth=None, parent_sorted_indices=None, parent_null_indices=None,
                 parent_index=None, node_index=None, child_direction=None):
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
        self.parent_sorted_indices = parent_sorted_indices
        self.parent_null_indices = parent_null_indices
        self.parent_index = parent_index
        self.node_index = node_index
        self.child_direction = child_direction


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
        self.left_indices, self.right_indices = DecisionTreeZhoumath._get_left_right_indices(filtered_sorted_indices,
                                                                                             self.best_index,
                                                                                             self.best_feature)
        sorted_best_feature_data = data[:, self.best_feature][filtered_sorted_indices[:, self.best_feature]]
        sorted_best_feature_data = np.ascontiguousarray(sorted_best_feature_data)
        self.best_threshold = np.round((sorted_best_feature_data[self.best_index] +
                                        sorted_best_feature_data[self.best_index + 1]) / 2, 7)

    def _renew_best_status_null(self, metrics, filtered_sorted_indices, sorted_data, feature_index, null_direction):
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

    def _finalize_best_status_null(self, filtered_sorted_indices, filtered_null_indices, selected_labels):
        """
        Finalize the best status considering null values.
        :param filtered_sorted_indices: Filtered sorted indices for each feature.
        :param filtered_null_indices: Null value indices for each feature.
        :param selected_labels: Labels for the selected data.
        :return: True if no valid split is found, otherwise False.
        """
        null_indices = filtered_null_indices[self.best_feature]
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
            left_sorted_indices = filtered_sorted_indices[self.best_feature][:(self.best_index - null_shape + 1)]
            self.left_indices = np.concatenate([left_sorted_indices, null_indices])
            self.right_indices = filtered_sorted_indices[self.best_feature][(self.best_index - null_shape + 1):]
        else:
            self.left_indices = filtered_sorted_indices[self.best_feature][:(self.best_index + 1)]
            self.right_indices = np.concatenate([filtered_sorted_indices[self.best_feature][(self.best_index + 1):],
                                                 null_indices])

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


# DecisionTree class
class DecisionTreeZhoumath:
    def __init__(self, split_criterion, search_method, max_depth=None, pos_weight=1):
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
        self.tree = None
        self.gini = 0

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
        if early_stop_rounds and self.search_method != 'bfs':
            raise ValueError("Early Stopping requires 'bfs' as the search method.")

        if np.any(np.isnan(data)):
            tree_with_null = DecisionTreeWithNullZhoumath(
                split_criterion=self.split_criterion,
                search_method=self.search_method,
                max_depth=self.max_depth
            )
            tree_with_null.fit(data, labels, val_data, val_labels, early_stop_rounds, random_state)
            self.tree = tree_with_null.tree
            return True

        if (val_data is not None) and (val_labels is not None):
            print("Early stop mode is opened. Search method can only be BFS.")
            early_stopper = EarlyStopper(val_data=val_data,
                                         val_labels=val_labels,
                                         early_stop_rounds=early_stop_rounds)
            self.search_method = 'bfs'
            self.best_tree = []
        else:
            early_stopper = None

        data = np.ascontiguousarray(data)
        data = DecisionTreeZhoumath._add_perturbation(data, random_state)
        labels = np.ascontiguousarray(labels.astype(np.int32))
        self.data = data
        self.labels = labels
        self.tree = self._build_tree(early_stopper)

        if early_stopper is not None:
            self.tree = self.best_tree

        self.data = None
        self.labels = None
        return True

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
        root_collection_node = CollectionNode(depth=0,
                                              row_indices=np.arange(self.labels.shape[0]),
                                              parent_sorted_indices=np.ascontiguousarray(np.argsort(self.data, axis=0)))

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
                    return True

            if (self.max_depth is not None and current_node.depth >= self.max_depth) or np.unique(labels).size == 1:
                probabilities = DecisionTreeZhoumath._calculate_probabilities(labels)
                current_tree_node = TreeNode(prob=probabilities)
                DecisionTreeZhoumath._add_node_to_tree(tree, current_node, current_tree_node)
                continue

            current_best_status, filtered_sorted_indices = self._choose_best_split(current_node)

            if current_best_status.best_feature is None or current_best_status.best_metric <= 0:
                probabilities = DecisionTreeZhoumath._calculate_probabilities(labels)
                current_tree_node = TreeNode(prob=probabilities)
                DecisionTreeZhoumath._add_node_to_tree(tree, current_node, current_tree_node)
                continue

            current_tree_node = TreeNode(feature=current_best_status.best_feature,
                                         threshold=current_best_status.best_threshold)

            if early_stopper is not None:
                probabilities = DecisionTreeZhoumath._calculate_probabilities(labels)
                current_tree_node.prob = probabilities

            DecisionTreeZhoumath._add_node_to_tree(tree, current_node, current_tree_node)
            right_node = CollectionNode(row_indices=current_best_status.right_indices,
                                        depth=current_node.depth + 1,
                                        parent_sorted_indices=filtered_sorted_indices,
                                        parent_index=current_node.node_index,
                                        child_direction="right")
            left_node = CollectionNode(row_indices=current_best_status.left_indices,
                                       depth=current_node.depth + 1,
                                       parent_sorted_indices=filtered_sorted_indices,
                                       parent_index=current_node.node_index,
                                       child_direction="left")
            collection.append(right_node)
            collection.append(left_node)

        if early_stopper is not None:
            early_stop_triggered = early_stopper._evaluate_early_stop(self, current_node, tree)

            if early_stop_triggered:
                return True

        return tree

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
    def _add_node_to_tree(tree, current_node, current_tree_node):
        """
        Helper function to add a node to the tree structure.
        :param tree: The tree list.
        :param current_node: Current node being processed.
        :param current_tree_node: Node to be added.
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
        Determine the best split for the current node.
        :param current_node: Current node being processed.
        :return: Best split status and filtered sorted indices.
        """
        current_best_status = BestStatus()

        if current_node.depth == 0:
            filtered_sorted_indices = current_node.parent_sorted_indices
        else:
            filtered_sorted_indices = DecisionTreeZhoumath._filter_sorted_indices(current_node)

        selected_labels = np.ascontiguousarray(self.labels[current_node.row_indices])
        base_metric = DecisionTreeZhoumath._calculate_base_metric(selected_labels, self.pos_weight, self.gini)

        filtered_sorted_labels = np.ascontiguousarray(self.labels[filtered_sorted_indices])
        metrics = DecisionTreeZhoumath._calculate_metrics(filtered_sorted_labels, base_metric, self.pos_weight, self.gini)

        if self.split_criterion == 'entropy_gain_ratio':
            intrinsic_value = DecisionTreeZhoumath._calculate_intrinsic_value(selected_labels)
            metrics = metrics / intrinsic_value

        current_best_status._renew_best_status(metrics, filtered_sorted_indices, self.data)

        if current_best_status.best_threshold <= 0:
            return None, None

        return current_best_status, filtered_sorted_indices

    @staticmethod
    def _filter_sorted_indices(current_node):
        """
        Retrieve the sorted indices for the given rows.
        :param current_node: Current node being processed.
        :return: Filtered sorted indices for all features.
        """
        sorted_indices_flattened = DecisionTreeZhoumath._flatten_sorted_indices(current_node.parent_sorted_indices)
        filtered_sorted_indices_flattened = sorted_indices_flattened[np.in1d(sorted_indices_flattened,
                                                                            current_node.row_indices)]
        return DecisionTreeZhoumath._unflatten_sorted_indices(current_node.parent_sorted_indices,
                                                              filtered_sorted_indices_flattened)

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

        if gini == 0:
            return -zero_rate * np.log2(zero_rate + 1e-9) - pos_weight * one_rate * np.log2(one_rate + 1e-9)
        else:
            zero_rate = zero_rate + 1e-9
            one_rate = one_rate + 1e-9
            zero_rate = zero_rate / (zero_rate + pos_weight * one_rate)
            one_rate = 1 - zero_rate
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
        left_probs = (np.arange(1, num_rows) / num_rows).reshape(-1, 1)
        right_probs = (np.arange(num_rows - 1, 0, -1) / num_rows).reshape(-1, 1)

        if gini == 0:
            left_entropy = -left_zero_rate * np.log2(left_zero_rate + 1e-9) - pos_weight * left_one_rate * np.log2(left_one_rate + 1e-9)
            right_entropy = -right_zero_rate * np.log2(right_zero_rate + 1e-9) - pos_weight * right_one_rate * np.log2(right_one_rate + 1e-9)
            weighted_entropy = left_probs * left_entropy + right_probs * right_entropy
            return base_metric - weighted_entropy
        else:
            left_zero_rate = left_zero_rate / (left_zero_rate + pos_weight * left_one_rate)
            right_zero_rate = right_zero_rate / (right_zero_rate + pos_weight * right_one_rate)
            left_one_rate = 1 - left_zero_rate
            right_one_rate = 1 - right_zero_rate
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

    def predict_proba(self, data, tree=None):
        """
        Predict probabilities for a batch of samples.
        :param data: Feature data.
        :param tree: Decision tree to use for prediction.
        :return: Probability predictions as a 2D array.
        """
        if tree is None:
            tree = self.tree.copy()

        X = np.ascontiguousarray(data)
        indices = np.arange(X.shape[0], dtype=int)
        current_node = np.zeros((X.shape[0]), dtype=int)
        probabilities = np.zeros((X.shape[0]))

        for i in range(len(tree)):
            node = tree[i]
            current_node, probabilities = DecisionTreeZhoumath._traverse_next_node(node, X, indices, current_node,
                                                                                   probabilities, i)
        return np.vstack([1 - probabilities, probabilities]).T

    @staticmethod
    def _traverse_next_node(node, X, indices, current_node, probabilities, i):
        """
        Navigate to the next node in the decision tree.
        :param node: Current node in the decision tree.
        :param X: Feature data.
        :param indices: Indices of the samples.
        :param current_node: Current node for each sample.
        :param probabilities: Probabilities for each sample.
        :param i: Index of the current node.
        :return: Updated current node and probabilities.
        """
        if node.prob is not None:
            probabilities[indices[current_node == i]] = node.prob[1]

        if node.feature is not None:
            index = indices[current_node == i]
            feature_values = X[index, node.feature]

            if node.null_direction == 'left':
                left_condition = (feature_values <= node.threshold) | np.isnan(feature_values)
            else:
                left_condition = (feature_values <= node.threshold)

            if node.left is not None and node.right is not None:
                current_node[index[left_condition]] = node.left
                current_node[index[~left_condition]] = node.right

        return current_node, probabilities

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


# DecisionTreeWithNullZhoumath extends DecisionTreeZhoumath
class DecisionTreeWithNullZhoumath(DecisionTreeZhoumath):
    def fit(self, data, labels, val_data, val_labels, early_stop_rounds, random_state=42):
        """
        Train the decision tree considering missing values.
        :param data: Feature data.
        :param labels: Labels.
        :param val_data: Validation feature data.
        :param val_labels: Validation labels.
        :param early_stop_rounds: Number of rounds for early stopping.
        :param random_state: Random seed for reproducibility.
        :return: True if the tree is successfully trained.
        """
        if (val_data is not None) and (val_labels is not None):
            print("Early stop mode is opened. Search method can only be BFS.")
            early_stopper = EarlyStopper(val_data=val_data,
                                         val_labels=val_labels,
                                         early_stop_rounds=early_stop_rounds)
            self.search_method = 'bfs'
            self.best_tree = []
        else:
            early_stopper = None

        data = np.ascontiguousarray(data)
        data = DecisionTreeZhoumath._add_perturbation(data, random_state)
        labels = np.ascontiguousarray(labels.astype(np.int32))
        self.data = data
        self.labels = labels
        self.tree = self._build_tree(early_stopper)

        if early_stopper is not None:
            self.tree = self.best_tree

        self.data = None
        self.labels = None
        return True

    def _build_tree(self, early_stopper):
        """
        Recursively build the decision tree considering missing values.
        :param early_stopper: Instance of EarlyStopper for early stopping.
        :return: A decision tree represented as a list of nodes.
        """
        data = self.data
        root_sorted_indices, root_null_indices = DecisionTreeWithNullZhoumath._init_root_indices(data)
        root_collection_node = CollectionNode(depth=0,
                                              row_indices=np.arange(self.labels.shape[0]),
                                              parent_sorted_indices=root_sorted_indices,
                                              parent_null_indices=root_null_indices)

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
                    return True

            if (self.max_depth is not None and current_node.depth >= self.max_depth) or np.unique(labels).size == 1:
                probabilities = DecisionTreeZhoumath._calculate_probabilities(labels)
                current_tree_node = TreeNode(prob=probabilities)
                DecisionTreeZhoumath._add_node_to_tree(tree, current_node, current_tree_node)
                continue

            current_best_status, filtered_sorted_indices, filtered_null_indices = self._choose_best_split(current_node)

            if current_best_status.best_feature is None or current_best_status.best_metric <= 0:
                probabilities = DecisionTreeZhoumath._calculate_probabilities(labels)
                current_tree_node = TreeNode(prob=probabilities)
                DecisionTreeZhoumath._add_node_to_tree(tree, current_node, current_tree_node)
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
                                        parent_sorted_indices=filtered_sorted_indices,
                                        parent_null_indices=filtered_null_indices,
                                        parent_index=current_node.node_index,
                                        child_direction="right")
            left_node = CollectionNode(row_indices=current_best_status.left_indices,
                                       depth=current_node.depth + 1,
                                       parent_sorted_indices=filtered_sorted_indices,
                                       parent_null_indices=filtered_null_indices,
                                       parent_index=current_node.node_index,
                                       child_direction="left")
            collection.append(right_node)
            collection.append(left_node)

        if early_stopper is not None:
            early_stop_triggered = early_stopper._evaluate_early_stop(self, current_node, tree)

            if early_stop_triggered:
                return True

        return tree

    @staticmethod
    def _init_root_indices(data):
        """
        Initialize root indices for handling missing values.
        :param data: Feature data.
        :return: Root sorted indices and root null indices.
        """
        root_sorted_indices = np.zeros(shape=data.shape, dtype=np.int32)
        root_null_indices = np.array([np.where(np.isnan(data[:, i]))[0] for i in range(data.shape[1])], dtype=object)
        null_quantities = np.array([root_null_indices[i].shape[0] for i in range(root_null_indices.shape[0])])
        data_no_nan = np.where(np.isnan(data), np.inf, data)
        root_sorted_indices_with_null = np.argsort(data_no_nan, axis=0)
        root_sorted_indices = np.array([
            root_sorted_indices_with_null[:-null_quantities[i], i] for i in range(root_null_indices.shape[0])
        ], dtype=object)

        return root_sorted_indices, root_null_indices

    def _choose_best_split(self, current_node):
        """
        Determine the best split for the current node, considering missing values.
        :param current_node: Current node being processed.
        :return: Best split status, filtered sorted indices, and filtered null indices.
        """
        current_best_status = BestStatus(best_feature=None,
                                         best_index=0,
                                         best_metric=0,
                                         best_threshold=None,
                                         best_null_direction=None)
        num_features = self.data.shape[1]

        if current_node.depth == 0:
            filtered_sorted_indices = current_node.parent_sorted_indices
            filtered_null_indices = current_node.parent_null_indices
        else:
            filtered_sorted_indices, filtered_null_indices = DecisionTreeWithNullZhoumath._filter_sorted_indices(current_node)

        selected_labels = np.ascontiguousarray(self.labels[current_node.row_indices])
        base_entropy = DecisionTreeZhoumath._calculate_base_metric(selected_labels, self.pos_weight, self.gini)

        if self.split_criterion == 'entropy_gain_ratio':
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
                metrics = DecisionTreeWithNullZhoumath._calculate_metrics(sorted_labels, base_entropy, self.pos_weight, self.gini)

                if self.split_criterion == 'entropy_gain_ratio':
                    intrinsic_value = intrinsic_value.reshape(-1)
                    metrics = metrics / intrinsic_value

                current_best_status._renew_best_status_null(metrics, filtered_sorted_indices, sorted_data, feature_index,
                                                            null_direction)

        no_best_split = current_best_status._finalize_best_status_null(filtered_sorted_indices, filtered_null_indices,
                                                                       selected_labels)

        if no_best_split:
            filtered_sorted_indices = None
            filtered_null_indices = None

        return current_best_status, filtered_sorted_indices, filtered_null_indices

    @staticmethod
    def _filter_sorted_indices(current_nodes):
        """
        Retrieve the sorted indices for the given rows, considering missing values.
        :param current_nodes: Current node being processed.
        :return: Filtered sorted and null indices for all features.
        """
        filtered_sorted_indices = []
        filtered_null_indices = []

        for i in range(current_nodes.parent_sorted_indices.shape[0]):
            parent_sorted_indices_i = current_nodes.parent_sorted_indices[i]
            filtered_sorted_indices_i = parent_sorted_indices_i[np.in1d(parent_sorted_indices_i, current_nodes.row_indices)]
            filtered_sorted_indices.append(np.ascontiguousarray(filtered_sorted_indices_i))
            parent_null_indices_i = current_nodes.parent_null_indices[i]
            filtered_null_indices_i = parent_null_indices_i[np.in1d(parent_null_indices_i, current_nodes.row_indices)]
            filtered_null_indices.append(np.ascontiguousarray(filtered_null_indices_i))

        return np.array(filtered_sorted_indices, dtype=object), np.array(filtered_null_indices, dtype=object)

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
        left_probs = (np.arange(1, num_rows) / num_rows)
        right_probs = (np.arange(num_rows - 1, 0, -1) / num_rows)

        if gini == 0:
            left_entropy = -left_zero_rate * np.log2(left_zero_rate + 1e-9) - pos_weight *  left_one_rate * np.log2(left_one_rate + 1e-9)
            right_entropy = -right_zero_rate * np.log2(right_zero_rate + 1e-9) - pos_weight * right_one_rate * np.log2(right_one_rate + 1e-9)
            weighted_entropy = left_probs * left_entropy + right_probs * right_entropy
            return base_metric - weighted_entropy
        else:
            left_zero_rate = left_zero_rate / (left_zero_rate + pos_weight * left_one_rate)
            right_zero_rate = right_zero_rate / (right_zero_rate + pos_weight * right_one_rate)
            left_one_rate = 1 - left_zero_rate
            right_one_rate = 1 - right_zero_rate
            left_gini = 1 - left_zero_rate ** 2 - left_one_rate ** 2
            right_gini = 1 - right_zero_rate ** 2 - right_one_rate ** 2
            weighted_gini = left_probs * left_gini + right_probs * right_gini
            return base_metric - weighted_gini
    
    