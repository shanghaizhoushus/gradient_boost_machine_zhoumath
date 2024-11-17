# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 12:50:56 2024

@author: zhoushus
"""


#import packages
import numpy as np
import pandas as pd
from collections import Counter
import warnings


# Setting
warnings.filterwarnings("ignore", category=UserWarning)


# DecisionTree class
class DecisionTreeZhoumath:
    def __init__(self, max_depth=None):
        """
        Initialize the decision tree.
        :param max_depth: Maximum depth of the tree.
        """
        
        self.max_depth = max_depth
        self.tree = None

    def calculate_entropy(self, labels):
        """
        Calculate entropy for a given label distribution.
        :param labels: Array of labels.
        :return: Entropy value.
        """
        
        label_counts = Counter(labels)
        entropy = 0.0

        for label, count in label_counts.items():
            prob = count / len(labels)
            entropy -= prob * np.log2(prob) if prob > 0 else 0

        return entropy

    def split_dataset(self, data, labels, feature_index, threshold):
        """
        Split the dataset based on a feature and threshold.
        :param data: Feature data.
        :param labels: Labels.
        :param feature_index: Feature index to split on.
        :param threshold: Threshold value.
        :return: Split data and labels.
        """
        
        left_mask = data[:, feature_index] <= threshold
        right_mask = ~left_mask
        left_data, left_labels = data[left_mask], labels[left_mask]
        right_data, right_labels = data[right_mask], labels[right_mask]
        
        return left_data, left_labels, right_data, right_labels

    def choose_best_split(self, data, labels):
        """
        Choose the best feature and threshold for splitting.
        :param data: Feature data.
        :param labels: Labels.
        :return: Best feature index, threshold, and information gain.
        """
        
        num_features = data.shape[1]
        base_entropy = self.calculate_entropy(labels)
        best_info_gain = -1
        best_feature = None
        best_threshold = None

        for feature_index in range(num_features):
            thresholds = np.unique(data[:, feature_index])

            for threshold in thresholds:
                left_data, left_labels, right_data, right_labels = self.split_dataset(
                    data, labels, feature_index, threshold
                )

                if len(left_labels) == 0 or len(right_labels) == 0:
                    continue

                prob_left = len(left_labels) / len(labels)
                prob_right = len(right_labels) / len(labels)
                new_entropy = prob_left * self.calculate_entropy(
                    left_labels
                ) + prob_right * self.calculate_entropy(right_labels)
                info_gain = base_entropy - new_entropy

                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold, best_info_gain

    def build_tree(self, data, labels, depth=0):
        """
        Recursively build the decision tree.
        :param data: Feature data.
        :param labels: Labels.
        :param depth: Current depth of the tree.
        :return: A decision tree dictionary.
        """
        
        if len(set(labels)) == 1:
            return {"prob": [1.0, 0.0]} if labels[0] == 0 else {"prob": [0.0, 1.0]}

        if self.max_depth is not None and depth >= self.max_depth:
            label_counts = Counter(labels)
            prob_0 = label_counts[0] / len(labels)
            prob_1 = label_counts[1] / len(labels)
            return {"prob": [prob_0, prob_1]}

        if len(data) == 0 or data.shape[1] == 0:
            label_counts = Counter(labels)
            prob_0 = label_counts[0] / len(labels)
            prob_1 = label_counts[1] / len(labels)
            return {"prob": [prob_0, prob_1]}

        best_feature, best_threshold, best_info_gain = self.choose_best_split(data, labels)

        if best_feature is None or best_info_gain <= 0:
            label_counts = Counter(labels)
            prob_0 = label_counts[0] / len(labels)
            prob_1 = label_counts[1] / len(labels)
            return {"prob": [prob_0, prob_1]}

        left_data, left_labels, right_data, right_labels = self.split_dataset(
            data, labels, best_feature, best_threshold
        )
        tree = {
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self.build_tree(left_data, left_labels, depth + 1),
            "right": self.build_tree(right_data, right_labels, depth + 1),
        }

        return tree

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

    def fit(self, data, labels):
        """
        Train the decision tree.
        :param data: Feature data.
        :param labels: Labels.
        """
        self.tree = self.build_tree(data, labels)

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

