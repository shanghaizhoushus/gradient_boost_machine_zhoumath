# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 12:50:56 2024

@author: zhoushus
"""

#import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from collections import Counter
import warnings
from cal_ranking_by_freq import calRankingByFreq2


#Setting
warnings.filterwarnings("ignore", category=UserWarning)


# 1. Calculate entropy
def calculate_entropy(labels):
    
    label_counts = Counter(labels)
    entropy = 0.0
    
    for label, count in label_counts.items():
        prob = count / len(labels)
        entropy -= prob * np.log2(prob) if prob > 0 else 0
        
    return entropy


# 2. Split dataset based on a feature and a threshold
def split_dataset(data, labels, feature_index, threshold):
    
    left_mask = data[:, feature_index] <= threshold
    right_mask = ~left_mask
    left_data, left_labels = data[left_mask], labels[left_mask]
    right_data, right_labels = data[right_mask], labels[right_mask]
    return left_data, left_labels, right_data, right_labels


# 3. Choose the best feature and threshold to split
def choose_best_split(data, labels):
    
    num_features = data.shape[1]
    base_entropy = calculate_entropy(labels)
    best_info_gain = -1
    best_feature = None
    best_threshold = None
    
    for feature_index in range(num_features):
        thresholds = np.unique(data[:, feature_index])
        
        for threshold in thresholds:
            left_data, left_labels, right_data, right_labels = split_dataset(data,
                                                                             labels,
                                                                             feature_index,
                                                                             threshold)
            
            if len(left_labels) == 0 or len(right_labels) == 0:
                continue
            
            prob_left = len(left_labels) / len(labels)
            prob_right = len(right_labels) / len(labels)
            new_entropy = prob_left * calculate_entropy(left_labels) + prob_right * calculate_entropy(right_labels)
            info_gain = base_entropy - new_entropy
            
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = feature_index
                best_threshold = threshold
    
    return best_feature, best_threshold, best_info_gain


# 4. Build the decision tree recursively
def build_tree(data, labels, depth=0, max_depth=None):
    
    if len(set(labels)) == 1:
        return {'prob': [1.0, 0.0]} if labels[0] == 0 else {'prob': [0.0, 1.0]}
    
    if max_depth is not None and depth >= max_depth:
        label_counts = Counter(labels)
        prob_0 = label_counts[0] / len(labels)
        prob_1 = label_counts[1] / len(labels)
        return {'prob': [prob_0, prob_1]}
    
    if len(data) == 0 or data.shape[1] == 0:
        label_counts = Counter(labels)
        prob_0 = label_counts[0] / len(labels)
        prob_1 = label_counts[1] / len(labels)
        return {'prob': [prob_0, prob_1]}
    
    best_feature, best_threshold, best_info_gain = choose_best_split(data, labels)
    
    if best_feature is None or best_info_gain <= 0:
        label_counts = Counter(labels)
        prob_0 = label_counts[0] / len(labels)
        prob_1 = label_counts[1] / len(labels)
        return {'prob': [prob_0, prob_1]}
    
    left_data, left_labels, right_data, right_labels = split_dataset(data,
                                                                     labels,
                                                                     best_feature,
                                                                     best_threshold)
    tree = {
        'feature': best_feature,
        'threshold': best_threshold,
        'left': build_tree(left_data, left_labels, depth + 1, max_depth),
        'right': build_tree(right_data, right_labels, depth + 1, max_depth)
    }
    
    return tree


# 5. Predict probabilities for a single sample
def predict_proba_single(tree, sample):
    
    if 'prob' in tree:
        return tree['prob']
    
    feature = tree['feature']
    threshold = tree['threshold']
    
    if sample[feature] <= threshold:
        return predict_proba_single(tree['left'], sample)
    else:
        return predict_proba_single(tree['right'], sample)


# 6. Predict probabilities for a batch of samples
def predict_proba(tree, data):
    
    pred_np = np.array([predict_proba_single(tree, sample) for sample in data])
    pred_pd = pd.DataFrame(pred_np, columns = [0, 1])
    
    return pred_pd

# Recursive function to replace feature indices with column names
def replace_features_with_column_names(tree, column_names):
    
    if 'feature' in tree:
        tree['feature'] = column_names[tree['feature']]

        if 'left' in tree:
            replace_features_with_column_names(tree['left'], column_names)
        if 'right' in tree:
            replace_features_with_column_names(tree['right'], column_names)
    return tree

#-------------------------

data = load_breast_cancer(as_frame=True)
X, y = np.array(data.data), np.array(data.target)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
max_depth = 3
decision_tree = build_tree(X,
                           y,
                           max_depth=max_depth)

#print("Constructed Decision Tree:", decision_tree)

y_pred = predict_proba(decision_tree, X).iloc[:,1]
auc_score = roc_auc_score(y, y_pred)
fpr, tpr, _ = roc_curve(y, y_pred)
ks = tpr[abs(tpr - fpr).argmax()] - fpr[abs(tpr - fpr).argmax()]

print(f"KS = {ks:.3f}\nAUC = {auc_score:.3f}")

plt.plot(fpr, fpr, label="Random Guess")
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.3f})")
plt.plot([fpr[abs(tpr - fpr).argmax()]] * len(fpr), np.linspace(fpr[abs(tpr - fpr).argmax()], tpr[abs(tpr - fpr).argmax()], len(fpr)), "--")
plt.title("ROC Curve of Decision Tree")
plt.legend()
plt.show()

#-------------------------

df2 = data.data
col_names = df2.columns.tolist()
decision_tree = replace_features_with_column_names(decision_tree, col_names)

print("Constructed Decision Tree:", decision_tree)

df2["y"] = data.target
df2["y_pred"] = y_pred
tmp = calRankingByFreq2(df2, label="y", score="y_pred", bins=10)

print(tmp)

