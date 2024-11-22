# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 14:52:54 2024

@author: zhoushus
"""


#import packages
import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from numba import njit
script_dir = os.path.abspath(os.path.join(os.getcwd(), '../../scripts/decision_tree_zhoumath'))
sys.path.insert(0, script_dir)
script_dir = os.path.abspath(os.path.join(os.getcwd(), '../../examples'))
sys.path.insert(0, script_dir)
from cal_ranking_by_freq import calRankingByFreq2
from decision_tree_zhoumath import DecisionTreeZhoumath


#Warmup njit
warmup_data = np.array([0,1])
@njit
def my_function(data):
    data2 = np.mean(data)
    return data2
data2 = my_function(warmup_data)


# Setting
warnings.filterwarnings("ignore", category=UserWarning)


# Load dataset
data = load_breast_cancer(as_frame=True)
X, y = np.array(data.data), np.array(data.target)
'''
data = pd.read_csv('../../../HIGGS.csv', header = None, nrows = 110000)
X = data.iloc[:, 1:]
y = data.iloc[:, 0]
'''


# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# Train the decision tree
max_depth = 3
split_criterion = 'gain'
search_method = 'dfs'
tree_model = DecisionTreeZhoumath(split_criterion=split_criterion,
                                  search_method=search_method,
                                  max_depth=max_depth)
tic = time.time()
tree_model.fit(X_train, y_train)
toc = time.time()
gap = toc-tic
print(f'The decision-tree-zhoumath model is bulit in {gap:.5f} seconds.')


# Train the decision tree
tic = time.time()
tree_model_sklearn = DecisionTreeClassifier(max_depth=max_depth)
tree_model_sklearn.fit(X_train, y_train)
toc = time.time()
gap = toc-tic
print(f'The decision-tree-sklearn model is bulit in {gap:.5f} seconds.')


# Predict and evaluate
y_test_pred = tree_model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_test_pred)
fpr, tpr, _ = roc_curve(y_test, y_test_pred)
ks = tpr[abs(tpr - fpr).argmax()] - fpr[abs(tpr - fpr).argmax()]
print(f"KS = {ks:.3f}\nAUC = {auc_score:.3f}")


# Plot ROC Curve
plt.plot(fpr, fpr, label="Random Guess")
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.3f})")
plt.plot(
    [fpr[abs(tpr - fpr).argmax()]] * len(fpr),
    np.linspace(fpr[abs(tpr - fpr).argmax()], tpr[abs(tpr - fpr).argmax()], len(fpr)),
    "--",
)
plt.title("ROC Curve of Decision Tree")
plt.legend()
plt.show()


# Replace feature indices with column names
df = pd.DataFrame(columns=data.data.columns[:], data=X_test)
col_names = df.columns
tree_model.replace_features_with_column_names(col_names)
print("Constructed Decision Tree:", tree_model.tree)


# Frequency-based ranking
df["y"] = y_test
df["y_pred"] = y_test_pred
tmp = calRankingByFreq2(df, label="y", score="y_pred", bins=10)
print(tmp)


