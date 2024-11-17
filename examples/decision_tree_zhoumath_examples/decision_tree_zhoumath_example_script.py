# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 14:52:54 2024

@author: zhoushus
"""


#import packages
import sys
import os
import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
script_dir = os.path.abspath(os.path.join(os.getcwd(), '../../scripts/decision_tree_zhoumath'))
sys.path.insert(0, script_dir)
script_dir = os.path.abspath(os.path.join(os.getcwd(), '../../examples'))
sys.path.insert(0, script_dir)
from cal_ranking_by_freq import calRankingByFreq2
from decision_tree_zhoumath import DecisionTreeZhoumath


# Setting
warnings.filterwarnings("ignore", category=UserWarning)


# Load dataset
data = load_breast_cancer(as_frame=True)
X, y = np.array(data.data), np.array(data.target)

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree
max_depth = 3
split_criterion = 'gain'
tree_model = DecisionTreeZhoumath(split_criterion=split_criterion, max_depth=max_depth)
tic = time.time()
tree_model.fit(X_train, y_train)
toc = time.time()
gap = toc-tic
print(f'The decision-tree-zhoumath model with max depth {max_depth} and split criterion {split_criterion} is bulit in {gap:.3f} seconds.')

# Predict and evaluate
y_test_pred = tree_model.predict_proba(X_test).iloc[:, 1]
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
df = pd.DataFrame(columns=data.data.columns, data=X_test)
col_names = df.columns
tree_model.replace_features_with_column_names(col_names)
print("Constructed Decision Tree:", tree_model.tree)

# Frequency-based ranking
df["y"] = y_test
df["y_pred"] = y_test_pred
tmp = calRankingByFreq2(df, label="y", score="y_pred", bins=10)
print(tmp)

