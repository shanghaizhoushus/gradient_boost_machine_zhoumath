# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 14:52:54 2024

@author: zhoushus
"""

#Import packages
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
from numba import njit
script_dir = os.path.abspath(os.path.join(os.getcwd(), '../../scripts/decision_tree_zhoumath'))
sys.path.insert(0, script_dir)
script_dir = os.path.abspath(os.path.join(os.getcwd(), '../../scripts/random_forest_zhoumath'))
sys.path.insert(0, script_dir)
script_dir = os.path.abspath(os.path.join(os.getcwd(), '../../examples'))
sys.path.insert(0, script_dir)
from cal_ranking_by_freq import calRankingByFreq2
from random_forest_zhoumath import RandomForestZhoumath
np.random.seed(42)

#Warmup njit
warmup_data = np.array([0,1])
@njit
def warmup_function(data):
    data2 = np.mean(data)
    return data2
data2 = warmup_function(warmup_data)

# Setting
warnings.filterwarnings("ignore", category=UserWarning)

#----------------------

# Load dataset
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
'''
data = pd.read_csv('../../../HIGGS.csv', header = None, nrows = 11000)
X = data.iloc[:, 1:]
y = data.iloc[:, 0]
'''

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train the decision tree with null handled
X_train = np.array(X_train)
mask = np.random.uniform(size = X_train.shape) > 1
X_train[mask] = np.nan
num_base_trees = 1000
rf_column_rate = 0.5
rf_sample_rate = 0.5
max_depth = 5
split_criterion = 'entropy_gain'
search_method = 'bfs'
random_sample_rate = 1
random_forest_model = RandomForestZhoumath(num_base_trees=num_base_trees,
                                           rf_column_rate=rf_column_rate,
                                           rf_sample_rate=rf_sample_rate,
                                           max_depth=max_depth,
                                           split_criterion=split_criterion,
                                           search_method=search_method,
                                           verbose_for_tree=False,
                                           verbose_for_rf=True)
tic = time.time()
random_forest_model.fit(data=X_train,
                        labels=y_train,
                        val_data=X_val,
                        val_labels=y_val,
                        early_stop_rounds_for_tree=None,
                        early_stop_rounds_for_forest=10)
toc = time.time()
gap = toc-tic
print(f'The decision-tree-zhoumath-with-null-zhoumath model is bulit in {gap:.5f} seconds.')

# Predict and evaluate
X_test = np.array(X_test)
mask = np.random.uniform(size = X_test.shape) > 1
X_test[mask] = np.nan
tic = time.time()
y_test_pred = random_forest_model.predict_proba(X_test)[:, 1]
toc = time.time()
gap = toc-tic
print(f'The decision-tree-with-null-zhoumath model is predicted in {gap:.5f} seconds.')
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

# Get feature importances
feature_importances_df = random_forest_model.feature_importances.get_feature_importances_df(data.data.columns)

# Replace feature indices with column names
df = pd.DataFrame(X_test)
col_names = data.data.columns.tolist()
random_forest_model.replace_features_with_column_names(col_names)
print("Decision Tree is constructed successfully.")

# Frequency-based ranking
df["y"] = y_test
df["y_pred"] = y_test_pred
tmp = calRankingByFreq2(df, label="y", score="y_pred", bins=10)
print(tmp)

#Save model to a pkl
random_forest_model.to_pkl("tree_model_zhoumath.pkl") 
