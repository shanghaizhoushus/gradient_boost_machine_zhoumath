# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 15:04:48 2024

@author: zhoushus
"""

# Import Packages
import numpy as np
import pandas as pd
import warnings


#Setting
warnings.filterwarnings("ignore")


def calRankingByFreq2(df, label, score="score", bins=10):
    """
    Calculate the ranking of a model score based on frequency bins.

    :param df: DataFrame containing the data.
    :param label: Column name for the target label.
    :param score: Column name for the model score. Default is "score".
    :param bins: Number of bins to divide the scores into. Default is 10.
    :return: DataFrame with ranking statistics.
    """
    
    df = df.copy()
    _, bin_edges = pd.qcut(df[score], q=bins, retbins=True, duplicates="drop")
    bin_edges[0], bin_edges[-1] = -np.inf, np.inf
    df["group"] = pd.cut(df[score], bins=bin_edges, include_lowest=True)
    total_bad = df[label].sum()
    total_samples = len(df)
    total_good = total_samples - total_bad
    avg_y = total_bad / total_samples
    grouped = df.groupby("group").agg(
        sample_count=(score, "count"),
        bad_count=(label, "sum"),
        score_max=(score, "max"),
        score_min=(score, "min"),
    )
    grouped["sample_pct"] = grouped["sample_count"] / total_samples
    grouped["bad_pct"] = grouped["bad_count"] / total_bad
    grouped["bad_rate"] = grouped["bad_count"] / grouped["sample_count"]
    grouped["lift"] = grouped["bad_rate"] / avg_y
    grouped["good_count"] = grouped["sample_count"] - grouped["bad_count"]
    grouped["good_pct"] = grouped["good_count"] / total_good
    grouped["cumulative_bad_pct"] = grouped["bad_pct"].cumsum()
    grouped["cumulative_good_pct"] = grouped["good_pct"].cumsum()
    grouped["ks"] = (grouped["cumulative_bad_pct"] - grouped["cumulative_good_pct"]).abs()
    grouped = grouped.reset_index().sort_index()
    grouped["score_range"] = grouped.index + 1
    grouped["score_range"] = grouped["score_range"].apply(lambda x: f"{x:03d}") + "-" + grouped["group"].astype(str)
    overall = pd.DataFrame({
        "score_range": ["Overall"],
        "sample_count": [total_samples],
        "bad_count": [total_bad],
        "sample_pct": [1],
        "bad_pct": [1],
        "bad_rate": [avg_y],
        "lift": [1],
        "score_max": [df[score].max()],
        "score_min": [df[score].min()],
        "ks": [grouped["ks"].max()],
    })
    result = pd.concat([grouped.reset_index(drop=True), overall], ignore_index=True)
    result = result[[
        "score_range", "score_max", "score_min", "sample_count", "bad_count", "sample_pct",
        "bad_pct", "bad_rate", "lift", "ks"
    ]]
    
    return result

