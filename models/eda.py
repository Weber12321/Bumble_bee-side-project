import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_regression

def read_json(filename):
    return pd.read_json(filename, encoding='utf-8')

def preprocessing(df: pd.DataFrame):
    df = df[~df.isna().any(axis=1)]
    df = df.astype('float')
    arr = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1].values
    return arr, y

def top_k(arr, y):
    top_k = SelectKBest(score_func=f_regression, k=10)
    X = top_k.fit_transform(arr, y)
    return X

def anova(X: np.array):
    t = (i for i in X)
    f_value, p_value = stats.f_oneway(*t)
    return f'f-value : {f_value}\np-value : {p_value}'

def run_stats(filename):
    X, y  = preprocessing(read_json(filename))
    return anova(top_k(X,y))

def run_k_means(filename):
    X, y = preprocessing(read_json(filename))
    k_means = KMeans(n_clusters=5, random_state=0).fit(X)
    return k_means.labels_

