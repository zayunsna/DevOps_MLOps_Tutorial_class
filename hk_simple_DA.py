#!/usr/bin/env python
# -*- coding: utf-8 -*-
########
# This Library contains the basic tool for Feature Engineering of Data Analysis 
# Of course, all function already developed more famous package such as sklearn, pandas, numpy, scipy or so on..
# But, In order to study basic concept of each algorithms or methods, I built this function.
# And.. who knows, it might help with data analysis and make it faster :)
########

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
import seaborn as sns

def elastic_naming(form:str, n:int):
    names = []
    for i in range(n):
        name = form + '_' + str(i)
        names.append(name)
    return names

################################################################################################################################
#### This part is making the dataframe sample for the function test.
# Generating a dataset with 3 features
def make_testdf(n_samples:int, nFeatures:int):
    X, _ = make_blobs(n_samples=n_samples, centers=3, n_features=nFeatures, random_state=42)
    column_names = elastic_naming(form='Feature', n=nFeatures)
    df = pd.DataFrame(X, columns=column_names)
    print(df.head())
    # Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled
################################################################################################################################


## PCA
def applyPCA(dataset:pd.DataFrame, n_components:int=2, drawing:bool = False):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(dataset)
    
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data_scaled)

    # Explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained Variance Ratio: {explained_variance}")
    # Cumulative explained variance
    cumulative_explained_variance = np.cumsum(explained_variance)
    print(f"Cumulative Explained Variance: {cumulative_explained_variance}")

    column_names = elastic_naming(form='Principal Component', n=n_components)
    # pca_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])
    pca_df = pd.DataFrame(data=principal_components, columns=column_names)

    if drawing:
        plt.figure(figsize=(14,6))
        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, align='center', label='Individual explained variance')
        plt.step(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, where='mid', label='Cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal component index')
        plt.legend(loc='best')
        plt.title('Scree Plot')

        plt.subplot(1, 2, 2)
        plt.scatter(pca_df[column_names[0]], pca_df[column_names[1]])
        plt.xlabel(column_names[0])
        plt.ylabel(column_names[1])
        plt.title('2 component PCA')

        plt.tight_layout()
        plt.show()

    return pca_df, principal_components, pca

df_scaled = make_testdf(n_samples=1000, nFeatures=5)
pca_df, pc, pca = applyPCA(dataset=df_scaled, n_components=5, drawing=True)
print(pca_df.head())


## GMM (with K-Means)

## Simple Linear Regression

## Multiple Linear Regression

## Lasso : If a specific feature is important than all, Lasso might gives better analysis result.

## Ridge : If all feature is important uniformly, Ridge can gives better analysis performance than Lasso.