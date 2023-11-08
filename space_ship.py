#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Below code is written by Gusthema (Owner) and elliot robot (Editer) 
# From Kaggle notebook "Spaceship Titanic with TFDF"

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

print("TensorFlow v"+ tf.__version__)
print("TensorFlow Decision Forests v" + tfdf.__version__)


dataset = pd.read_csv("./data/spaceship-titanic/train.csv")

print("Full train dataset shape is {}".format(dataset.shape))

# print(dataset.head(5))
# print(dataset.describe())
# print(dataset.info())

plot_df = dataset.Transported.value_counts()
plot_df.plot(kind="bar")

fig, ax = plt.subplots(5, 1, figsize=(10, 10))
plt.subplots_adjust(top = 0.955, hspace=0.4)

sns.histplot(dataset['Age'], color = 'b', bins=50, ax=ax[0])
sns.histplot(dataset['FoodCourt'], color = 'b', bins=50, ax=ax[1], log_scale=(False,True))
sns.histplot(dataset['ShoppingMall'], color = 'b', bins=50, ax=ax[2], log_scale=(False,True))
sns.histplot(dataset['Spa'], color = 'b', bins=50, ax=ax[3], log_scale=(False,True))
sns.histplot(dataset['VRDeck'], color = 'b', bins=50, ax=ax[4], log_scale=(False,True))

# plt.show()

