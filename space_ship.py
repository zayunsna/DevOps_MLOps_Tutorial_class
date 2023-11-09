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

# Not-use info dropping ['PassengerId', 'Name']
df = dataset.drop(['PassengerId', 'Name'], axis = 1)

# handing the missing value
df.isnull().sum().sort_values(ascending=False)
df[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = df[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(value=0)
df.isnull().sum().sort_values(ascending=False)

# TF-DF currently not support the boolean type, but others can use it with TD-DF.
# do the convertion boolean data into integer.
labels = ["Transported", "VIP", 'CryoSleep']
for i in range(len(labels)):
    df[labels[i]] = df[labels[i]].astype(int)
# print(df.head(5))

# Seperate the Cabin info into Deck, Cabin number, Side.
df[["Deck", "Cabin_num", "Side"]] = df["Cabin"].str.split("/", expand=True)

# And then Remove Cabin column
try:
    df = df.drop('Cabin', axis=1)
except KeyError:
    print("Field dose not exist")
# print(df.head(5))

# Split the dataset into training and testing dataset.
def split_dataset(dataset, test_ratio=0.2):
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]

train_df, valid_df = split_dataset(df)
print("{} examples in training, {} examples in testing".format(len(train_df), len(valid_df)))

# Convert the dataset into TensorFlow dataset format from pandas format. pd.DataFrame -> tf.data.Dataset
label = "Transported"
train_df = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label=label)
valid_df = tfdf.keras.pd_dataframe_to_tf_dataset(valid_df, label=label)

# Now we need to select a Model.
# 1. RandomForestModel
# 2. GradientBoostedTreesModel
# 3. CartModel
# 4. DistributedGradientBoostedTreesModel
# can check the all avaliable model in TensforFlow Decision Forests.
# print(tfdf.keras.get_all_models())

# Let's choose the model and make Configuration of the model
# Ex) rf = tfdf.keras.RandomForestModel(hyperparameter_template="benchmark_rank1")
rf = tfdf.keras.RandomForestModel()
rf.compile(metrics=["accuracy"]) # Optional, you can use this to include a list of eval metrics.

# Do train a Model using training dataset
rf.fit(x=train_df)

# Vsualize the model
# tfdf.model_plotter.plot_model_in_colab(rf, tree_idx=0, max_depth=3)
inspector = rf.make_inspector()
logs = inspector.training_logs()
inspector.evaluation()
evaluation = rf.evaluate(x=valid_df, return_dict=True)
for name, value in evaluation.items():
    print(f"{name}: {value:.4f}")
plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Accuracy (out-of-bag)")
plt.show()

# Variable Importances
print(f"Available variable importances:")
for importance in inspector.variable_importances().keys():
    print("\t", importance)

print(inspector.variable_importances()["NUM_AS_ROOT"])


# Load the test dataset
test_df = pd.read_csv('./data/spaceship-titanic/test.csv')
submission_id = test_df.PassengerId

# Replace NaN values with zero
test_df[['VIP', 'CryoSleep']] = test_df[['VIP', 'CryoSleep']].fillna(value=0)

# Creating New Features - Deck, Cabin_num and Side from the column Cabin and remove Cabin
test_df[["Deck", "Cabin_num", "Side"]] = test_df["Cabin"].str.split("/", expand=True)
test_df = test_df.drop('Cabin', axis=1)

# Convert boolean to 1's and 0's
test_df['VIP'] = test_df['VIP'].astype(int)
test_df['CryoSleep'] = test_df['CryoSleep'].astype(int)

# Convert pd dataframe to tf dataset
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df)

# Get the predictions for testdata
predictions = rf.predict(test_ds)
n_predictions = (predictions > 0.5).astype(bool)
output = pd.DataFrame({'PassengerId': submission_id,
                       'Transported': n_predictions.squeeze()})

output.head()

sample_submission_df = pd.read_csv('./data/spaceship-titanic/sample_submission.csv')
sample_submission_df['Transported'] = n_predictions
sample_submission_df.to_csv('./data/spaceship-titanic/submission.csv', index=False)
sample_submission_df.head()