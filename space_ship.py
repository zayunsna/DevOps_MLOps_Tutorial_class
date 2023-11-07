#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

print(dataset.head(5))

print(dataset.describe())
print(dataset.info())