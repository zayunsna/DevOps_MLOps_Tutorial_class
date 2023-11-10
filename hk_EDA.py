#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import array


################################################################################################################################
#### This part is making the dataframe sample for the function test.
import random
import string
# Define the number of rows and columns
n_rows = 1000
n_cols = 8

# Create random data for each column
data = {
    'Object_String': ["".join(random.choices(string.ascii_letters, k=8)) for _ in range(n_rows)],
    'Object_Bool': [random.choice([True, False]) for _ in range(n_rows)],
    'Float_1': np.random.logistic(1, 1, n_rows),
    'Float_2': np.random.normal(5, 1, n_rows),
    'Int': np.random.randint(1, 100, size=(n_rows)),
    'Float_3': np.random.gamma(2., 2., n_rows),
    'Date': pd.date_range(start='2023-01-01', periods=n_rows, freq='D'),
    'Mixed': [random.choice([random.randint(1, 50), random.uniform(1.0, 50.0), 
                             "".join(random.choices(string.ascii_letters, k=5))]) for _ in range(n_rows)]
}

# Convert to DataFrame
df = pd.DataFrame(data)
df['Int'] = df['Int'].astype('Int64')

# Introduce approx 5~10% missing values
for col in df.columns:
    frac = random.randrange(5,10) * 0.01
    df.loc[df.sample(frac=frac).index, col] = np.nan
#### sample dataframe part end.
################################################################################################################################
file_path = '../DevOps_MLOps_Tutorial_class/data/spaceship-titanic/train.csv'
df2 = pd.read_csv(file_path)

df2[["Deck", "Cabin_num", "Side"]] = df2["Cabin"].str.split("/", expand=True)
df2 = df2.drop('Cabin', axis=1)
df2 = df2.drop('PassengerId', axis=1)
df2 = df2.drop('Name', axis=1)
#### sample dataframe-2 part end.
################################################################################################################################

def create_grid(n):
    root = n ** 0.5
    rows = int(root) if root.is_integer() else int(root) + 1
    cols = rows if rows * (rows - 1) < n else rows - 1
    return rows, cols

def getInfo(dataset : pd.DataFrame):
    print("\n### Data Summary")
    print(dataset.describe())
    print("\n### Summarized data")
    print(dataset.info())
    print("\n### Data Head info")
    print(dataset.head())

def columnsInfo(dataset: pd.DataFrame, type:array = []):
    column = dataset.select_dtypes(include=type)
    column_name = column.columns
    column_count = len(column_name)
    return column_name, column_count

def drawPlot(type:string, grid_x:int, grid_y:int, dataset:pd.DataFrame, entry:int, item:array, nbins : int = 50, ylog : bool = False):
    fig, ax = plt.subplots(grid_x, grid_y, figsize = (10, 10))
    ax = ax.flatten()
    for i in range(entry):
        ax[i].grid()
        if ylog: ax[i].set_yscale('log')
        if type == 'num': sns.histplot(dataset[item[i]], color='b', bins=nbins, ax=ax[i])
        else: sns.countplot(x=dataset[item[i]], ax=ax[i])
    plt.show()

def numericalPlot(dataset : pd.DataFrame, nbins : int = 50, ylog : bool = False):
    numerical_column_name, column_count = columnsInfo(dataset, ['int', 'float'])
    grid_x, grid_y = create_grid(column_count)
    drawPlot("num", grid_x, grid_y, dataset, column_count, numerical_column_name, nbins, ylog)

def categoricalPlot(dataset : pd.DataFrame):
    categorical_column_name, column_count = columnsInfo(dataset, ['object'])
    grid_x, grid_y = create_grid(column_count)
    drawPlot("category", grid_x, grid_y, dataset, column_count, categorical_column_name)
    
def setMissingValue(dataset: pd.DataFrame, method:string = "random"):
    ## TODO : Need to build how control the different cases
    return 0

def setOneHotEncoding(dataset: pd.DataFrame, item:array) -> pd.DataFrame:
    for name in item:
        encoded_column = pd.get_dummies(dataset[name], prefix=name, drop_first=False)
        dataset = dataset.drop(name, axis=1)
        dataset = pd.concat([dataset, encoded_column], axis = 1)
    return dataset

def scaler(dataset:pd.DataFrame):
    sc = MinMaxScaler(feature_range=(0,1))
    sc = sc.fit(dataset)
    scaled_data = sc.transform(dataset)
    return pd.DataFrame(scaled_data, columns=dataset.columns)

def calPCC(feature_a:np.ndarray, feature_b:np.ndarray, _range : list):
    fitter, C_p = np.polyfit(feature_a, feature_b, 1, cov=True)
    
    xModel = np.linspace(_range[0][0], _range[0][1], 500)
    yModel = np.polyval(fitter, xModel)

    TT = np.vstack([xModel**(1-i) for i in range(2)]).T
    yi = np.dot(TT, fitter)
    C_yi = np.dot(TT, np.dot(C_p, TT.T))
    sig_yi = np.sqrt(np.diag(C_yi))

    return xModel, yModel, yi, sig_yi

def getFeatureCorrelation(dataset: pd.DataFrame, feature_a:str, feature_b:str, _range:list):
    xData = dataset[feature_a]
    yData = dataset[feature_b]
    xModel, yModel, yi, sig_yi = calPCC(xData, yData, _range)

    fig = plt.figure(figsize=(8,6), dpi=100)
    fig = fig.add_subplot(111)

    plt.hist2d(x = xData, y = yData, bins=50, norm=mpl.colors.LogNorm(), range = _range)
    fig.plot(xModel, yModel)
    fig.fill_between(xModel, yi+sig_yi, yi-sig_yi, alpha=.25)
    plt.colorbar()
    plt.grid()
    plt.xlabel(feature_a)
    plt.ylabel(feature_b)
    plt.show()

def getFeatureHeatmap(dataset: pd.DataFrame):
    target_column_name, _ = columnsInfo(dataset, ['int', 'float'])
    numerical_dataset = dataset[target_column_name]
    scaled_numerical_dataset= scaler(numerical_dataset).corr()
    plt.figure(figsize=(14, 10))
    plot = sns.heatmap(scaled_numerical_dataset, cmap='Blues', vmin=-1, vmax=1, annot=True)
    plt.xticks(rotation=45, ha='right')
    plt.show()

def removeColumns(dataset: pd.DataFrame, colume_name:array = []) -> pd.DataFrame:
    for i in range(len(colume_name)):
        dataset = dataset.drop(colume_name[i], axis=1)
    return dataset

# df = numpy.random # Test data
# df2 = kaggle example data

# getInfo(df2)
# numericalPlot(df2, 20)
# numericalPlot(df2, 20, True)
# categoricalPlot(df2)
# print(df2.head())
# print("#"*50)
# df3 = setOneHotEncoding(df2, ['HomePlanet', 'Destination', 'Deck', 'Side'])
# print(df3[:10])
# print(df3.loc[:10, 'Transported':'Side_S'])

# _range = [[0,15000],[0,30000]]
# df2[['RoomService', 'FoodCourt']] = df2[['RoomService', 'FoodCourt']].fillna(value=0)
# getFeatureCorrelation(df2, 'RoomService', 'FoodCourt', _range)
# getFeatureHeatmap(df2)