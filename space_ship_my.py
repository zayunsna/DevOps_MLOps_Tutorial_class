import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
color_palette = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray']

dataset = pd.read_csv("./data/spaceship-titanic/train.csv")

print("Full train dataset shape is {}".format(dataset.shape))

# print(dataset.head(5))
print(dataset.describe())
print(dataset.info())

# EDA : Numerical data distribution check
numerical = ['Age', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'RoomService']
fig, ax = plt.subplots(3, 2, figsize=(10, 10))
plt.subplots_adjust(top = 0.955, hspace=0.4)
ax = ax.flatten()
for i in range(len(numerical)):
    # if i == 0:
    #     sns.histplot(dataset[numerical[i]], color='b', bins=50, ax=ax[i])
    # else:
    #     sns.histplot(dataset[numerical[i]], color='b', bins=50, ax=ax[i], log_scale=(False, True))
    sns.histplot(dataset[numerical[i]], color='b', bins=50, ax=ax[i])
    ax[i].set_yscale('log')
    ax[i].grid()
# plt.show()

# Seperate the Cabin info into Deck, Cabin number, Side.
dataset[["Deck", "Cabin_num", "Side"]] = dataset["Cabin"].str.split("/", expand=True)
# And then Remove Cabin column
dataset = dataset.drop('Cabin', axis=1)

print(dataset.head(5))

# EDA : Categorical data distribution check
categorical = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Cabin_num', 'Side']
fig, ax = plt.subplots(4, 2, figsize=(10, 10))
plt.subplots_adjust(top = 0.955, bottom = 0.045, hspace=0.27)
ax = ax.flatten()
for i in range(len(categorical)):
    ax[i].grid()
    sns.countplot(x = dataset[categorical[i]], hue=dataset[categorical[i]], ax=ax[i], palette=[color_palette[i]], legend=False)

if len(categorical) % 2 != 0:
    fig.delaxes(ax[-1])
plt.show()
