# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data = pd.read_csv('../data/data.csv')
movies = pd.read_csv('../data/movies.csv')
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')


# print(data['Rating'].min(), data['Rating'].max())
print(movies)

# %%
# all ratings
fig, ax = plt.subplots()
ax.hist(data['Rating'], bins=list(range(1, 7)))
ax.set_xlabel('Rating')
ax.set_ylabel('Count')
ax.set_xticks(np.arange(1.5, 6.5, 1))
ax.set_xticklabels(list(range(1, 6)))
ax.set_title('All Ratings')
plt.show()


# %%
# most popular 
# TODO
most_popular = data.groupby('Movie ID').count() \
    .sort_values('User ID', ascending=False)
print(most_popular)
data_popular = data[data['Movie ID'].isin(most_popular.index[:10])]
data_popular['Movie ID'] = data_popular['Movie ID'].map(lambda mid: movies[movies['Movie ID'] == mid].iloc[0]['Movie Title'])
most_popular['Movie ID'] = most_popular['Movie ID'].map(lambda mid: movies[movies['Movie ID'] == mid].iloc[0]['Movie Title'])
# ax = sns.histplot(data=data_popular, x='Movie ID', hue='Rating', multiple='dodge')
ax = sns.barplot(data=most_popular, x='Movie ID', y='Rating')
ax.set_title('Most popular')
ax.tick_params(axis='x', rotation=60, labelright=0)


# %%
# highest rated 
highest_rated = data.groupby('Movie ID').mean() \
    .sort_values('Rating', ascending=False).index[:10]
data_best = data[data['Movie ID'].isin(highest_rated)]
data_best['Movie ID'] = data_best['Movie ID'].map(lambda mid: movies[movies['Movie ID'] == mid].iloc[0]['Movie Title'])
ax = sns.histplot(data=data_best, x='Movie ID', hue='Rating', multiple='dodge')
ax.tick_params(axis='x', rotation=60, labelright=0)
ax.set_title('Highest rated')