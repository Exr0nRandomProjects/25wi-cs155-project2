# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import skew

data = pd.read_csv('../data/data.csv')
movies = pd.read_csv('../data/movies.csv')
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')


# print(data['Rating'].min(), data['Rating'].max())
print(movies.columns)

plt.rcParams['savefig.dpi'] = 200
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

# %%
# all ratings
def plot_all_ratings():
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.histplot(data['Rating'], bins=list(range(1, 7)), ax=ax)
    ax.set_xlabel('Rating')
    ax.set_ylabel('Count')
    ax.set_xticks(np.arange(1.5, 6.5, 1))
    ax.set_xticklabels(list(range(1, 6)))
    ax.set_title('All Ratings')
    plt.show()
plot_all_ratings()


# %%
# most popular 
def plot_most_popular(data):
    most_popular = data.groupby('Movie ID').count() \
        .sort_values('User ID', ascending=False).index[:10]
    data_popular = data[data['Movie ID'].isin(most_popular)]
    data_popular['Movie Name'] = data_popular['Movie ID'].map(lambda mid: movies[movies['Movie ID'] == mid].iloc[0]['Movie Title'])
    ax = sns.histplot(data=data_popular, x='Movie Name', hue='Rating', multiple='dodge')
    ax.set_title('Most popular')
    # ax.tick_params(axis='x', rotation=60, labelright=0, ha='right')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right', rotation_mode='anchor')
    return most_popular, data_popular
most_popular, data_popular = plot_most_popular(data)
plt.show()


# %% most popular, show num votes


# %%
# highest rated 
def plot_highest_rated(data):
    highest_rated = data.groupby('Movie ID').mean() \
        .sort_values('Rating', ascending=False).index[:10]
    data_best = data[data['Movie ID'].isin(highest_rated)]
    data_best['Movie Name'] = data_best['Movie ID'].map(lambda mid: movies[movies['Movie ID'] == mid].iloc[0]['Movie Title'])
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.histplot(data=data_best, x='Movie Name', hue='Rating', multiple='dodge', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right', rotation_mode='anchor')
    ax.set_title('Highest rated')
plot_highest_rated(data)
plt.show()

# %% 
# visualization of genres 

def plot_genres(data, genres):
    fig, axs = plt.subplots(1, len(genres), figsize=(9, 3), sharey=True, sharex=True)
    fig.suptitle('Rating distribution by Genre')
    for ax, genre in zip(axs, genres):
        movie_ids = movies[movies[genre] > 0]['Movie ID']
        movies_g = data[data['Movie ID'].isin(movie_ids)]
        # movies_g = movies_g.groupby('Movie ID').mean()
        # movies_g['Movie ID'] = movies_g['Movie ID'].map(lambda mid: movies[movies['Movie ID'] == mid].iloc[0]['Movie Title'])
        sns.histplot(ax=ax, data=movies_g, x='Rating', bins=np.arange(0.5, 5.6, 1))
        ax.set_xticks(np.arange(1, 5.5, 1))
        ax.set_xticklabels(list(range(1, 6)), weight='bold')
        ax.set_title(genre, weight='bold')
    plt.tight_layout()
    plt.show()
plot_genres(data, ['Adventure', 'Crime', 'Sci-Fi'])

# %% 
# skew vis 

def plot_skew(data, most_popular):
    global_skew = skew(data['Rating'])
    print("skew of all ratings:", global_skew)
    fig, ax = plt.subplots(figsize=(4, 3))
    skews = [skew(data[data['Movie ID'] == m]['Rating']) for m in most_popular]
    print(skews)
    movie_name_by_id = { k: v for k, v in zip(movies['Movie ID'], movies['Movie Title']) }
    movie_names = [movie_name_by_id[i] for i in most_popular]
    print(movie_names, skews)
    sns.barplot(x=movie_names, y=skews, ax=ax)
    ax.hlines([global_skew], label='Global Skew', xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], color='black', lw=2, ls='--')
    ax.legend()
    ax.set_title("Skew of most popular movies")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')
    ax.set_xlabel('Movie Title')
    ax.set_ylabel('Skew')
    plt.show()
plot_skew(data, most_popular)