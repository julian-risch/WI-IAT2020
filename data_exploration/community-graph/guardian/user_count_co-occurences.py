#%% 
import numpy as np
import pandas as pd
import networkx as nx

#%%
df = pd.read_csv('/mnt/data/datasets/newspapers/guardian/train_test/train_all.csv')

#%%
df.head(1)

#%%
df = df[['article_id', 'author_id', 'comment_id']]

#%%
unique_authors = df['author_id'].unique()

#%%
unique_articles = df['article_id'].unique()

#%%
print('Number of Authors: ', len(unique_authors))
print('Number of Articles: ', len(unique_articles))

#%%
articles_to_index_dict = { article_id : i for i, article_id in enumerate(unique_articles) }

#%%
authors_to_index_dict = { author_id : i for i, author_id in enumerate(unique_authors) }

#%%
occ_matrix = np.zeros((len(unique_authors), len(unique_articles)))

#%%
