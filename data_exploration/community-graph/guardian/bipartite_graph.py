#%%
import numpy as np
import pandas as pd
import networkx as nx

#%%
df = pd.read_csv('/mnt/data/datasets/newspapers/guardian/train_test/train_all.csv')

#%%
df.head(1)


#%%
G = nx.Graph()

#%%
unique_users = df['author_id'].unique()
unique_articles = df['article_id'].unique()

unique_articles = ['a' + str(article) for article in unique_articles]

#%%
G.add_nodes_from(unique_users, type="'user'")

#%%
G.add_nodes_from(unique_articles, type="'article'")

#%%
G.number_of_nodes()

#%%
for index, comment in df.iterrows():
    G.add_edge(comment['author_id'], 'a' + str(comment['article_id']))

#%%
print(nx.info(G))



#%%
df.count()

#%%
