#%%
import numpy as np
import pandas as pd
import networkx as nx

#%%
df = pd.read_csv(
    '/mnt/data/datasets/newspapers/guardian/train_test/train_all.csv')
df_match = pd.read_csv(
    '~/jp-data-analysis/data/selection/last_10_comments_of_author.csv')

df_user_max = pd.read_csv('~/jp-data-analysis/data/user_categories/user_most_commented_category.csv')

#%%
comments = df_match['comment_id']

#%%
df.head(1)

#%%
df = df[df['comment_id'].isin(comments)]

#%%
G = nx.Graph()

#%%
unique_users = df['author_id'].unique()
unique_articles = df['article_id'].unique()

unique_articles = ['a' + str(article) for article in unique_articles]



#%%
G.add_nodes_from(unique_users, type="'user'")
nx.set_node_attributes(G, df_user_max[['category']].to_dict()['category'], 'category')

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
nx.write_graphml(
    G, "jp-data-analysis/data/graph_guardian_bipartite_max_10.graphml")

#%%
