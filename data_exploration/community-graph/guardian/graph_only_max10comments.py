# %%
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
root_path = '/mnt/data/vikuen/data/guardian'

# %%
df = pd.read_csv(
    root_path + '/train-set_all.csv')
df_match = pd.read_csv(
    root_path + '/graph/last_20_comments_of_author.csv')

df_user_max = pd.read_csv('~/jp-data-analysis/data/user_categories/user_most_commented_category.csv')

comments = df_match['comment_id'].unique()

# %%
df_in = df

# %%
print('Number of Comments', df_in['comment_id'].nunique())

# %%
df = df[df['comment_id'].isin(comments)]

# %%
print('Number of Comments', df['comment_id'].nunique())

# %%
print('Unique authors: ', df['author_id'].nunique())

# %%
df = df[['article_id', 'author_id', 'comment_id']]


# %%
df.head(1)

# %%
article_author_list_df = df.groupby('article_id')['author_id'].apply(lambda x: list(set(x)))


# %%
article_author_list_df = article_author_list_df.reset_index()


# %%
article_author_list_df.iloc[0]['author_id'][0]

# %%
G = nx.Graph()


# %%
# add nodes
G.add_nodes_from(df['author_id'].unique())

#%%
df_user_max.index = df_user_max['author_id']

#%%
nx.set_node_attributes(G, df_user_max[['category']].to_dict()['category'], 'category')

# %%
for index, row in article_author_list_df.iterrows():
    source_author = row['author_id'][0]
    for author in row['author_id'][1:]:
        G.add_edge(source_author, author)

# %%
print(nx.info(G))


# %%
nx.write_graphml(
    G, root_path + "/graph/graph_last20comments.graphml")

# %%
nx.write_edgelist(G, root_path + '/graph/graph_last20comments.edgelist',data=False)

# %%
to_del = [n for n in G if G.degree(n) < 1]
G.remove_nodes_from(to_del)

# %%
# to_del = [n for n in G if G.degree(n) > 100]
# G.remove_nodes_from(to_del)

# %%
print(nx.info(G))

# %%
id_to_degree_dict = {n: G.degree(n) for n in G}


# %%
degree_df = pd.DataFrame.from_dict(id_to_degree_dict, orient='index')


# %%
degree_df = degree_df.reset_index().rename(
    {'index': 'author_id', 0: 'degree'}, axis='columns')

# %%
degree_df.head(1)

# %%
# # df.merge(degree_df, on='author_id')
user_article_comment_count_df = df.groupby(
    'author_id')[['comment_id', 'article_id']].nunique().reset_index()

# %%
merged = user_article_comment_count_df.merge(degree_df, on='author_id')

# %%
merged = merged.rename({'comment_id': 'comment_count',
                        'article_id': 'article_count'}, axis='columns')

# %%
merged['degree'].max()


# %%
merged[merged['degree'] == 14776]

# #%%
# merged.describe()

# #%%
# merged[merged['degree'] > 11].describe()

# #%%
# sns.boxplot(merged['degree'])

# #%%
# article_count_df = df.groupby('article_id')['author_id'].nunique().reset_index()

# #%%
# article_count_df.describe()

# #%%

# %%
