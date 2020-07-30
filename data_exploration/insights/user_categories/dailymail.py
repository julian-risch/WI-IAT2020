# %%
import pandas as pd

# %%
path = ''
article_path = ''
output_path = ''
df = pd.read_csv(path)

# %%
df.head(1)

#%%
df['article_id'] = df['article_id'].fillna(0).astype(int)

# %%
df_articles = pd.read_csv(article_path)

# %%
df_articles.head()

# %%
df_articles['category'] = df_articles['article_url'].apply(
    lambda x: None if len(x.split('/')) == 3 else x.split('/')[3] )

# %%
df_articles.head(10)

#%%
df['author_id'].nunique()

#%%
df.dtypes

#%%
df_articles.dtypes

#%%
df['article_id'] = df['article_id'].astype(object)

# %%
df = df.merge(df_articles, on='article_id', how='left')

#%%
df['author_id'].nunique()
# %%
df.head(20)

# %%
df['category'].nunique()

# %%
user_category_count = df.groupby(['author_id', 'category'])[
    'comment_id'].count()

# %%
user_category_count.head(20)

#%%
out = pd.crosstab(df.author_id, df.category)

#%%
out.reset_index()

#%%
out.to_csv(output_path, index=False)

#%%
out_user_max = user_category_count.reset_index().groupby('author_id').apply(lambda x: x.sort_values(by='comment_id', ascending=False).head(1))

#%%
out_user_max = out_user_max.rename({'author_id': 'author'}, axis='columns').reset_index()[['author_id', 'category', 'comment_id']]

#%%
out_user_max.to_csv(output_path, index=False)

#%%
out_user_max.max()

#%%
df_articles.count()

#%%
df['article_id'].nunique()

#%%


#%%
