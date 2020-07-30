#%%
import numpy as np
import pandas as pd
import seaborn as sns

#%%
root_path = '/mnt/data/vikuen/data/guardian'
#%%
df = pd.read_csv(root_path + '/train-set_all.csv')

#%%
df.head(1)

#%%
df['timestamp'] = pd.to_datetime(df['timestamp'])

#%%
first_ten_comments = df.groupby('author_id').apply(lambda x: x.sort_values(by='timestamp', ascending=False).head(20))

#%%
authors = first_ten_comments.rename({'author_id': 'author'}, axis='columns').reset_index()

#%%
authors.head(20)
#%%
authors.to_csv(root_path + '/graph/last_20_comments_of_author.csv', index=False)

#%%
