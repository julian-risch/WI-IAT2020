#%%
import numpy as np
import pandas as pd
import seaborn as sns

#%%
df = pd.read_csv('/mnt/data/datasets/newspapers/daily-mail/train_test/train_all.csv')

#%%
df.head(1)

#%%
df['timestamp'] = pd.to_datetime(df['timestamp'])

#%%
first_ten_comments = df.groupby('author_id').apply(lambda x: x.sort_values(by='timestamp', ascending=False).head(7))

#%%
authors = first_ten_comments.rename({'author_id': 'author'}, axis='columns').reset_index()

#%%
authors.head(20)
#%%
authors.to_csv('~/jp-data-analysis/data/selection/daily-mail_last_7_comments_of_author.csv', index=False)

#%%
