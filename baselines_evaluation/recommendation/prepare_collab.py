import pandas as pd
import pickle
from tqdm import tqdm

from constants import ROOT_PATH


def prepare_collab():
    df_test = pd.read_csv(ROOT_PATH + 'test-set_all.csv')
    df_test_author_ids = df_test['author_id'].unique()
    # UNCOMMENT
    df_train = pd.read_csv(ROOT_PATH + 'train-set_all.csv')
    author_ids = df_train[df_train['author_id'].isin(df_test_author_ids)]['author_id'].unique()
    article_counts = df_train.groupby(['author_id', 'article_id'])['comment_id'].nunique()
    article_counts = article_counts.reset_index()
    article_ids = df_train['article_id'].unique()
    article_to_position = {article_id: i for i, article_id in enumerate(article_ids)}

    # %%
    number_of_articles = df_train['article_id'].nunique()

    # %%
    def get_user_representation(user_id):
        selection = article_counts[article_counts['author_id'] == user_id]
        user_dict = {}
        for index, row in selection.iterrows():
            article_id = row['article_id']
            count = row['comment_id']
            user_dict[article_id] = count
        return user_dict

    # %%
    user_vectors = {}
    for user in tqdm(author_ids):
        user_vectors[user] = get_user_representation(user)
    return user_vectors, article_to_position, len(article_ids)
