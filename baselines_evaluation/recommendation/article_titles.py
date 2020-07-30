import pandas as pd

from constants import ROOT_PATH, COMMENTS_TO_REPRESENT_USER
from users.users import Users
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('/mnt/data/vikuen/data/guardian/evaluation/article_categories_crawled.csv')

df = df[['id', 'title']]

df_test = pd.read_csv(ROOT_PATH + 'test-set_all.csv')
df_test_author_ids = df_test['author_id'].unique()
# UNCOMMENT
# df_train = pd.read_csv('/mnt/data/vikuen/data/guardian/train-set_all.csv')
# if for smaller vector
df_train = pd.read_csv(ROOT_PATH + 'train-set_all.csv')
df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])
df_train = df_train.sort_values(by='timestamp', ascending=False)
author_ids = df_train[df_train['author_id'].isin(df_test_author_ids)]['author_id'].unique()

train_article_ids = df_train[df_train['author_id'].isin(df_test_author_ids)]['article_id'].unique()
test_article_ids = df_test['article_id'].unique()

article_titles_train = df[df['id'].isin(train_article_ids)]['title'].tolist()
article_titles_train_ids = df[df['id'].isin(train_article_ids)]['id'].tolist()

article_titles_test = df[df['id'].isin(test_article_ids)]['title'].tolist()
article_titles_test_ids = df[df['id'].isin(test_article_ids)]['id'].tolist()

# settings that you use for count vectorizer will go here
tfidf_vectorizer = TfidfVectorizer(use_idf=True)

# just send in all your docs here
tfidf_vectorizer = tfidf_vectorizer.fit(article_titles_train)
tfidf_train_vectors = tfidf_vectorizer.transform(article_titles_train)
tfidf_test_vectors = tfidf_vectorizer.transform(article_titles_test)

df_test_comment_to_article = df_test[['article_id', 'comment_id']]
df_test_comment_to_article.index = df_test_comment_to_article.comment_id
test_comment_to_article_dict = df_test_comment_to_article['article_id'].to_dict()

df_representation = df_train[df_train['author_id'].isin(df_test_author_ids)]
df_train_user_articles = df_representation.groupby('author_id')['article_id'].apply(
    lambda x: np.unique(x[:COMMENTS_TO_REPRESENT_USER])).reset_index()

users = Users()

author_ids = np.load(f'{ROOT_PATH}usable_authors_test.npy')

train_article_pos_dict = {k: v for v, k in enumerate(article_titles_train_ids)}
test_article_pos_dict = {k: v for v, k in enumerate(article_titles_test_ids)}

length = tfidf_test_vectors[0].todense().shape[1]


def get_user_representation(user_id):
    user_train_articles = df_train_user_articles[df_train_user_articles['author_id'] == user_id]['article_id'].iloc[0]
    user_train_mask = []
    for article in user_train_articles:
        pos = train_article_pos_dict.get(article)
        if pos is not None:
            user_train_mask.append(pos)

    mx = tfidf_train_vectors[user_train_mask]
    return mx.mean(axis=0)


def get_comment_section_representation(comment_section_ids):
    if len(comment_section_ids) > 0:
        article_id = test_comment_to_article_dict[comment_section_ids[0]]
        if test_article_pos_dict.get(article_id) is not None:
            return tfidf_test_vectors[test_article_pos_dict.get(article_id)].todense()
        else:
            return np.zeros((1, length))
    else:
        return np.zeros((1, length))
