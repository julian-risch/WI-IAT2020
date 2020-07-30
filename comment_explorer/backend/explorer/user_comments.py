import pandas as pd
import logging
import numpy as np


class Users:
    def __init__(self, selection_path):
        self.selection_path = selection_path
        self.user_comments_dict = None
        self.df = None
        self._build_index()

    def _build_index(self):
        self.df = pd.read_csv(self.selection_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values(by='timestamp', ascending=True)
        self.user_comments_dict = self.df.groupby('author_id')['comment_id'].apply(list).to_dict()
        logging.info('User comment index built')

    def get_user_comments(self, user_id):
        return self.user_comments_dict[int(user_id)]

    def get_all_users(self):
        return list(self.user_comments_dict.keys())

    def get_comment_info(self, comment_id):
        comment_meta = self.df[self.df['comment_id'] == int(comment_id)].iloc[0]
        upvotes = comment_meta['upvotes']
        if np.isnan(upvotes):
            upvotes = None
        else:
            upvotes = int(upvotes)
        parent_comment_id = comment_meta['parent_comment_id']
        if np.isnan(parent_comment_id):
            parent_comment_id = None
        else:
            parent_comment_id = int(parent_comment_id)
        article_id = comment_meta['article_id']
        if np.isnan(article_id):
            article_id = None
        else:
            article_id = int(article_id)

        author_id = comment_meta['author_id']
        if np.isnan(author_id):
            author_id = None
        else:
            author_id = int(author_id)

        comment_id = int(comment_meta['comment_id'])

        return parent_comment_id, str(comment_meta['timestamp']), upvotes, article_id, author_id, comment_id

    def get_article_comments(self, article_id):
        return self.df[self.df['article_id'] == int(article_id)]['comment_id'].tolist()
