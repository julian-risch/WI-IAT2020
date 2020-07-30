import pandas as pd
import users.utils as utils

from constants import COMMENTS_TO_REPRESENT_USER


class Users:
    def __init__(self):
        self.train_data_path = utils.get_path('train-set_all.csv')

        self.user_representation = None
        self.most_k_recent_comments = COMMENTS_TO_REPRESENT_USER

        self._build_helpers()

    def _build_helpers(self):
        self._build_user_representation_dict()

    def _build_user_representation_dict(self):
        df_train = pd.read_csv(self.train_data_path)
        df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])
        df_train = df_train.sort_values(by='timestamp', ascending=False)
        user_representation = df_train.groupby('author_id')['comment_id'].apply(
            lambda x: list(x)[:self.most_k_recent_comments]).reset_index()
        user_representation.index = user_representation['author_id']
        self.user_representation = user_representation['comment_id'].to_dict()

    def get_user_representation(self, user_id):
        return self.user_representation[user_id]
