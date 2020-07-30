import csv

import pandas as pd

from src.constants import ROOT_PATH
from src.testing import utils
import numpy as np


class EvaluationDataset:
    def __init__(self, comments_to_train_path=None):
        self.train_data_path = utils.get_path('train-set_all.csv')

        self.user_representation = None

        self.comments_to_train_path = comments_to_train_path
        evaluation_dataset_path = ROOT_PATH + 'evaluation_dataset.csv'

        self.test_df = pd.read_csv(evaluation_dataset_path)

        self.most_k_recent_comments = 20
        self.num_negative = 50
        self._build_helpers()
        self.current_line = 0

    def __iter__(self):
        return self

    def __len__(self):
        return int(len(self.test_df) / 51)

    def __next__(self):
        if self.current_line == len(self.test_df):
            raise StopIteration
        end_position = self.current_line + self.num_negative + 1
        current_selection = self.test_df[self.current_line:end_position]
        sections = current_selection['comment_ids'].tolist()

        sections = [utils.parse_input_list(section) for section in sections]

        article_ids = current_selection['article_id'].tolist()
        y = current_selection['y'].tolist()
        # y = [1] + [0] * 50
        author_id = current_selection.iloc[0]['author_id']
        user_rep = self.get_user_representation(author_id)

        self.current_line = end_position
        return author_id, user_rep, article_ids, sections, y

    def _build_helpers(self):
        self._build_user_representation_dict()

    def _build_user_representation_dict(self):
        df_train = pd.read_csv(self.train_data_path)
        if self.comments_to_train_path is not None:
            comments_to_train = np.load(self.comments_to_train_path)
            df_train = df_train[df_train['comment_id'].isin(comments_to_train)]
            print('Comments to train loaded, comments left:', len(df_train))
        df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])
        df_train = df_train.sort_values(by='timestamp', ascending=False)
        user_representation = df_train.groupby('author_id')['comment_id'].apply(
            lambda x: list(x)[:self.most_k_recent_comments]).reset_index()
        user_representation.index = user_representation['author_id']
        self.user_representation = user_representation['comment_id'].to_dict()

    def get_user_representation(self, user_id):
        return self.user_representation[user_id]


