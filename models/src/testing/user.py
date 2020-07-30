import pandas as pd
import src.testing.utils as utils
import csv
import random
import numpy as np


class Users:
    def __init__(self):
        self.test_positive_path = utils.get_path('test_positive/concatenated.csv')
        self.test_negative_offset_path = utils.get_path('test_negative/offsets.csv')
        self.commentids_for_representations = np.load(utils.get_path('comment_ids_not_to_train.npy'))
        self.selection_header = {el: i for i, el in
                                 enumerate(['author_id', 'article_id', 'max_timestamp', 'comment_ids'])}
        self.train_data_path = utils.get_path('train-set_all.csv')

        self.user_representation = None

        self.test_data_true_df = None
        self.test_data_false_offsets_dict = None
        self.test_data_false_part_dict = None

        self.most_k_recent_comments = 40

        self._build_helpers()

    def _build_helpers(self):
        self._build_user_representation_dict()
        self._load_test_data_true()
        self._load_test_data_false_offsets()

    def _build_user_representation_dict(self):
        df_train = pd.read_csv(self.train_data_path)
        df_train = df_train[df_train['comment_id'].isin(self.commentids_for_representations)].reset_index()
        df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])
        df_train = df_train.sort_values(by='timestamp', ascending=False)
        user_representation = df_train.groupby('author_id')['comment_id'].apply(
            lambda x: list(x)[:self.most_k_recent_comments]).reset_index()
        user_representation.index = user_representation['author_id']
        self.user_representation = user_representation['comment_id'].to_dict()

    def _load_test_data_true(self):
        self.test_data_true_df = pd.read_csv(self.test_positive_path)

    def _load_test_data_false_offsets(self):
        test_data_false_offsets = pd.read_csv(self.test_negative_offset_path)
        test_data_false_offsets.index = test_data_false_offsets['author_id']
        self.test_data_false_offsets_dict = test_data_false_offsets['offset'].to_dict()
        self.test_data_false_part_dict = test_data_false_offsets['part'].to_dict()

    def _get_user_test_false_offsets_part(self, user):
        offsets = self.test_data_false_offsets_dict[user]
        offsets = utils.parse_input_list(offsets)
        part = self.test_data_false_part_dict[user]
        return part, offsets

    def _parse_line(self, line):
        return list(csv.reader([line]))[0]

    def _get_csv_line_by_offset(self, file, offset):
        with open(file, 'r', encoding='utf-8') as f:
            f.seek(offset)
            return self._parse_line(f.readline())

    def get_user_representation(self, user_id):
        return self.user_representation[user_id]

    def get_positive_test_samples(self, user_id):
        selection = self.test_data_true_df[self.test_data_true_df['author_id'] == user_id]['comment_ids']
        test_data_selection = []
        for row in selection:
            test_data_selection.append(utils.parse_input_list(row))
        return test_data_selection

    def get_negative_test_samples(self, user_id, samples, num_negative):
        part, offsets = self._get_user_test_false_offsets_part(user_id)
        path = utils.get_test_negative_comment_ids_path(part)
        len_offsets = len(offsets)
        comments_ids = []
        for n in range(samples):
            for i in range(num_negative):
                if (n * num_negative) + i >= len_offsets:
                    offset = random.choice(offsets)
                else:
                    offset = offsets[(n * num_negative) + i]
                line = self._get_csv_line_by_offset(path, offset)
                comments_ids.append(utils.parse_input_list(line[self.selection_header['comment_ids']]))
        return comments_ids
