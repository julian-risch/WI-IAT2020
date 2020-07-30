import os
import ast
import pandas as pd
import random
import csv
import numpy as np
from tqdm import tqdm
import pickle
import fire

from train_data_line_offsets import LineByteOffset

# set root path here
root_path = ''


def get_path(path):
    return os.path.join(root_path, path)


def parse_input_list(list_string):
    return ast.literal_eval(list_string)


def get_train_positive_comment_ids_path(part):
    return get_path(f'train_positive/partition_{part}.csv')


def get_train_negative_comment_ids_path(part):
    return get_path(f'train_negative/partition-{part}.csv')


class Users:
    def __init__(self):
        self.selection_header = {el: i for i, el in
                                 enumerate(['author_id', 'article_id', 'max_timestamp', 'comment_ids'])}
        self.train_data_path = get_path('train_set_all.csv')

        self.train_comment_ids_not_to_use = None

        self.train_data_positive_offset_path = get_path('train_positive/offsets.csv')
        self.train_data_negative_offset_path = get_path('train_negative/offsets.csv')

        self.user_representation = None

        self.train_data_positive_offsets_dict = None
        self.train_data_positive_part_dict = None

        self.train_data_negative_offsets_dict = None
        self.train_data_negative_part_dict = None

        self._build_helpers()

    def _build_helpers(self):
        self._load_train_data_positive_offsets()
        self._load_train_data_negative_offsets()

    def _build_user_representation_dict(self):
        df_train = pd.read_csv(self.train_data_path)
        user_representation = df_train.groupby('author_id')['comment_id'].apply(list).reset_index()
        user_representation.index = user_representation['author_id']
        self.user_representation = user_representation['comment_id'].to_dict()

    def _load_train_data_positive_offsets(self):
        train_data_positive_offsets = pd.read_csv(self.train_data_positive_offset_path)
        train_data_positive_offsets.index = train_data_positive_offsets['author_id']
        self.train_data_positive_offsets_dict = train_data_positive_offsets['offset'].to_dict()
        self.train_data_positive_part_dict = train_data_positive_offsets['part'].to_dict()

    def _load_train_data_negative_offsets(self):
        train_data_negative_offsets = pd.read_csv(self.train_data_negative_offset_path)
        train_data_negative_offsets.index = train_data_negative_offsets['author_id']
        self.train_data_negative_offsets_dict = train_data_negative_offsets['offset'].to_dict()
        self.train_data_negative_part_dict = train_data_negative_offsets['part'].to_dict()

    def _get_user_test_false_offsets_part(self, user):
        offsets = self.test_data_false_offsets_dict[user]
        offsets = parse_input_list(offsets)
        part = self.test_data_false_part_dict[user]
        return part, offsets

    def _get_user_train_positive_offsets_part(self, user):
        offsets = self.train_data_positive_offsets_dict[user]
        offsets = parse_input_list(offsets)
        part = self.train_data_positive_part_dict[user]
        return part, offsets

    def _get_user_train_negative_offsets_part(self, user):
        offsets = self.train_data_negative_offsets_dict[user]
        offsets = parse_input_list(offsets)
        part = self.train_data_negative_part_dict[user]
        return part, offsets

    def _parse_line(self, line):
        return list(csv.reader([line]))[0]

    def _get_csv_line_by_offset(self, file, offset):
        with open(file, 'r', encoding='utf-8') as f:
            f.seek(offset)
            return self._parse_line(f.readline())

    def get_positive_training_samples(self, user_id):
        part, offsets = self._get_user_train_positive_offsets_part(user_id)
        path = get_train_positive_comment_ids_path(part)
        comments_ids = []
        for offset in offsets:
            line = self._get_csv_line_by_offset(path, offset)
            ids = parse_input_list(line[self.selection_header['comment_ids']])
            if len(ids) > 0:
                comments_ids.append(ids)
        return comments_ids

    def get_negative_training_samples(self, user_id):
        # at the moment 5 negative examples on articles at that day
        # mind to filter out not_to_train_comment_ids...
        part, offsets = self._get_user_train_negative_offsets_part(user_id)
        path = get_train_negative_comment_ids_path(part)
        comments_ids = []
        for offset in offsets:
            line = self._get_csv_line_by_offset(path, offset)
            ids = parse_input_list(line[self.selection_header['comment_ids']])
            if len(ids) > 0:
                comments_ids.append(ids)
        return comments_ids


def prepare( output_path, usable_authors, offset_output_path, offset_dict_out_path, pairwise=False):
    users = Users()
    train_authors = np.load(usable_authors)

    print(len(train_authors))
    train_authors = [author for author in train_authors if
                     author in users.train_data_negative_offsets_dict and author in users.train_data_positive_offsets_dict]
    if pairwise:
        with open(output_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['author_id', 'comment_ids_pos', 'comment_ids_neg'])
    else:
        with open(output_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['author_id', 'comment_ids', 'commented'])

    print('test')
    for author in tqdm(train_authors):
        negative = users.get_negative_training_samples(author)
        positive = users.get_positive_training_samples(author)

        if len(positive) < len(negative):
            negative = negative[:len(positive)]
        if len(positive) > len(negative):
            positive = positive[:len(negative)]

        if pairwise:
            for index in range(len(positive)):
                line = [author, positive[index], negative[index]]
                with open(output_path, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(line)

        else:
            for sample in negative:
                line = [author, sample, 0]
                with open(output_path, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(line)
            for sample in positive:
                line = [author, sample, 1]
                with open(output_path, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(line)

    collector = LineByteOffset(
        get_path(output_path),
        get_path(offset_output_path)
    )

    collector.start()

    df = pd.read_csv(offset_output_path)
    df.index = df.line
    test = df.offset.to_dict()

    pickle.dump(test, open(offset_dict_out_path, "wb"))


if __name__ == '__main__':
    fire.Fire(prepare)
