import pickle
import csv
import pandas as pd
import logging
import ast

from src.constants import MAX_COMMENTS_USER_SECTION, CORPUS_COMMENT_ID_TO_INDEX, TOKENIZED_COMMENTS_PATH


class DataFetcher:
    def __init__(self, positive_negative_path, train_path, dictionary):
        # path to negative and positive comments
        self.positive_negative_path = positive_negative_path
        # path to only positive values from selection
        self.train_path = train_path
        # author_id -> [comment_ids, ...]
        self.user_representation = None
        self.commentid_to_article_id_dict = None
        self._build_user_representation_dict()
        self._build_comment_id_to_article_dict()

        self.dictionary = dictionary

        print('Length of dictionary:', len(self.dictionary))

        self.corpus_comment_id_to_index = None
        self.corpus_list = None

        self._prepare()

        logging.info('Data Fetcher successfully prepared')

    def _prepare(self):
        self.corpus_comment_id_to_index = pickle.load(open(
            CORPUS_COMMENT_ID_TO_INDEX,
            'rb'))
        self.corpus_list = pickle.load(open(
            TOKENIZED_COMMENTS_PATH,
            'rb'))

    def _build_user_representation_dict(self):
        df_train = pd.read_csv(self.train_path)
        df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])
        df_train = df_train.sort_values(by='timestamp', ascending=False)
        user_representation = df_train.groupby('author_id')['comment_id'].apply(
            lambda x: list(x)[:MAX_COMMENTS_USER_SECTION]).reset_index()
        user_representation.index = user_representation['author_id']
        self.user_representation = user_representation['comment_id'].to_dict()
        logging.info('User representations dictionary loaded')

    def _build_comment_id_to_article_dict(self):
        comment_to_article = pd.read_csv(self.train_path)
        comment_to_article.index = comment_to_article['comment_id']
        self.commentid_to_article_id_dict = comment_to_article['article_id'].to_dict()

    def _parse_line(self, line):
        return list(csv.reader([line]))[0]

    def load_comment_text(self, comment_id):
        # returns comment text for comment_id
        return self.corpus_list[self.corpus_comment_id_to_index[comment_id]]

    def tokenid2token(self, token_id):
        return self.dictionary[token_id]

    def get_user_representation(self, author_id, comment_id):
        comment_id_article = self.commentid_to_article_id_dict.get(comment_id)
        # exclude this article from user representation if it is present -> do not use situation to learn situation
        if comment_id_article is not None:
            user_rep = self.user_representation[author_id]
            return [comment_id for comment_id in user_rep if
                    self.commentid_to_article_id_dict[comment_id] != comment_id_article]
        else:
            return self.user_representation[author_id]

    def get_train_line(self, offset):
        with open(self.positive_negative_path, 'r', encoding='utf-8') as f:
            f.seek(offset)
            line = f.readline()
            author_id, comment_ids, y = self._parse_line(line)
            comment_ids_list = ast.literal_eval(comment_ids)
            return int(author_id), comment_ids_list, int(y)
