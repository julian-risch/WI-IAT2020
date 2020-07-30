import pandas as pd
import os
import printer
import numpy as np
import csv
from tqdm import tqdm


class Selector:
    def __init__(self, train_data_path, authors_path, output_path, partitions, n_partition, random_sample=False,
                 comments_to_train=None, replies=True, num_replies=3, most_k_recent_comments=42):
        self.train_data_df_top_comments = None
        self.train_data_df_replies = None
        self.output_path = output_path

        self.authors = None

        self._load_data(train_data_path)
        self._prepare_data()
        self._get_authors(authors_path, partitions, n_partition)
        self.has_header = False
        self.most_k_recent_comments = most_k_recent_comments
        self.random_sample = random_sample
        self.comments_to_train = comments_to_train
        self.get_replies = replies
        self.num_replies = num_replies

    def _load_data(self, train_data_path):
        if os.path.exists(train_data_path):
            self.train_data_df_top_comments = pd.read_csv(train_data_path)[
                ['timestamp', 'author_id', 'comment_id', 'article_id', 'parent_comment_id']]
            printer.print_success('Input Data loaded')
        else:
            printer.print_error('Train data path does not exist!')

    def _prepare_data(self):
        self.train_data_df_top_comments['timestamp'] = pd.to_datetime(self.train_data_df_top_comments['timestamp'])
        self.train_data_df_top_comments = self.train_data_df_top_comments.sort_values(by='timestamp', ascending=False)
        printer.print_progress('Data prepared')

    def _get_authors(self, authors_path, num_partitions, n_partition):
        self.authors = np.load(authors_path)
        printer.print_progress(f'Got {len(self.authors)} unique authors')
        self.authors = np.array_split(self.authors, num_partitions)
        printer.print_warning(f'Number of parititions: {len(self.authors)}')
        self.authors = self.authors[n_partition]
        printer.print_progress(f'Got {len(self.authors)} unique authors')

    def run(self):
        for i, author_id in enumerate(tqdm(self.authors)):
            own_comments = self.train_data_df_top_comments[self.train_data_df_top_comments['author_id'] == author_id]
            if self.comments_to_train is not None:
                own_comments = own_comments[own_comments['comment_id'].isin(self.comments_to_train)]
            if len(own_comments) > self.most_k_recent_comments and self.random_sample:
                own_comments = own_comments.sample(n=self.most_k_recent_comments, random_state=123)
            elif len(own_comments) > self.most_k_recent_comments:
                own_comments = own_comments.head(self.most_k_recent_comments)
            articles_df = own_comments.groupby('article_id')[['timestamp']].max().reset_index()

            for index, row in tqdm(articles_df.iterrows()):
                article_id = row['article_id']
                max_timestamp = row['timestamp']
                article_comments = self.train_data_df_top_comments[
                    self.train_data_df_top_comments['article_id'] == article_id]
                article_comments = article_comments[article_comments['timestamp'] <= max_timestamp]
                top_article_comments = article_comments[article_comments['parent_comment_id'].isna()]
                top_article_comments = top_article_comments[top_article_comments['author_id'] != author_id].head(10)

                comments = []
                top_comment_ids = top_article_comments['comment_id'].tolist()

                if self.get_replies:
                    for top_comment_id in top_comment_ids:
                        reply_article_comments = article_comments[~article_comments['parent_comment_id'].isna()]
                        reply_comments = \
                        reply_article_comments[reply_article_comments['parent_comment_id'] == top_comment_id][
                            'comment_id'].to_list()[-self.num_replies:]
                        comments.append(top_comment_id)
                        comments += reply_comments
                else:
                    comments = top_comment_ids

                line = [author_id, article_id, max_timestamp, comments]

                if not self.has_header:
                    with open(self.output_path, 'w') as f:
                        writer = csv.writer(f)
                        writer.writerow(['author_id', 'article_id', 'max_timestamp', 'comment_ids'])
                        writer.writerow(line)
                    self.has_header = True
                else:
                    with open(self.output_path, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(line)
