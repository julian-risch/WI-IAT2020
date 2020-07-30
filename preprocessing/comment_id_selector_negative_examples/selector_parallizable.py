import pandas as pd
import os
import printer
import numpy as np
import csv
from tqdm import tqdm
from datetime import timedelta


class NegativeExampleSelector:
    def __init__(self, data_path, authors_path, article_dates_path, output_path, partitions, n_partition, k, test=False,
                 random_sample=False, replies=True):
        self.data_path = data_path
        self.data_df = None

        self.article_dates_path = article_dates_path
        self.article_dates_df = None

        self.output_path = output_path
        self.author_path = authors_path

        self.authors = None
        self.has_header = False

        self._load_data()
        self._prepare_data()
        self._get_authors(authors_path, partitions, n_partition)
        self.k = k
        self.test = test
        self.day_before_count_max = 50
        self.most_k_recent_comments = 42
        self.max_comments = 10
        self.random_sample = random_sample

        self.get_replies = replies
        self.num_replies = 3

    def _load_data(self):
        self.article_dates_df = pd.read_csv(self.article_dates_path)
        self.article_dates_df['timestamp'] = pd.to_datetime(self.article_dates_df['date'])

        if os.path.exists(self.data_path):
            self.data_df = pd.read_csv(self.data_path)[
                ['timestamp', 'author_id', 'comment_id', 'article_id', 'parent_comment_id']]
            printer.print_success('Input Data loaded ')
        else:
            printer.print_error('Train data path does not exist!')

    def _prepare_data(self):
        self.data_df['timestamp'] = pd.to_datetime(self.data_df['timestamp'])
        self.data_df = self.data_df.sort_values(by='timestamp', ascending=False)
        printer.print_progress('Data prepared')

    def _get_relevant_articles_to_comment(self, date):
        # on the same day
        return self.article_dates_df[(self.article_dates_df['timestamp'].dt.date == date.date())]['article_id'].unique()

    def _get_relevant_articles_to_comment_date_before(self, date):
        return self.article_dates_df[self.article_dates_df['timestamp'].dt.date == date.date() - timedelta(days=1)][
            'article_id'].unique()

    def _get_authors(self, authors_path, num_partitions, n_partition):
        self.authors = np.load(authors_path)
        np.random.shuffle(self.authors)
        printer.print_warning(f'Number of authors: {len(self.authors)}')
        self.authors = np.array_split(self.authors, num_partitions)
        printer.print_warning(f'Number of parititions: {len(self.authors)}')
        self.authors = self.authors[n_partition]
        printer.print_progress(f'Got {len(self.authors)} unique authors')

    def _get_article_comments(self, article, timestamp):
        article_comments = self.data_df[self.data_df['article_id'] == article]

        article_comments = article_comments[article_comments['timestamp'] < timestamp]
        top_article_comments = article_comments[article_comments['parent_comment_id'].isna()].head(self.max_comments)

        comments = []
        top_comment_ids = top_article_comments['comment_id'].tolist()

        if self.get_replies:
            for top_comment_id in top_comment_ids:
                reply_article_comments = article_comments[~article_comments['parent_comment_id'].isna()]
                reply_comments = reply_article_comments[reply_article_comments['parent_comment_id'] == top_comment_id][
                                     'comment_id'].to_list()[-self.num_replies:]
                comments.append(top_comment_id)
                comments += reply_comments
        else:
            comments = top_comment_ids

        return comments

    def run(self):
        for author_id in tqdm(self.authors):
            own_comments = self.data_df[self.data_df['author_id'] == author_id]
            if len(own_comments) > self.most_k_recent_comments and self.random_sample:
                own_comments = own_comments.sample(n=self.most_k_recent_comments, random_state=123)
            elif len(own_comments) > self.most_k_recent_comments:
                own_comments = own_comments.head(self.most_k_recent_comments)
            articles_df = own_comments.groupby('article_id')[['timestamp']].max().reset_index()
            article_ids = articles_df['article_id'].unique()
            np.random.shuffle(article_ids)
            for index, row in articles_df.iterrows():
                relevant_articles = self._get_relevant_articles_to_comment(row['timestamp'])

                np.random.shuffle(relevant_articles)

                num_of_articles = 0
                timestamp = row['timestamp']

                for article in relevant_articles:
                    if article not in article_ids:
                        comments = self._get_article_comments(article, timestamp)
                        line = [author_id, article, row['timestamp'], comments]

                        if len(comments) > 0:
                            num_of_articles += 1
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

                            if num_of_articles == self.k:
                                break

                day_before_count = 0
                if (num_of_articles < self.k or num_of_articles < 1) and self.test:
                    relevant_articles = self._get_relevant_articles_to_comment_date_before(timestamp)

                    np.random.shuffle(relevant_articles)

                    num_of_articles = 0

                    for article in relevant_articles:
                        if article not in article_ids:
                            comments = self._get_article_comments(article, timestamp)

                            line = [author_id, article, row['timestamp'], comments]

                            if len(comments) > 0:
                                num_of_articles += 1
                                day_before_count += 1
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

                                if num_of_articles == self.k or day_before_count >= self.day_before_count_max:
                                    break
