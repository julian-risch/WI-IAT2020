import pandas as pd
from tqdm import tqdm
import numpy as np
import csv
from datetime import timedelta


class TestDataSetGenerator:
    def __init__(self, positive_sample_path, test_data_path, article_dates_path, output_path, get_replies=True):
        self.positive_sample_df = pd.read_csv(positive_sample_path)
        self.test_data_df = pd.read_csv(test_data_path)
        self.output_path = output_path
        self.max_comments = 10
        self.num_replies = 3
        self.get_replies = get_replies

        self.article_dates_path = article_dates_path
        self.num_negative_samples = 50

        self.header = ['author_id', 'article_id', 'comment_ids', 'y']

        self.article_dates_df = pd.read_csv(self.article_dates_path)
        self._prepare_data()
        self._write_header()

    def _write_header(self):
        with open(self.output_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.header)

    def _prepare_data(self):
        self.positive_sample_df['timestamp'] = pd.to_datetime(self.positive_sample_df['timestamp'])
        self.test_data_df['timestamp'] = pd.to_datetime(self.test_data_df['timestamp'])
        self.test_data_df = self.test_data_df.sort_values(by='timestamp', ascending=False)

        self.article_dates_df['timestamp'] = pd.to_datetime(self.article_dates_df['date'])

    def _get_positive_sample_representation(self, article_id, timestamp, author_id):
        return [author_id, article_id, self._get_article_comments(article_id, timestamp), 1]

    def _get_negative_samples(self, article_id, timestamp, author_id):
        samples = []

        article_ids = self.test_data_df[self.test_data_df['author_id'] == author_id]['article_id'].unique()

        relevant_articles = self._get_relevant_articles_to_comment(timestamp)

        np.random.shuffle(relevant_articles)

        num_of_articles = 0
        for article in relevant_articles:
            if article not in article_ids:
                comments = self._get_article_comments(article, timestamp)
                line = [author_id, article, comments, 0]

                if len(comments) > 0:
                    num_of_articles += 1
                    samples.append(line)

                    if num_of_articles == self.num_negative_samples:
                        break

        if num_of_articles < self.num_negative_samples:
            relevant_articles = self._get_relevant_articles_to_comment_date_before(timestamp)

            np.random.shuffle(relevant_articles)

            for article in relevant_articles:
                if article not in article_ids:
                    comments = self._get_article_comments(article, timestamp)

                    line = [author_id, article, comments, 0]
                    if len(comments) > 0:
                        num_of_articles += 1
                        samples.append(line)

                        if num_of_articles == self.num_negative_samples:
                            break

        return samples

    def _get_relevant_articles_to_comment(self, date):
        # on the same day
        return self.article_dates_df[(self.article_dates_df['timestamp'].dt.date == date.date())]['article_id'].unique()

    def _get_relevant_articles_to_comment_date_before(self, date):
        return self.article_dates_df[self.article_dates_df['timestamp'].dt.date == date.date() - timedelta(days=1)][
            'article_id'].unique()

    def _get_article_comments(self, article, timestamp):
        article_comments = self.test_data_df[self.test_data_df['article_id'] == article]

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

    def _write_line(self, line):
        with open(self.output_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(line)

    def run(self):
        num_situations = 0
        for i, row in tqdm(self.positive_sample_df.iterrows(), total=len(self.positive_sample_df)):
            timestamp = row['timestamp']
            article_id = row['article_id']
            author_id = row['author_id']
            positive_sample = self._get_positive_sample_representation(article_id, timestamp, author_id)

            if len(positive_sample[2]) > 0:
                negative_samples = self._get_negative_samples(article_id, timestamp, author_id)

                if len(negative_samples) == self.num_negative_samples:
                    self._write_line(positive_sample)

                    for n in negative_samples:
                        self._write_line(n)

                    num_situations += 1
        print('FINISHED')
        print('Number of situations in evaluation dataset: ', num_situations)
