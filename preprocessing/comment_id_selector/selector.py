import pandas as pd
import os
import printer
from tqdm import tqdm


class Selector:
    def __init__(self, train_data_path, output_path, min_comment_count):
        self.train_data_df = None
        self.output_path = output_path
        self.min_comment_count = min_comment_count

        self.authors = None

        self._load_data(train_data_path)
        self._prepare_data()
        self._get_authors()

        self.has_header = False

    def _load_data(self, train_data_path):
        if os.path.exists(train_data_path):
            self.train_data_df = pd.read_csv(train_data_path)[['timestamp', 'author_id', 'comment_id', 'article_id']]
            printer.print_success('Input Data loaded')
        else:
            printer.print_error('Train data path does not exist!')

    def _prepare_data(self):
        self.train_data_df['timestamp'] = pd.to_datetime(self.train_data_df['timestamp'])
        printer.print_progress('Data prepared')

    def _get_authors(self):
        comment_count_df = self.train_data_df.groupby('author_id')['comment_id'].count()
        comment_count_df = comment_count_df.reset_index()
        comment_count_df = comment_count_df[comment_count_df['comment_id'] >= self.min_comment_count]
        self.authors = comment_count_df['author_id'].unique()
        printer.print_progress(f'Got {len(self.authors)} unique authors')

    def run(self):
        for i, author_id in enumerate(tqdm(self.authors)):
            own_comments = self.train_data_df[self.train_data_df['author_id'] == author_id]
            articles_df = own_comments.groupby('article_id')[['timestamp']].max().reset_index()

            author_id_column = {'author_id': author_id}
            own_comments_flag = {'flag': True}

            own_comments = own_comments.assign(**own_comments_flag)

            comments = None

            for index, row in articles_df.iterrows():
                article_id = index
                max_timestamp = row['timestamp']

                article_comments = self.train_data_df[self.train_data_df['article_id'] == article_id]
                article_comments = article_comments[article_comments['timestamp'] <= max_timestamp]
                article_comments = article_comments[article_comments['author_id'] != author_id]

                article_comments.assign(**author_id_column)

                comments = pd.concat([comments, article_comments])

                other_comments_flag = {'flag': False}

            comments = comments.assign(**other_comments_flag)
            comments = pd.concat([own_comments, comments])

            comments = comments[['author_id', 'flag', 'comment_id', 'article_id']]

            if not self.has_header:
                comments.to_csv(self.output_path, mode='w', index=False)
                self.has_header = True
            else:
                comments.to_csv(self.output_path, header=False, mode='a', index=False)

