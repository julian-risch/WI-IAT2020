import pandas as pd


class Articles:
    def __init__(self, path):
        self.path = path

        self.df = pd.read_csv(path)
        self.df.index = self.df['article_url']

    def get_start_urls(self):
        return self.df['article_url'].tolist()

    def get_article_id(self, article_url):
        return self.df.loc[article_url].article_id
