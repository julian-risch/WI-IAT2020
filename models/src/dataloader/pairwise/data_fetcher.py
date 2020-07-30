import ast

from src.dataloader.pointwise.data_fetcher import DataFetcher


class DataFetcherPairwise(DataFetcher):
    def __init__(self, positive_negative_path, train_path, dictionary, node2vec=False):
        self.node2vec = node2vec
        super(DataFetcherPairwise, self).__init__(positive_negative_path, train_path, dictionary)

    def _prepare(self):
        if not self.node2vec:
            super()._prepare()

    def get_train_line(self, offset):
        with open(self.positive_negative_path, 'r', encoding='utf-8') as f:
            f.seek(offset)
            line = f.readline()
            author_id, comment_ids_pos, comment_ids_neg = self._parse_line(line)
            comment_ids_pos_list = ast.literal_eval(comment_ids_pos)
            comment_ids_neg_list = ast.literal_eval(comment_ids_neg)
            return int(author_id), comment_ids_pos_list, comment_ids_neg_list
