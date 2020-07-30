import pandas as pd
import logging
import csv

class TextCollector:
    def __init__(self, raw_comments_path, offset_path):
        self.raw_comments_path = raw_comments_path
        self.offset_path = offset_path
        self.offset_dict = None
        self.header_dict = None

        self._prepare()
        self._load_offsets()

    def _parse_line(self, line):
        return list(csv.reader([line]))[0]

    def _prepare(self):
        header = ['article_id', 'comment_author_id', 'comment_id', 'comment_text', 'timestamp', 'parent_comment_id',
                  'upvotes']
        self.header_dict = {el: i for i, el in enumerate(header)}

    def _get_line(self, offset):
        with open(self.raw_comments_path, 'r', encoding='utf-8') as f:
            f.seek(offset)
            comment_text = self._parse_line(f.readline())[self.header_dict['comment_text']]
            return comment_text

    def _load_offsets(self):
        offset_df = pd.read_csv(self.offset_path)
        self.offset_dict = offset_df.set_index('comment_id').to_dict()['offset']
        logging.info('Comment text offsets loaded')

    def get_comment_text(self, comment_id):
        return self._get_line(self.offset_dict[int(comment_id)])
