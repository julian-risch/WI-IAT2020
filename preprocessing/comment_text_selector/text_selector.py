import pandas as pd
import csv

import printer
from tqdm import tqdm


class CommentSelector:
    def __init__(self, raw_comments_path, selection_path, offset_path, output_path):
        self.selection_path = selection_path
        self.offset_path = offset_path

        self.raw_comments_path = raw_comments_path
        self.offset_dict = None

        self._load_offsets()

        self.header_dict = None
        self._prepare()

        self.selection_df = None
        self._load_selection()

        self.output_path = output_path

    def _prepare(self):
        header = ['article_id', 'comment_author_id', 'comment_id', 'comment_text', 'timestamp', 'parent_comment_id',
                  'upvotes']
        self.header_dict = {el: i for i, el in enumerate(header)}

    def _load_offsets(self):
        offset_df = pd.read_csv(self.offset_path)
        self.offset_dict = offset_df.set_index('comment_id').to_dict()['offset']
        printer.print_progress('Offsets loaded')

    def _load_selection(self):
        self.selection_df = pd.read_csv(self.selection_path)

    def _parse_line(self, line):
        return list(csv.reader([line]))[0]

    def _get_line(self, offset):
        with open(self.raw_comments_path, 'r', encoding='utf-8') as f:
            f.seek(offset)
            comment_text = self._parse_line(f.readline())[self.header_dict['comment_text']]
            return comment_text

    def _write_header(self):
        with open(self.output_path, 'w', encoding='utf-8  ') as f:
            writer = csv.writer(f)
            writer.writerow(['comment_id', 'comment_text'])
        printer.print_progress('Head added')

    def run(self):
        self._write_header()

        for index, row in tqdm(self.selection_df.iterrows(), total=self.selection_df.shape[0]):
            comment_id = row['comment_id']
            offset = self.offset_dict[comment_id]

            comment_text = self._get_line(offset)

            with open(self.output_path, 'a', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([comment_id, comment_text])






