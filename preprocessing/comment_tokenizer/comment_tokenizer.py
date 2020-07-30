import pandas as pd
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import csv
from sklearn.feature_extraction.text import CountVectorizer

import printer


class CommentTokenizer:
    def __init__(self, input_path, output_path):
        self.docs = pd.read_csv(input_path)
        self.docs = self.docs.fillna('0')
        self.comment_ids = self.docs['comment_id']
        self.docs = self.docs['comment_text'].tolist()
        self.tokenized_docs = []
        printer.print_progress('Text loaded')
        self.output_path = output_path

    def _write_header(self):
        with open(self.output_path, 'w', encoding='utf-8  ') as f:
            writer = csv.writer(f)
            writer.writerow(['comment_id', 'comment_text'])
        printer.print_progress('Header added')

    def run(self):
        self._write_header()
        index = 0

        for doc in tqdm(self.docs):
            comment = word_tokenize(doc)
            with open(self.output_path, 'a', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([self.comment_ids[index], comment])

            index += 1

        self.docs = None
