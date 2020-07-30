import csv
import ast


class DataCorpus:
    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        with open(self.file_path, encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                comment = ast.literal_eval(row[1])
                comment = [token.lower() for token in comment]
                yield comment
