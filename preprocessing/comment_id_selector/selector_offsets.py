import csv
import subprocess
import printer
import pandas as pd
from tqdm import tqdm


class ByteOffsetCollectorSelector:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.header_dict = None

        self._prepare()

        self.number_of_lines = None

        self._get_num_of_lines()

        self.output = []
        self.output_header = ['comment_id', 'offset']

    def _prepare(self):
        header = ['author_id', 'article_id', 'max_timestamp', 'comment_ids']
        self.header_dict = {el: i for i, el in enumerate(header)}

    def _get_num_of_lines(self):
        self.number_of_lines = int(subprocess.check_output(["wc", "-l", self.input_path]).split()[0]) - 1
        printer.print_warning(f'Number of lines: {self.number_of_lines}')

    def _save(self):
        df = pd.DataFrame(self.output, columns=self.output_header)
        df.to_csv(self.output_path, index=False)
        printer.print_success(f'Saved to {self.output_path}')

    def _parse_line(self, line):
        return list(csv.reader([line]))[0]

    def start(self):
        printer.print_progress('Start collecting')
        with open(self.input_path, 'rb') as f:
            f.readline()  # move over header
            for i in tqdm(range(self.number_of_lines)):
                offset = f.tell()
                line = f.readline().decode('utf-8')
                if not line:
                    break
                comment_id = self._parse_line(line)[self.header_dict['author_id']]
                self.output.append([comment_id, offset])
        self._save()
