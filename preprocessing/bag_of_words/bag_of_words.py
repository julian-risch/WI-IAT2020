import gensim
import subprocess
import logging
import printer
from tqdm import tqdm
from gensim.models import TfidfModel
from bag_of_words.Corpus import DataCorpus

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class TFIDFModel:
    def __init__(self, dictionary_path, text_path, corpus_output_path, tfidf_output_path):
        self.dictionary = gensim.corpora.Dictionary.load(dictionary_path)
        self.data = DataCorpus(text_path)
        self.corpus = []

        self.number_of_lines = 0
        self._get_num_of_lines(text_path)
        self.model = None
        self.output_path = tfidf_output_path
        self.corpus_output_path = corpus_output_path

    def _get_num_of_lines(self, path):
        self.number_of_lines = int(subprocess.check_output(["wc", "-l", path]).split()[0]) - 1
        printer.print_warning(f'Number of lines: {self.number_of_lines}')

    def _save(self):
        self.model.save(self.output_path)

    def _save_corpus(self):
        gensim.corpora.MmCorpus.serialize(self.corpus_output_path, self.corpus)

    def make_corpus(self):
        for line in tqdm(self.data, total=self.number_of_lines):
            self.corpus.append(self.dictionary.doc2bow(line))
        printer.print_success('Finished to create corpus')

    def make_tf_idf(self):
        printer.print_progress('Run TFIDF Model')
        self.model = TfidfModel(self.corpus, normalize=False)
        printer.print_success('Finished to create corpus')

    def run(self):
        self.make_corpus()
        self.make_tf_idf()

