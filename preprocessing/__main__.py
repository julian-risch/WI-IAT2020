import fire
import os
import pandas as pd
import multiprocessing as mp
import gensim
from bag_of_words.Corpus import DataCorpus
from bag_of_words.bag_of_words import TFIDFModel
from comment_id_selector.selector_offsets import ByteOffsetCollectorSelector
from comment_id_selector.selector_parallalizeable_minified import Selector as ParallelSelectorMin
import printer
from comment_id_selector_negative_examples.selector_parallizable import NegativeExampleSelector
from comment_text_selector.text_offset_collector import ByteOffsetCollector
from comment_text_selector.text_selector import CommentSelector
from comment_tokenizer.comment_tokenizer import CommentTokenizer
from gensim import corpora
import numpy as np
from tqdm import tqdm
import pickle
import ast

from comment_tokenizer.tokenized_text_offsets import ByteTokenizedOffsetCollector
from test_dataset_generator.generator import TestDataSetGenerator

# set it here for all processes
root_path = ''


def run_selector_parallel_min(partitions, n_partition):
    train_data_path = root_path + 'train-set_all.csv'
    authors_path = root_path + 'usable_authors_to_train_val.npy'
    output_path = root_path + f'train_positive/partition_{n_partition}.csv'
    selector = ParallelSelectorMin(train_data_path, authors_path, output_path, partitions, n_partition, None)

    printer.print_success('START')
    selector.run()
    printer.print_success('FINISHED')


def run_selector_negative_paralallel(partitions, n_partition):
    data_path = root_path + 'train-set_all.csv'
    authors_path = root_path + 'usable_authors_to_train_val.npy'
    article_dates_path = root_path + 'c_articles_dates.csv'
    output_path = root_path + f'train_negative/partition-{n_partition}.csv'
    k = 1
    np.random.seed(123)
    selector = NegativeExampleSelector(data_path, authors_path, article_dates_path, output_path, partitions,
                                       n_partition, k)

    printer.print_success('START')
    selector.run()
    printer.print_success('FINISHED')


def run_selector_parallel_min_test(partitions, n_partition):
    # train_data_path, authors_path, output_path, partitions, n_partition
    test_data_path = root_path + 'test-set_all.csv'
    authors_path = root_path + 'usable_authors_test.npy'
    output_path = root_path + f'test_positive/partition-{n_partition}.csv'
    selector = ParallelSelectorMin(test_data_path, authors_path, output_path, partitions, n_partition,
                                   random_sample=False)

    printer.print_success('START')
    selector.run()
    printer.print_success('FINISHED')


def run_selector_negative_paralallel_test(partitions, n_partition):
    test_path = root_path + 'test-set_all.csv'
    authors_path = root_path + 'usable_authors_test.npy'
    article_dates_path = root_path + 'c_articles_dates.csv'
    output_path = root_path + f'test_negative/partition-{n_partition}.csv'
    np.random.seed(123)
    selector = NegativeExampleSelector(test_path, authors_path, article_dates_path, output_path, partitions,
                                       n_partition, 50, True, False)

    printer.print_success('START')
    selector.run()
    printer.print_success('FINISHED')


def run_selector_negative_paralallel_validation(partitions, n_partition):
    val_data_path = root_path + 'validation-set_all.csv'
    authors_path = root_path + 'usable_authors_validation.npy'
    article_dates_path = root_path + 'c_articles_dates.csv'
    output_path = root_path + f'val_negative/partition-{n_partition}.csv'
    selector = NegativeExampleSelector(val_data_path, authors_path, article_dates_path, output_path, partitions,
                                       n_partition, 1, False, False)

    printer.print_success('START')
    selector.run()
    printer.print_success('FINISHED')


def run_selector_parallel_min_validation(partitions, n_partition):
    val_data_path = root_path + 'validation-set_all.csv'
    authors_path = root_path + 'usable_authors_validation.npy'
    output_path = root_path + f'val_positive/partition-{n_partition}.csv'
    selector = ParallelSelectorMin(val_data_path, authors_path, output_path, partitions, n_partition)

    printer.print_success('START')
    selector.run()
    printer.print_success('FINISHED')


def run_preprocessing_process(partitions, n_partition):
    run_selector_parallel_min(partitions, n_partition)
    run_selector_negative_paralallel(partitions, n_partition)
    run_selector_parallel_min_test(partitions, n_partition)
    run_selector_negative_paralallel_test(partitions, n_partition)
    run_selector_parallel_min_validation(partitions, n_partition)
    run_selector_negative_paralallel_validation(partitions, n_partition)


def run_evaluation_dataset_generator():
    positive_sample_path = root_path + 'positive_test_samples.csv'
    test_data_path = root_path + 'test-set_all.csv'
    article_dates_path = root_path + 'c_articles_dates.csv'
    output_path = root_path + 'evaluation_dataset.csv'
    test_generator = TestDataSetGenerator(positive_sample_path, test_data_path, article_dates_path, output_path)
    test_generator.run()


def run_preprocessing(partitions=4):
    processes = [mp.Process(target=run_preprocessing_process, args=(4, x)) for x in range(partitions)]
    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()


def collect_offsets(raw_input, output_path):
    collector = ByteOffsetCollector(raw_input, output_path)
    collector.start()


def collect_offsets_selector_negative():
    for i in range(4):
        path = f'negative/partition-{i}_test.csv'
        output = f'offsets-{i}.csv'
        raw_input = os.path.join(root_path, path)
        output_path = os.path.join(root_path, output)
        collector = ByteOffsetCollectorSelector(raw_input, output_path)
        collector.start()


def collect_offsets_selector_positive():
    dfs = []
    folder = 'val_negative'
    for i in range(4):
        path = f'{folder}/partition-{i}.csv'
        output = f'{folder}/offsets-{i}.csv'
        raw_input = os.path.join(root_path, path)
        output_path = os.path.join(root_path, output)
        collector = ByteOffsetCollectorSelector(raw_input, output_path)

        collector.start()
        output_df = pd.read_csv(output_path)
        output_df['part'] = i

        dfs.append(output_df)

    df = pd.concat(dfs)
    out = df.groupby(['comment_id', 'part'])['offset'].apply(list).reset_index()
    out['author_id'] = out['comment_id']
    out[['author_id', 'part', 'offset']].to_csv(root_path + f'/{folder}/offsets.csv', index=False)


def collect_texts(raw_comments_path, selection_path, offset_path, output_path):
    collector = CommentSelector(raw_comments_path, selection_path, offset_path, output_path)
    printer.print_success('START')
    collector.run()
    printer.print_success('FINISHED')


def tokenize_texts(input_path, output_path):
    tokenizer = CommentTokenizer(input_path, output_path)
    printer.print_success('START')
    tokenizer.run()
    printer.print_success('FINISHED')


def tfidf(dictionary_path, text_path, corpus_output_path, output_path):
    model = TFIDFModel(dictionary_path, text_path, corpus_output_path, output_path)
    model.run()


def make_dictionary(tokenized_path, output_path):
    corpus = DataCorpus(tokenized_path)
    dictionary = corpora.Dictionary(corpus)
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=200000)
    dictionary.save(output_path)


def complete_comment_corpus_selection(output_path):
    df_train = pd.read_csv(root_path + 'train-set_all.csv')
    df_test = pd.read_csv(root_path + 'test-set_all.csv')
    df_val = pd.read_csv(root_path + 'validation-set_all.csv')
    df = pd.concat([df_train, df_test, df_val])
    df.to_csv(output_path)


def prepare_model(tokenized_texts, dictionary_path):
    test_set = pd.read_csv(tokenized_texts)
    dictionary = gensim.corpora.Dictionary.load(dictionary_path)
    out_list = []
    out_dict = {}
    for index, row in tqdm(test_set.iterrows(), total=len(test_set)):
        out_list.append([dictionary.token2id[token.lower()] for token in ast.literal_eval(row['comment_text']) if
                         token.lower() in dictionary.token2id])
        out_dict[row['comment_id']] = index
    # %%
    pickle.dump(out_dict, open(
        f'{root_path}training_data/comment_id_to_position_{len(dictionary)}_words_dict.p', "wb"))
    # %%
    pickle.dump(out_list,
                open(f'{root_path}training_data/tokenized_comments_{len(dictionary)}_words_list.p',
                     "wb"))


def prepare_node2vec_helper():
    df = pd.read_csv(root_path + 'complete_selection.csv')
    df.index = df['comment_id']
    dic = df['author_id'].to_dict()
    pickle.dump(dic,
                open(root_path + "training_data/commet_ids_to_author_id_test-val_train.p", "wb"))


def prepare(raw_comments_path, offset_path):
    selection_output = root_path + 'complete_selection.csv'
    complete_comment_corpus_selection(selection_output)

    selection_path = selection_output
    output_path = root_path + 'train_test_val_texts.csv'
    collect_texts(raw_comments_path, selection_path, offset_path, output_path)
    print('Texts collected Train Test Val')
    input_path = output_path
    output_path = root_path + 'train_test_texts_tokenized.csv'
    tokenize_texts(input_path, output_path)
    print('Texts tokenized Train Test Val')

    selection_path = root_path + 'train-set_all.csv'
    offset_path = root_path + 'c_comments_offsets.csv'
    output_path = root_path + 'train_texts.csv'
    collect_texts(raw_comments_path, selection_path, offset_path, output_path)
    input_path = output_path
    output_path = root_path + 'train_texts_tokenized.csv'
    tokenize_texts(input_path, output_path)

    prepare_text()
    prepare_model()


def prepare_model(tokenized_path, dictionary_path):
    # Prepare Helper for PyTorch Models
    offsets_tokenized_text(tokenized_path, root_path + 'texts_tokenized_offsets.csv')
    print('Tokenized Text Offsets collected')
    prepare_model(tokenized_path, dictionary_path)
    print('Model Helpers created')


def prepare_text():
    #
    # # Tokenize Train Comments
    input_path = root_path + 'train_texts.csv'
    output_path = root_path + 'train_texts_tokenized.csv'
    tokenize_texts(input_path, output_path)
    print('Texts tokenized Train ')
    # # Tokenize TrainTestVal Comments
    output_path = root_path + 'train_test_texts_tokenized.csv'
    input_path = root_path + 'train_test_val_texts.csv'
    tokenize_texts(input_path, output_path)
    print('Texts tokenized Train Test Val')
    #
    # # Make Dictionary from Train Texts
    tokenized_path = root_path + 'train_texts_tokenized.csv'
    output_path = root_path + 'bag_of_words/dictionary.dict'
    make_dictionary(tokenized_path, output_path)
    #
    # # Make TFIDF MODEL for Analysis
    dictionary_path = root_path + 'bag_of_words/dictionary.dict'
    corpus_output_path = root_path + 'bag_of_words/corpus.mm'
    text_path = root_path + 'train_texts_tokenized.csv'
    output_path = root_path + 'bag_of_words/tfidf.model'
    model = TFIDFModel(dictionary_path, text_path, corpus_output_path, output_path)
    model.run()
    model._save()

    # MAKE CORPUS
    dictionary_path = root_path + 'bag_of_words/dictionary.dict'
    output_path = root_path + 'bag_of_words/tfidf2.model'
    corpus_output_path = root_path + 'bag_of_words/corpus.mm'
    tokenized_path = root_path + 'train_test_texts_tokenized.csv'
    model = TFIDFModel(dictionary_path, tokenized_path, corpus_output_path, output_path)
    model.run()
    model._save_corpus()

    prepare_model(tokenized_path, dictionary_path)


def offsets_tokenized_text(input_path, output_path):
    collector = ByteTokenizedOffsetCollector(input_path, output_path)
    collector.start()


if __name__ == '__main__':
    fire.Fire()
