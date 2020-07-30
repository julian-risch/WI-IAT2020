import pickle

import pandas as pd
from tqdm import tqdm
from gensim import corpora
from gensim.models import TfidfModel
import gensim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import csv

from constants import ROOT_PATH
from users.users import Users

users = Users()

dictionary_path = ROOT_PATH + 'bag_of_words/dictionary.dict'
dictionary = gensim.corpora.Dictionary.load(dictionary_path)

comment_id_to_position_corpus = pickle.load(open(ROOT_PATH + f'training_data/comment_id_to_position_{len(dictionary)}_words_dict.p', "rb"))

corpus_path = ROOT_PATH + 'bag_of_words/corpus.mm'
corpus = corpora.MmCorpus(corpus_path)

tfidf_model = TfidfModel.load(ROOT_PATH + 'bag_of_words/tfidf.model')

n_items = len(dictionary)

print('Number of Words: ', n_items)


def get_comment_position_bag_of_words_corpus(comment_id):
    return comment_id_to_position_corpus[comment_id]


def get_tfidf_vector(comment_id, matrix, index):
    position = get_comment_position_bag_of_words_corpus(comment_id)
    tfidf_values = tfidf_model[corpus[position]]
    for i, value in tfidf_values:
        matrix[index][i] = value
    return matrix


def get_author_tfidf_representation(author_id):
    comment_ids = users.get_user_representation(author_id)
    user_representation = np.zeros((len(comment_ids), n_items))
    for index, comment_id in enumerate(comment_ids):
        get_tfidf_vector(comment_id, user_representation, index)
    return np.mean(np.array(user_representation), axis=0)


def get_comment_position_bag_of_words_corpus(comment_id):
    return comment_id_to_position_corpus[comment_id]


def get_comment_section_representation(comment_section_ids):
    if len(comment_section_ids) < 1:
        return np.zeros(n_items)

    comments = np.zeros((len(comment_section_ids), n_items))

    for index, comment_id in enumerate(comment_section_ids):
        get_tfidf_vector(comment_id, comments, index)
    return np.mean(comments, axis=0)


def evaluate(interacted_items_count_testset, hits, k):
    precision = hits / k
    recall = hits / interacted_items_count_testset
    return interacted_items_count_testset, precision, recall
