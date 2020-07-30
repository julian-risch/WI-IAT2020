import numpy as np
import pandas as pd
from tqdm import tqdm
import csv
from sklearn.metrics.pairwise import cosine_similarity
import pickle

from constants import ROOT_PATH
from recommendation.prepare_collab import prepare_collab
from users.users import Users


user_vector_info_dict, article_to_position, vector_size = prepare_collab()
df_test = pd.read_csv(ROOT_PATH + 'test-set_all.csv')
df_test.index = df_test['comment_id']
comment_id_to_user_id = df_test['author_id'].to_dict()


def get_user_representation(user_id):
    user_info_dict = user_vector_info_dict[user_id]
    out = np.zeros(vector_size)
    for key, value in user_info_dict.items():
        out[article_to_position[key]] = value
    return out


def get_comment_section_representation(comment_section_ids):
    user_ids = [comment_id_to_user_id[comment_id] for comment_id in comment_section_ids]
    user_ids = list(filter(lambda user_id: user_id in user_vector_info_dict, user_ids))
    if len(user_ids) < 1:
        return np.zeros(vector_size)
    else:
        return np.mean(np.array([get_user_representation(user) for user in user_ids]), axis=0)


def evaluate(interacted_items_count_testset, hits, k):
    precision = hits / k
    recall = hits / interacted_items_count_testset
    return interacted_items_count_testset, precision, recall

