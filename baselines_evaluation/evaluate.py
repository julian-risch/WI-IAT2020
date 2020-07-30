import csv
from random import shuffle

from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from evaluation_dataset import EvaluationDataset


def evaluate(interacted_items_count_testset, hits, k):
    precision = hits / k
    recall = hits / interacted_items_count_testset
    return interacted_items_count_testset, precision, recall


def run_evaluation(path_out, get_user_representation, get_comment_section_representation, rep_transform=None, random=False):
    with open(path_out, mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(['author_id', 'k', 'hits_at_k', 'interacted_count', 'precision', 'recall', 'AP', 'documents', 'ranking'])
    evaluation_dataset = EvaluationDataset()
    K = [1, 3, 5, 10, 15, 20, 30]
    for author_id, user_rep, article_ids, sections, y in tqdm(evaluation_dataset, total=len(evaluation_dataset)):
        if not random:
            user_rep = get_user_representation(author_id)
            rep = np.array([get_comment_section_representation(l) for l in sections])
            if rep_transform is not None:
                rep = rep_transform(rep)
            sim = list(cosine_similarity(user_rep.reshape(1, -1), rep)[0])

            out = pd.DataFrame({'v': sim, 'flag': y, 'article_ids': article_ids})
            out = out.sort_values(by='v', ascending=False)
            article_ids_ranked = out['article_ids'].tolist()
            y_true = out['flag'].tolist()
        else:
            shuffle(y)
            y_true = y
            article_ids_ranked = article_ids
            out = y

        interacted_items_count_testset = 1

        for k in K:
            hits_at_k = sum(y_true[:k])
            interacted_items_count_testset, precision, recall = evaluate(interacted_items_count_testset, hits_at_k, k)
            precisions_at = []
            for i, el in enumerate(y_true[:k]):
                precisions_at.append(sum(y_true[:i + 1]) / (i + 1))

            AP_at_k = sum(precisions_at) / k

            with open(path_out, 'a', encoding='utf-8') as f:
                writer = csv.writer(f)
                # ['author_id', 'k', 'hits_at_k', 'interacted_count', 'precision', 'recall']
                writer.writerow(
                    [author_id, k, hits_at_k, interacted_items_count_testset, precision, recall, round(AP_at_k, 4),
                     len(out), article_ids_ranked])
