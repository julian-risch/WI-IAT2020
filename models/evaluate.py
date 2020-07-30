import csv
import pickle
import fire
import gensim
import pandas as pd
import torch
from tqdm import tqdm
from collections import OrderedDict

from src.constants import DICTIONARY_PATH, MAX_LENGTH_USER_REPRESENATION, MAX_LENGTH_COMMENT_SECTION, ROOT_PATH, \
    COMMENT_ID_TO_AUTHOR_DICT_PATH, TRAINING_DATA_PATH, TRAIN_SET_ALL_PATH
from src.dataloader.pointwise.data_fetcher import DataFetcher
from src.model.pointwise.deepconn_model import DeepCoNN
from src.model.pointwise.graph_only import CommunityGraphModel
from src.model.pointwise.model import HomophilyContentCNNFM
from src.testing.evaluation_data import EvaluationData
from src.testing.evaluation_dataset import EvaluationDataset
from src.utils.utils import create_node2vec_embedding_layer


def evaluate_situation(interacted_items_count_testset, hits, k):
    precision = hits / k
    recall = hits / interacted_items_count_testset
    return interacted_items_count_testset, precision, recall


def load_deep_conn_model(path):
    checkpoint = torch.load(path)
    config = checkpoint['config']
    state_dict = checkpoint['state_dict']

    del config['pairwise']
    del config['learning_rate']
    del config['batch_size']

    print(config)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params

    model = DeepCoNN(**config)
    model.load_state_dict(new_state_dict)
    return model


def load_graph_model(path, user_emb_path):
    # checkpoint = torch.load(path)
    # config = checkpoint['config']
    # state_dict = checkpoint['state_dict']

    # load params

    NODE2VEC_EMB_DIM, num_authors, node2vec_emb_layer, author_to_pos_dict = create_node2vec_embedding_layer(
        user_emb_path,
        True)
    model = CommunityGraphModel(node2vec_embedding_dim=NODE2VEC_EMB_DIM, node2vec_embedding=node2vec_emb_layer,
                                dropout=0.35)
    # model.load_state_dict(state_dict)
    print('Graph Model loaded')
    return model


def load_own_model(path, user_emb_path):
    print('Load HomophilyCoNN')
    NODE2VEC_EMB_DIM, num_authors, node2vec_emb_layer, author_to_pos_dict = create_node2vec_embedding_layer(
        user_emb_path,
        True)

    checkpoint = torch.load(path)
    config = checkpoint['config']
    state_dict = checkpoint['state_dict']

    # del config['pairwise']

    to_keep_set = ['node2vec_emb_layer', 'NODE2VEC_EMB_DIM',
                   'MAX_LENGTH_USER_REPRESENATION',
                   'MAX_LENGTH_COMMENT_SECTION',
                   'dropout',
                   'user_num_kernels',
                   'number of kernels',
                   'section_num_kernels',
                   'user_kernel_size',  # number of words in window
                   'section_kernel_size',
                   'latent_factors_deepconn',  # embedding size
                   'freeze_embeddings',
                   'latent_factors_user',
                   'latent_factors_section']

    to_keep_set = set(to_keep_set)

    keys = list(config.keys())
    for k in keys:
        if k not in to_keep_set:
            del config[k]

    print(config)
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #    name = k[7:]  # remove `module.`
    #    new_state_dict[name] = v

    model = HomophilyContentCNNFM(node2vec_emb_layer,
                                  NODE2VEC_EMB_DIM,
                                  MAX_LENGTH_USER_REPRESENATION,
                                  MAX_LENGTH_COMMENT_SECTION,
                                  **config)
    model.load_state_dict(state_dict)
    return model


def predict(model_type, model, user_content, user_emb, user_emb_offsets, section_content, section_emb,
            section_emb_offsets):
    if model_type == 'deepconn':
        return model(user_content.unsqueeze(0).cuda(), section_content.unsqueeze(0).cuda())
    elif model_type == 'graph':
        user_emb, user_emb_offsets = user_emb.cuda(), user_emb_offsets.cuda()
        section_emb, section_emb_offsets = section_emb.cuda(), section_emb_offsets.cuda()
        return model.pred(user_emb, user_emb_offsets, section_emb, section_emb_offsets)
    elif model_type == 'own':
        user_emb, user_emb_offsets = user_emb.cuda(), user_emb_offsets.cuda()
        section_emb, section_emb_offsets = section_emb.cuda(), section_emb_offsets.cuda()
        return model.pred(user_content.unsqueeze(0).cuda(), user_emb, user_emb_offsets,
                          section_content.unsqueeze(0).cuda(), section_emb, section_emb_offsets)


# model: 'deepconn', 'graph', 'own'
def evaluate(
        model_path,
        user_embedding_path,
        path_out,
        model_type='own',
        gpu_id=0):
    torch.cuda.set_device(gpu_id)
    model = None
    if model_type == 'deepconn':
        model = load_deep_conn_model(model_path)
    elif model_type == 'graph':
        # TODO
        model = load_graph_model(model_path, user_embedding_path)
    elif model_type == 'own':
        model = load_own_model(model_path, user_embedding_path)

    model.cuda()
    model.eval()

    print('Model loaded')

    NODE2VEC_EMB_DIM, num_authors, node2vec_emb_layer, author_to_pos_dict = create_node2vec_embedding_layer(
        user_embedding_path, True)

    comment_id_to_author_dict_path = COMMENT_ID_TO_AUTHOR_DICT_PATH

    comment_id_to_author_dict = pickle.load(open(comment_id_to_author_dict_path, 'rb'))

    with open(path_out, mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(
            ['author_id', 'k', 'hits_at_k', 'interacted_count', 'precision', 'recall', 'AP', 'documents', 'ranking'])

    K = [1, 3, 5, 10, 15, 20, 30]

    data_fetcher = DataFetcher(TRAINING_DATA_PATH,
                               TRAIN_SET_ALL_PATH,
                               gensim.corpora.Dictionary.load(DICTIONARY_PATH))

    evaluation_data = EvaluationData(data_fetcher, comment_id_to_author_dict, author_to_pos_dict)
    evaluation_dataset = EvaluationDataset()

    for author_id, user_rep, article_ids, sections, y in tqdm(evaluation_dataset, total=len(evaluation_dataset)):
        user_content, user_emb, user_emb_offsets = evaluation_data.get_author_data(author_id, user_rep)
        # import pdb; pdb.set_trace()
        score = []
        for section in sections:
            section_content, section_emb, section_emb_offsets = evaluation_data.get_comment_section(section)
            new_score = predict(model_type, model, user_content, user_emb, user_emb_offsets, section_content,
                                section_emb,
                                section_emb_offsets)

            score.append(float(new_score))

        out = pd.DataFrame({'v': score, 'flag': y, 'article_ids': article_ids})

        out = out.sort_values(by='v', ascending=False)
        article_ids_ranked = out['article_ids'].tolist()
        y_true = out['flag'].tolist()

        interacted_items_count_testset = 1

        for k in K:
            hits_at_k = sum(y_true[:k])
            interacted_items_count_testset, precision, recall = evaluate_situation(interacted_items_count_testset,
                                                                                   hits_at_k, k)
            precisions_at = []
            for i, el in enumerate(y_true[:k]):
                precisions_at.append(sum(y_true[:i + 1]) / (i + 1))

            AP_at_k = sum(precisions_at) / k

            with open(path_out, 'a', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(
                    [author_id, k, hits_at_k, interacted_items_count_testset, precision, recall, round(AP_at_k, 4),
                     len(out), article_ids_ranked])


if __name__ == '__main__':
    fire.Fire(evaluate)
