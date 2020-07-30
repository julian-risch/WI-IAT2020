from torch import nn
import torch
import gensim
from fasttext import load_model
import numpy as np
import os
import glob

from src.constants import WORD_EMBEDDING_PATH, DICTIONARY_PATH


def create_node2vec_embedding_layer(user_embedding_path, freeze_weights):
    model = gensim.models.KeyedVectors.load_word2vec_format(user_embedding_path)

    num_authors = len(model.vocab) + 2
    EMBEDDING_DIM = model.vector_size
    embedding_matrix = np.zeros((num_authors, EMBEDDING_DIM))

    author_to_pos_dict = {}
    for i, word in enumerate(model.vocab):
        vector = model[word]
        embedding_matrix[i + 1] = vector

        author_id = int(word)
        author_to_pos_dict[author_id] = i + 1


    # embedding_matrix[0] = np.random.rand(EMBEDDING_DIM)
    embedding_matrix = torch.from_numpy(embedding_matrix)

    emb_layer = nn.EmbeddingBag(num_authors, EMBEDDING_DIM, mode='mean')  # fallback embedding at 0
    emb_layer.weight = nn.Parameter(embedding_matrix)
    if freeze_weights:
        emb_layer.weight.requires_grad = False
    return EMBEDDING_DIM, num_authors, emb_layer, author_to_pos_dict


def create_embedding_layer(freeze_weights):
    dictionary = gensim.corpora.Dictionary.load(DICTIONARY_PATH)
    num_words = len(dictionary)
    ft_model = load_model(WORD_EMBEDDING_PATH)
    EMBEDDING_DIM = ft_model.get_dimension()
    embedding_matrix = np.zeros((num_words + 1, EMBEDDING_DIM))

    for word, i in dictionary.token2id.items():
        embedding_matrix[i + 1] = ft_model.get_word_vector(word.lower()).astype('float32')

    # add PAD as pad term
    embedding_matrix[0] = ft_model.get_word_vector('<PAD/>').astype('float32')

    embedding_matrix = torch.from_numpy(embedding_matrix)

    emb_layer = nn.Embedding(num_words, EMBEDDING_DIM)
    emb_layer.weight = nn.Parameter(embedding_matrix)
    if freeze_weights:
        emb_layer.weight.requires_grad = False
    return EMBEDDING_DIM, num_words, emb_layer


def checkpoint_model(model, path, train_acc, loss, iterations):
    snapshot_prefix = os.path.join(path, 'snapshot')
    snapshot_path = snapshot_prefix + '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'.format(train_acc, loss.item(),
                                                                                        iterations)
    torch.save(model, snapshot_path)
    for f in glob.glob(snapshot_prefix + '*'):
        if f != snapshot_path:
            os.remove(f)


def save_config(file_path, **kwargs):
    with open(file_path, mode='w') as f:
        for key, value in kwargs.items():
            f.write(f'{key} = {value}\n')
    return kwargs
