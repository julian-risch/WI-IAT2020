import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import gensim
import random

from src.constants import MAX_LENGHT_COMMENT, MAX_LENGTH_COMMENT_SECTION, MAX_LENGTH_USER_REPRESENATION
from src.dataloader.pointwise.data_fetcher import DataFetcher


class DatasetUserBehavior(Dataset):
    def __init__(self, positive_negative_path, train_path, dict_path, line_offset_dict_path,
                 comment_id_to_author_dict_path, author_id_to_emb_pos_dict, max_comments_user_section=MAX_LENGTH_USER_REPRESENATION,
                 max_length_user_rep=MAX_LENGTH_USER_REPRESENATION,
                 max_length_comment_section=MAX_LENGTH_COMMENT_SECTION, reduce_percentage=None, comments_to_train_path=None):

        self.line_offset_dict = pickle.load(open(line_offset_dict_path, "rb"))

        self.reduce_percentage = reduce_percentage
        if self.reduce_percentage is not None:
            # reduce number of training samples by sampling
            self._random_sample()

        self.comment_id_to_author_dict = pickle.load(open(comment_id_to_author_dict_path, 'rb'))

        self.data_fetcher = None
        self._load_data_fetcher(positive_negative_path, train_path, dict_path)

        self.max_length_user_rep = max_length_user_rep
        self.max_length_comment_section = max_length_comment_section

        self.author_id_to_emb_pos_dict = author_id_to_emb_pos_dict
        self.max_comments_user_section = max_comments_user_section

    def __len__(self):
        return len(self.line_offset_dict)

    def _load_data_fetcher(self, positive_negative_path, train_path, dict_path):
        self.data_fetcher = DataFetcher(positive_negative_path, train_path,
                                        gensim.corpora.Dictionary.load(dict_path))

    def _get_section_emb(self, comment_ids):
        section_emb = [0] * self.max_comments_user_section
        for index, comment_id in enumerate(comment_ids):
            section_author_id = self.comment_id_to_author_dict[comment_id]
            if section_author_id in self.author_id_to_emb_pos_dict:
                section_emb[index] = self.author_id_to_emb_pos_dict[section_author_id]

        return section_emb

    def _get_section_content(self, comment_ids, max_length):
        out = np.zeros((1, max_length))
        pointer = 0
        for comment_id in comment_ids:
            comment_tokens = self.data_fetcher.load_comment_text(comment_id)  # cut of last at 0.99
            comment_tokens = comment_tokens[:MAX_LENGHT_COMMENT - 1]
            for token in comment_tokens:
                out[0][pointer] = token + 1
                pointer += 1

        return out

    def __getitem__(self, index):
        # load train line
        author_id, comment_ids, y = self.data_fetcher.get_train_line(self.line_offset_dict[index])
        # load user representation as text
        user_repr = self.data_fetcher.get_user_representation(author_id, comment_ids[0])

        out_user = self._get_section_content(user_repr, self.max_length_user_rep)
        out_section = self._get_section_content(comment_ids, self.max_length_comment_section)
        section_emb = self._get_section_emb(comment_ids)

        return {
            'user_content': torch.from_numpy(out_user).type(torch.LongTensor),
            'user_emb': self.author_id_to_emb_pos_dict[
                author_id] if author_id in self.author_id_to_emb_pos_dict else 0,
            # fallback random embedding
            'section_content': torch.from_numpy(out_section).type(
                torch.LongTensor),
            'section_emb': torch.Tensor(section_emb),
            'commented': y
        }

    def _random_sample(self):
        random.seed(123)
        k = (1 - self.reduce_percentage) * len(self.line_offset_dict.keys())
        print(k)
        print(f'Number of samples before reducing it: {len(self.line_offset_dict)}')
        to_pop = random.sample(list(self.line_offset_dict.keys()), k=int(k))
        for el in to_pop:
            del self.line_offset_dict[el]
        self.line_offset_dict = {index: value for index, (key, value) in enumerate(self.line_offset_dict.items())}
        print(f'Random sampled {self.reduce_percentage} training samples, current length {len(self.line_offset_dict)}')
