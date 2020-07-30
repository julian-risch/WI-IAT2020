import torch
import numpy as np

from src.constants import MAX_LENGTH_USER_REPRESENATION, MAX_LENGHT_COMMENT, MAX_LENGTH_COMMENT_SECTION
from train_model import make_embeddingbag_parallel


class EvaluationData:
    def __init__(self, data_fetcher, comment_id_to_author_dict, author_id_to_emb_pos_dict):
        self._data_fetcher = data_fetcher
        self._comment_id_to_author_dict = comment_id_to_author_dict
        self._author_to_emb_pos_dict = author_id_to_emb_pos_dict

    def _get_section_data(self, section):
        out_section = np.zeros((1, MAX_LENGTH_COMMENT_SECTION))
        pointer = 0
        for comment_id in section:
            comment_tokens = self._data_fetcher.load_comment_text(comment_id)
            comment_tokens = comment_tokens[:MAX_LENGHT_COMMENT - 1]
            for token in comment_tokens:
                out_section[0][pointer] = token + 1
                pointer += 1

        section_emb = [0] * MAX_LENGTH_COMMENT_SECTION
        for index, comment_id in enumerate(section):
            section_author_id = self._comment_id_to_author_dict[comment_id]
            if section_author_id in self._author_to_emb_pos_dict:
                section_emb[index] = self._author_to_emb_pos_dict[section_author_id]

        return torch.from_numpy(out_section).type(torch.LongTensor), torch.Tensor(section_emb)

    def get_author_data(self, author_id, author_representation):
        user_emb = torch.LongTensor([self._author_to_emb_pos_dict[author_id] if author_id in self._author_to_emb_pos_dict else 0])

        user_content = np.zeros((1, MAX_LENGTH_USER_REPRESENATION))
        pointer = 0
        for comment_id in author_representation:
            comment_tokens = self._data_fetcher.load_comment_text(comment_id)
            comment_tokens = comment_tokens[:MAX_LENGHT_COMMENT - 1]
            for token in comment_tokens:
                user_content[0][pointer] = token + 1
                pointer += 1

        user_offsets = [0]
        user_emb_offsets = torch.LongTensor(user_offsets)
        user_emb, user_emb_offsets = user_emb, user_emb_offsets
        return torch.from_numpy(user_content).type(torch.LongTensor), user_emb, user_emb_offsets

    def get_comment_section(self, section):
        section_content, section_emb = self._get_section_data(section)

        section_emb = [el for el in section_emb.tolist() if el != 0]

        section_offsets = [0]
        section_emb_offsets = torch.LongTensor(section_offsets)
        section_emb = torch.LongTensor(section_emb).long()

        section_emb, section_emb_offsets = section_emb, section_emb_offsets
        return section_content, section_emb, section_emb_offsets
