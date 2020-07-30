import gensim
import torch
from src.dataloader.pairwise.data_fetcher import DataFetcherPairwise
from src.dataloader.pointwise.dataset import DatasetUserBehavior


class DatasetUserBehaviorPairwise(DatasetUserBehavior):
    def _load_data_fetcher(self, positive_negative_path, train_path, dict_path):
        self.data_fetcher = DataFetcherPairwise(positive_negative_path, train_path,
                                                gensim.corpora.Dictionary.load(dict_path))

    def __getitem__(self, index):
        # load train line
        author_id, comment_ids_pos, comment_ids_neg = self.data_fetcher.get_train_line(self.line_offset_dict[index])
        # load user representation as text
        user_repr = self.data_fetcher.get_user_representation(author_id, comment_ids_pos[0])

        out_user = self._get_section_content(user_repr, self.max_length_user_rep)
        out_section_pos = self._get_section_content(comment_ids_pos, self.max_length_comment_section)
        out_section_neg = self._get_section_content(comment_ids_pos, self.max_length_comment_section)

        section_emb_pos = self._get_section_emb(comment_ids_pos)
        section_emb_neg = self._get_section_emb(comment_ids_neg)

        return {
            'user_content': torch.from_numpy(out_user).type(torch.LongTensor),
            'user_emb': self.author_id_to_emb_pos_dict[
                author_id] if author_id in self.author_id_to_emb_pos_dict else 0,
            # fallback random embedding
            'section_content_pos': torch.from_numpy(out_section_pos).type(
                torch.LongTensor),
            'section_content_neg': torch.from_numpy(out_section_neg).type(
                torch.LongTensor),
            'section_emb_pos': torch.Tensor(section_emb_pos),
            'section_emb_neg': torch.Tensor(section_emb_neg),
            'y': 1
        }