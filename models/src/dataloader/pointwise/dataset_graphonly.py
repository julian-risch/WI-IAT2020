import torch
import gensim
from src.dataloader.pointwise.data_fetcher import DataFetcher
from src.dataloader.pointwise.dataset import DatasetUserBehavior


class DatasetUserBehaviorNode2Vec(DatasetUserBehavior):
    def _load_data_fetcher(self, positive_negative_path, train_path, dict_path):
        self.data_fetcher = DataFetcher(positive_negative_path, train_path,
                                        gensim.corpora.Dictionary.load(dict_path))

    def __getitem__(self, index):
        # load train line
        author_id, comment_ids, y = self.data_fetcher.get_train_line(self.line_offset_dict[index])
        # load user representation as text
        section_emb = self._get_section_emb(comment_ids)

        return {
            'user_emb': self.author_id_to_emb_pos_dict[
                author_id] if author_id in self.author_id_to_emb_pos_dict else 0,
            # fallback random embedding
            'section_emb': torch.Tensor(section_emb),
            'commented': y
        }