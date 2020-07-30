import gensim
import torch
from src.dataloader.pairwise.data_fetcher import DataFetcherPairwise
from src.dataloader.pointwise.dataset_graphonly import DatasetUserBehaviorNode2Vec


class DatasetUserBehaviorNode2VecPairwise(DatasetUserBehaviorNode2Vec):

    def _load_data_fetcher(self, positive_negative_path, train_path, dict_path):
        self.data_fetcher = DataFetcherPairwise(positive_negative_path, train_path,
                                                gensim.corpora.Dictionary.load(dict_path), node2vec=True)

    def __getitem__(self, index):
        # load train line
        author_id, comment_ids_pos, comment_ids_neg = self.data_fetcher.get_train_line(self.line_offset_dict[index])
        # load user representation as text
        section_emb_pos = self._get_section_emb(comment_ids_pos)
        section_emb_neg = self._get_section_emb(comment_ids_neg)

        return {
            'user_emb': self.author_id_to_emb_pos_dict[
                author_id] if author_id in self.author_id_to_emb_pos_dict else 0,
            # fallback random embedding
            'section_emb_pos': torch.Tensor(section_emb_pos),
            'section_emb_neg': torch.Tensor(section_emb_neg),
            'y': 1
        }
