import torch
import gensim
from src.dataloader.pairwise.data_fetcher import DataFetcherPairwise
from src.dataloader.pointwise.dataset import DatasetUserBehavior


class DatasetUserBehaviorSectionOnlyPairwise(DatasetUserBehavior):
    def _load_data_fetcher(self, positive_negative_path, train_path, dict_path):
        self.data_fetcher = DataFetcherPairwise(positive_negative_path, train_path,
                                                gensim.corpora.Dictionary.load(dict_path))

    def __getitem__(self, index):
        # load train line
        author_id, comment_ids_pos, comment_ids_neg = self.data_fetcher.get_train_line(self.line_offset_dict[index])
        # load user representation as text
        user_repr = self.data_fetcher.get_user_representation(author_id)
        #
        out_user = self._get_section_content(user_repr, self.max_length_user_rep)
        out_section_pos = self._get_section_content(comment_ids_pos, self.max_length_comment_section)
        out_section_neg = self._get_section_content(comment_ids_neg, self.max_length_comment_section)

        return {
            'user_content': torch.from_numpy(out_user).type(torch.LongTensor),
            'section_content_pos': torch.from_numpy(out_section_pos).type(
                torch.LongTensor),
            'section_content_neg': torch.from_numpy(out_section_neg).type(
                torch.LongTensor),
            'y': 1
        }
