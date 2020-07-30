import torch
from src.dataloader.pointwise.dataset import DatasetUserBehavior


class DatasetUserBehaviorSectionOnly(DatasetUserBehavior):
    def __getitem__(self, index):
        # load train line
        author_id, comment_ids, y = self.data_fetcher.get_train_line(self.line_offset_dict[index])
        # load user representation as text
        user_repr = self.data_fetcher.get_user_representation(author_id, comment_ids[0])
        #
        out_user = self._get_section_content(user_repr, self.max_length_user_rep)
        out_section = self._get_section_content(comment_ids, self.max_length_comment_section)

        return {
            'user_content': torch.from_numpy(out_user).type(torch.LongTensor),
            'section_content': torch.from_numpy(out_section).type(
                torch.LongTensor),
            'commented': y
        }
