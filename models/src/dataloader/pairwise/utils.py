from torch.utils.data import DataLoader

from src.dataloader.pairwise.dataset import DatasetUserBehaviorPairwise
from src.dataloader.pairwise.dataset_graphonly import DatasetUserBehaviorNode2VecPairwise

from src.constants import DICTIONARY_PATH, MAX_COMMENTS_USER_SECTION, MAX_LENGTH_USER_REPRESENATION, \
    MAX_LENGTH_COMMENT_SECTION, ROOT_PATH, POSITIVE_NEGATIVE_PATH_L2R, TRAIN_SET_ALL_PATH, COMMENT_ID_TO_AUTHOR_DICT_PATH, \
    LINE_TRAIN_OFFSET_DICT_PATH_L2R, VAL_DATA_PATH_L2R, LINE_VAL_OFFSET_DICT_PATH_L2R
from src.dataloader.pairwise.dataset_section import DatasetUserBehaviorSectionOnlyPairwise


def get_data_loader_pairwise(author_to_pos_dict, batch_size, SHUFFLE_TRAIN, num_workers,
                             reduced_date_size):
    train_dataset = DatasetUserBehaviorPairwise(
        positive_negative_path=POSITIVE_NEGATIVE_PATH_L2R,
        train_path=TRAIN_SET_ALL_PATH,
        dict_path=DICTIONARY_PATH,
        comment_id_to_author_dict_path=COMMENT_ID_TO_AUTHOR_DICT_PATH,
        line_offset_dict_path=LINE_TRAIN_OFFSET_DICT_PATH_L2R,
        author_id_to_emb_pos_dict=author_to_pos_dict,
        max_comments_user_section=MAX_COMMENTS_USER_SECTION,
        max_length_user_rep=MAX_LENGTH_USER_REPRESENATION,
        max_length_comment_section=MAX_LENGTH_COMMENT_SECTION,
    )

    val_dataset = DatasetUserBehaviorPairwise(
        positive_negative_path=VAL_DATA_PATH_L2R,
        train_path=TRAIN_SET_ALL_PATH,
        dict_path=DICTIONARY_PATH,
        comment_id_to_author_dict_path=COMMENT_ID_TO_AUTHOR_DICT_PATH,
        line_offset_dict_path=LINE_VAL_OFFSET_DICT_PATH_L2R,
        author_id_to_emb_pos_dict=author_to_pos_dict,
        max_comments_user_section=MAX_COMMENTS_USER_SECTION,
        max_length_user_rep=MAX_LENGTH_USER_REPRESENATION,
        max_length_comment_section=MAX_LENGTH_COMMENT_SECTION
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=SHUFFLE_TRAIN, num_workers=num_workers,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            drop_last=True)

    return train_loader, val_loader


def get_data_loader_node2vec_pairwise(author_to_pos_dict, batch_size, SHUFFLE_TRAIN, num_workers, reduced_date_size):
    train_dataset = None
    val_dataset = DatasetUserBehaviorNode2VecPairwise(
        positive_negative_path=VAL_DATA_PATH_L2R,
        train_path=TRAIN_SET_ALL_PATH,
        dict_path=DICTIONARY_PATH,
        comment_id_to_author_dict_path=COMMENT_ID_TO_AUTHOR_DICT_PATH,
        line_offset_dict_path=LINE_VAL_OFFSET_DICT_PATH_L2R,
        author_id_to_emb_pos_dict=author_to_pos_dict,
        max_comments_user_section=MAX_COMMENTS_USER_SECTION,
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    return None, val_loader


def get_data_loader_section_pairwise(batch_size, SHUFFLE_TRAIN, num_workers, reduced_date_size):
    train_dataset = DatasetUserBehaviorSectionOnlyPairwise(
        positive_negative_path=POSITIVE_NEGATIVE_PATH_L2R,
        train_path=TRAIN_SET_ALL_PATH,
        dict_path=DICTIONARY_PATH,
        comment_id_to_author_dict_path=COMMENT_ID_TO_AUTHOR_DICT_PATH,
        line_offset_dict_path=LINE_VAL_OFFSET_DICT_PATH_L2R,
        author_id_to_emb_pos_dict={},
        max_comments_user_section=MAX_COMMENTS_USER_SECTION,
        max_length_user_rep=MAX_LENGTH_USER_REPRESENATION,
        max_length_comment_section=MAX_LENGTH_COMMENT_SECTION,
    )

    val_dataset = DatasetUserBehaviorSectionOnlyPairwise(
        positive_negative_path=POSITIVE_NEGATIVE_PATH_L2R,
        train_path=TRAIN_SET_ALL_PATH,
        dict_path=DICTIONARY_PATH,
        comment_id_to_author_dict_path=COMMENT_ID_TO_AUTHOR_DICT_PATH,
        line_offset_dict_path=LINE_VAL_OFFSET_DICT_PATH_L2R,
        author_id_to_emb_pos_dict={},
        max_comments_user_section=MAX_COMMENTS_USER_SECTION,
        max_length_user_rep=MAX_LENGTH_USER_REPRESENATION,
        max_length_comment_section=MAX_LENGTH_COMMENT_SECTION,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=SHUFFLE_TRAIN, num_workers=num_workers,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            drop_last=True)

    return train_loader, val_loader
