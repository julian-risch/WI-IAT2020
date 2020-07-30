from torch.utils.data import DataLoader

from src.constants import DICTIONARY_PATH, MAX_COMMENTS_USER_SECTION, MAX_LENGTH_USER_REPRESENATION, \
    MAX_LENGTH_COMMENT_SECTION, ROOT_PATH, POSITIVE_NEGATIVE_PATH, TRAIN_SET_ALL_PATH, COMMENT_ID_TO_AUTHOR_DICT_PATH, \
    LINE_OFFSET_DICT_PATH, VAL_DATA_PATH, LINE_OFFSET_DICT_PATH_VAL
from src.dataloader.pointwise.dataset import DatasetUserBehavior
from src.dataloader.pointwise.dataset_graphonly import DatasetUserBehaviorNode2Vec
from src.dataloader.pointwise.dataset_section import DatasetUserBehaviorSectionOnly


def get_data_loader(author_to_pos_dict, batch_size, SHUFFLE_TRAIN, num_workers, reduced_date_size):
    train_dataset = DatasetUserBehavior(
        positive_negative_path=POSITIVE_NEGATIVE_PATH,
        train_path=TRAIN_SET_ALL_PATH,
        dict_path=DICTIONARY_PATH,
        comment_id_to_author_dict_path=COMMENT_ID_TO_AUTHOR_DICT_PATH,
        line_offset_dict_path=LINE_OFFSET_DICT_PATH,
        author_id_to_emb_pos_dict=author_to_pos_dict,
        max_comments_user_section=MAX_COMMENTS_USER_SECTION,
        max_length_user_rep=MAX_LENGTH_USER_REPRESENATION,
        max_length_comment_section=MAX_LENGTH_COMMENT_SECTION,
        reduce_percentage=reduced_date_size
    )

    val_dataset = DatasetUserBehavior(
        positive_negative_path=VAL_DATA_PATH,
        train_path=TRAIN_SET_ALL_PATH,
        dict_path=DICTIONARY_PATH,
        comment_id_to_author_dict_path=COMMENT_ID_TO_AUTHOR_DICT_PATH,
        line_offset_dict_path=LINE_OFFSET_DICT_PATH_VAL,
        author_id_to_emb_pos_dict=author_to_pos_dict,
        max_comments_user_section=MAX_COMMENTS_USER_SECTION,
        max_length_user_rep=MAX_LENGTH_USER_REPRESENATION,
        max_length_comment_section=MAX_LENGTH_COMMENT_SECTION,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=SHUFFLE_TRAIN, num_workers=num_workers,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            drop_last=True)

    return train_loader, val_loader


def get_data_loader_section(batch_size, SHUFFLE_TRAIN, num_workers, reduce_percentage):
    train_dataset = DatasetUserBehaviorSectionOnly(
        positive_negative_path=POSITIVE_NEGATIVE_PATH,
        train_path=TRAIN_SET_ALL_PATH,
        dict_path=DICTIONARY_PATH,
        line_offset_dict_path=LINE_OFFSET_DICT_PATH,
        max_comments_user_section=MAX_COMMENTS_USER_SECTION,
        max_length_user_rep=MAX_LENGTH_USER_REPRESENATION,
        max_length_comment_section=MAX_LENGTH_COMMENT_SECTION,
        reduce_percentage=reduce_percentage,
        author_id_to_emb_pos_dict={},
        comment_id_to_author_dict_path=COMMENT_ID_TO_AUTHOR_DICT_PATH,
    )

    val_dataset = DatasetUserBehaviorSectionOnly(
        positive_negative_path=VAL_DATA_PATH,
        train_path=TRAIN_SET_ALL_PATH,
        dict_path=DICTIONARY_PATH,
        line_offset_dict_path=LINE_OFFSET_DICT_PATH_VAL,
        max_comments_user_section=MAX_COMMENTS_USER_SECTION,
        max_length_user_rep=MAX_LENGTH_USER_REPRESENATION,
        max_length_comment_section=MAX_LENGTH_COMMENT_SECTION,
        author_id_to_emb_pos_dict={},
        comment_id_to_author_dict_path=COMMENT_ID_TO_AUTHOR_DICT_PATH,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=SHUFFLE_TRAIN, num_workers=num_workers,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            drop_last=True)

    return train_loader, val_loader


def get_data_loader_node2vec(author_to_pos_dict, batch_size, SHUFFLE_TRAIN, num_workers, reduce_percentage):
    train_dataset = None

    val_dataset = DatasetUserBehaviorNode2Vec(
        positive_negative_path=VAL_DATA_PATH,
        train_path=TRAIN_SET_ALL_PATH,
        dict_path=DICTIONARY_PATH,
        comment_id_to_author_dict_path=COMMENT_ID_TO_AUTHOR_DICT_PATH,
        line_offset_dict_path=LINE_OFFSET_DICT_PATH_VAL,
        author_id_to_emb_pos_dict=author_to_pos_dict,
        max_comments_user_section=MAX_COMMENTS_USER_SECTION,
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            drop_last=True)

    return None, val_loader
