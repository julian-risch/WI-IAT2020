import os
import ast

from src.constants import ROOT_PATH


def get_path(path):
    root_path = ROOT_PATH
    return os.path.join(root_path, path)


def parse_input_list(list_string):
    return ast.literal_eval(list_string)


def get_test_negative_comment_ids_path(part):
    return get_path(f'test_negative/partition-{part}_test.csv')

