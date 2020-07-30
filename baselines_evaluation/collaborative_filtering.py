import fire

from constants import ROOT_PATH
from evaluate import run_evaluation
from recommendation.collaborative_filtering import get_user_representation, get_comment_section_representation


def run():
    path_out = f'{ROOT_PATH}results/evaluation_collaborative.csv'
    run_evaluation(path_out, get_user_representation, get_comment_section_representation)


if __name__ == '__main__':
    fire.Fire(run)
