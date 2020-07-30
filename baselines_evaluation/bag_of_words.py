import fire

from constants import ROOT_PATH
from evaluate import run_evaluation
from recommendation.bag_of_words import get_author_tfidf_representation, get_comment_section_representation


def run():
    path_out = ROOT_PATH + '/results/evaluation_bow.csv'
    run_evaluation(path_out, get_author_tfidf_representation, get_comment_section_representation)


if __name__ == '__main__':
    fire.Fire(run)
