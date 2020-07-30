import fire

from constants import ROOT_PATH
from evaluate import run_evaluation
from recommendation.article_titles import get_comment_section_representation, get_user_representation


def rep_transform(rep):
    nsamples, nx, ny = rep.shape
    return rep.reshape((nsamples, nx * ny))


def run():
    path_out = ROOT_PATH + '/results/evaluation_data_titles.csv'
    run_evaluation(path_out, get_user_representation, get_comment_section_representation, rep_transform)


if __name__ == '__main__':
    fire.Fire(run)
