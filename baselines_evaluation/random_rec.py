import fire

from constants import ROOT_PATH
from evaluate import run_evaluation


def run():
    path_out = f'{ROOT_PATH}results/evaluation_random.csv'
    run_evaluation(path_out, None, None, random=True)


if __name__ == '__main__':
    fire.Fire(run)
