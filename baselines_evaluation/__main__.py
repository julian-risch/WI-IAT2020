import fire
from recommendation.bag_of_words import run
from recommendation.collaborative_filtering import run_collaborative


def recommend_bag(dataset):
    run(dataset)


def recommend_collaborative(dataset):
    run_collaborative(dataset)


if __name__ == '__main__':
    fire.Fire()
