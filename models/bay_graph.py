import fire
from ax.service.managed_loop import optimize

from src.constants import CHECKPOINT_SAVE_PATH
from train_graphonly import start_train

experiment_folder = 'graph/'


def evaluation_function(params):
    result = start_train(
        learning_rate=0.05,
        checkpoint_save_path=CHECKPOINT_SAVE_PATH + experiment_folder,
        pairwise=True,
        epochs=5,
        walk_length=params['walk_length'],
        walks_per_source=params['walks_per_source'],
        context_size=params['context_size'],
        num_dimensions=params['num_dimensions'],
        p=1,
        q=1)
    return {'out_acc': result['out_acc']}


def train_grid():
    parameters = [
        {"name": "walk_length", "type": "choice", "values": [10, 20, 30, 50, 100]},
        {"name": "walks_per_source", "type": "choice", "values": [10, 20, 30]},
        {"name": "context_size", "type": "choice", "values": [10, 20, 30]},
        {"name": "num_dimensions", "type": "choice", "values": [25, 50, 100]},
    ]

    best_parameters, values, experiment, model = optimize(parameters=parameters,
                                                          evaluation_function=evaluation_function,
                                                          objective_name='out_acc',
                                                          total_trials=15,)

    print('Best Parameters')
    print(best_parameters)
    print(values)


if __name__ == '__main__':
    fire.Fire(train_grid)
