import fire
from ax.service.managed_loop import optimize

from src.constants import CHECKPOINT_SAVE_PATH
from train_deepconn import start_train

experiment_folder = 'deepconn_fm/'


def evaluation_function(params):
    result = {}

    result = start_train(
        gpu_ids=(0, 1),
        num_workers=4,
        reduced_date_size=None,
        batch_size=100,
        learning_rate=0.0001,
        dropout=0.1,
        checkpoint_save_path=CHECKPOINT_SAVE_PATH + experiment_folder,
        pairwise=False,
        num_epochs=5,
        user_num_kernels=params['user_num_kernels'],  # number of kernels
        user_kernel_size=params['user_kernel_size'],  # number of words in window
        user_latent_factors=params['user_latent_factors']
    )
    return {'out_acc': result['out_acc']}


def train_grid():
    parameters = [
        {"name": "user_num_kernels", "type": "choice", "values": [25, 50, 100]},
        {"name": "user_kernel_size", "type": "choice", "values": [2, 3, 4]},
        {"name": "user_latent_factors", "type": "choice", "values": [25, 50, 100]},
    ]

    best_parameters, values, experiment, model = optimize(parameters=parameters,
                                                          evaluation_function=evaluation_function,
                                                          objective_name='out_acc',
                                                          total_trials=10)
    print('Best Parameters')
    print(best_parameters)
    print(values)


if __name__ == '__main__':
    fire.Fire(train_grid)
