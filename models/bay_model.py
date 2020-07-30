import fire
from ax.service.managed_loop import optimize

from src.constants import CHECKPOINT_SAVE_PATH
from train_model import start_train

experiment_folder = 'main_model/'
user_embedding_path = ''
pretrained_deepconn_path = ''


def evaluation_function(params):
    result = start_train(
        user_embedding_path=user_embedding_path,
        pretrained_deepconn_path=pretrained_deepconn_path,
        batch_size=100,
        learning_rate=0.0001,
        dropout=0.1,
        checkpoint_save_path=CHECKPOINT_SAVE_PATH + experiment_folder,
        pairwise=False,
        num_epochs=3,
        user_num_kernels=100,  # number of kernels
        user_kernel_size=3,  # number of words in window
        latent_factors_deepconn=50,
        latent_factors_user=params['latent_factors_user'],
        latent_factors_section=params['latent_factors_user'],
        freeze=False,
        gpu_ids=0
    )
    return {'out_acc': result['out_acc']}


def train_grid():
    parameters = [
        {"name": "latent_factors_user", "type": "choice", "values": [25, 50, 100]},
    ]

    best_parameters, values, experiment, model = optimize(parameters=parameters,
                                                          evaluation_function=evaluation_function,
                                                          objective_name='out_acc',
                                                          total_trials=3)
    print('Best Parameters')
    print(best_parameters)
    print(values)


if __name__ == '__main__':
    fire.Fire(train_grid)
