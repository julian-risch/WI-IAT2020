import os
import fire
import torch
from torch import nn
from torchtrainer import TorchTrainer
from torchtrainer.callbacks import ProgressBar, CSVLogger, CSVLoggerIteration,
from torchtrainer.metrics import BinaryAccuracy

from src.constants import MAX_LENGTH_USER_REPRESENATION, MAX_LENGTH_COMMENT_SECTION, WORD_EMBEDDING_PATH, \
    CHECKPOINT_SAVE_PATH
from src.dataloader.pairwise.utils import get_data_loader_section_pairwise
from src.dataloader.pointwise.utils import get_data_loader_section
from src.model.pairwise.deepconn_model import DeepCoNNPairwise
from src.model.pointwise.deepconn_model import DeepCoNN
from src.utils.EarlyStoppingCallback import EarlyStoppingIteration
from src.utils.custom_checkpoint import CustomCheckpointEpoch
from src.utils.utils import save_config
from utils import create_directory


def print_divider():
    print('=' * 100)


def transform_fn(batch):
    y = batch['commented'].cuda()
    user_content = batch['user_content'].cuda()

    section_content = batch['section_content'].cuda()

    return (user_content, section_content), y.float()


def transform_fn_pairwise(batch):
    y = batch['y'].cuda()
    user_content = batch['user_content'].cuda()

    section_content_pos = batch['section_content_pos'].cuda()
    section_content_neg = batch['section_content_neg'].cuda()

    return (user_content, section_content_pos, section_content_neg), y.float()


def start_train(
        gpu_ids=(0, 1),
        reduced_date_size=None,
        # CNN
        num_workers=4,
        num_epochs=6,
        batch_size=100 * 2,
        learning_rate=0.0001,
        user_num_kernels=100,  # number of kernels
        user_kernel_size=3,  # number of words in window
        user_latent_factors=150,  # embedding size
        dropout=0.1,
        freeze_embeddings=True,
        pairwise=False,
        checkpoint_save_path=CHECKPOINT_SAVE_PATH + 'deepconn_fm/'):

    path, folder_name = create_directory(checkpoint_save_path)
    torch.cuda.empty_cache()

    section_latent_factors = user_latent_factors
    section_num_kernels = user_num_kernels
    section_kernel_size = user_kernel_size

    model = None
    if pairwise:
        model = DeepCoNNPairwise(
            max_length_user_rep=MAX_LENGTH_USER_REPRESENATION,
            max_length_comment_section=MAX_LENGTH_COMMENT_SECTION,
            dropout=dropout,
            user_num_kernels=user_num_kernels,
            # number of kernels
            section_num_kernels=section_num_kernels,
            user_kernel_size=user_kernel_size,  # number of words in window
            section_kernel_size=section_kernel_size,
            user_latent_factors1=user_latent_factors,  # embedding size
            section_latent_factors1=section_latent_factors,
            freeze_embeddings=freeze_embeddings
        )
    else:
        model = DeepCoNN(
            max_length_user_rep=MAX_LENGTH_USER_REPRESENATION,
            max_length_comment_section=MAX_LENGTH_COMMENT_SECTION,
            dropout=dropout,
            user_num_kernels=user_num_kernels,
            # number of kernels
            section_num_kernels=section_num_kernels,
            user_kernel_size=user_kernel_size,  # number of words in window
            section_kernel_size=section_kernel_size,
            user_latent_factors1=user_latent_factors,  # embedding size
            section_latent_factors1=section_latent_factors,
            freeze_embeddings=freeze_embeddings
        )

    config = {
        'max_length_user_rep': MAX_LENGTH_USER_REPRESENATION,
        'max_length_comment_section': MAX_LENGTH_COMMENT_SECTION,
        'dropout': dropout,
        'user_num_kernels': user_num_kernels,
        # number of kernels
        'section_num_kernels': section_num_kernels,
        'user_kernel_size': user_kernel_size,  # number of words in window
        'section_kernel_size': section_kernel_size,
        'user_latent_factors1': user_latent_factors,  # embedding size
        'section_latent_factors1': section_latent_factors,
        'freeze_embeddings': freeze_embeddings,
        'pairwise': pairwise,
        'learning_rate': learning_rate,
        'batch_size': batch_size
    }

    save_config(os.path.join(path, 'parameters.config'), **config)

    if type(gpu_ids) == int:
        print(f'Selected cuda:{gpu_ids}')
        torch.cuda.set_device(gpu_ids)
        model.cuda()
    elif gpu_ids is not None and len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        model.cuda()

        if len(gpu_ids) > 1:
            model = nn.DataParallel(model, device_ids=list(gpu_ids))

    train_loader, val_loader = None, None
    if pairwise:
        train_loader, val_loader = get_data_loader_section_pairwise(batch_size, True, num_workers,
                                                                    reduced_date_size)
    else:
        train_loader, val_loader = get_data_loader_section(batch_size, True, num_workers,
                                                           reduced_date_size)

    loss = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    trainer = TorchTrainer(model)

    validate_every = len(train_loader) // 4

    # print('validation_every: ', validate_every)
    early_stopping = EarlyStoppingIteration(patience=15, min_delta=0.001, monitor='val_running_loss')

    callbacks = [
        ProgressBar(),
        CSVLogger(file=os.path.join(path, 'epoch_log.csv')),
        CSVLoggerIteration(file=os.path.join(path, 'iteration_log.csv')),
        CustomCheckpointEpoch(path, WORD_EMBEDDING_PATH, config, monitor='val_running_loss'),
        early_stopping
    ]

    metrics = [BinaryAccuracy()]

    trainer.prepare(optimizer, loss, train_loader, val_loader,
                    transform_fn=transform_fn if not pairwise else transform_fn_pairwise, callbacks=callbacks,
                    metrics=metrics, validate_every=validate_every)

    result = trainer.train(num_epochs, batch_size)

    print('Finished training with the following results:')
    print(result)

    result = early_stopping.best_log
    result['path'] = path
    result['out_acc'] = (result['val_binary_acc'], 0)
    return result


if __name__ == '__main__':
    fire.Fire(start_train)
