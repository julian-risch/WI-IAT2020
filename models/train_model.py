import os
from collections import OrderedDict

import fire
import torch
import numpy as np
from torch import nn
from torchtrainer import TorchTrainer
from torchtrainer.callbacks import ProgressBar, CSVLogger, CSVLoggerIteration, VisdomLinePlotter, \
    ReduceLROnPlateauCallback
from torchtrainer.callbacks.visdom import VisdomIteration
from torchtrainer.metrics import BinaryAccuracy

from src.model.pointwise.model import HomophilyContentCNNFM
from src.utils.EarlyStoppingCallback import EarlyStoppingIteration
from src.utils.custom_checkpoint import CustomCheckpointEpoch

from src.constants import MAX_LENGTH_USER_REPRESENATION, MAX_LENGTH_COMMENT_SECTION, WORD_EMBEDDING_PATH, ROOT_PATH, \
    CHECKPOINT_SAVE_PATH
from src.dataloader.pairwise.utils import get_data_loader_pairwise
from src.model.pairwise.model import HomophilyContentCNNPairwise
from src.utils.utils import create_node2vec_embedding_layer, save_config
from src.dataloader.pointwise.utils import get_data_loader
from utils import create_directory


def print_divider():
    print('=' * 100)


def flatten_section_emb_list(section_emb):
    flatten_section_emb = []
    for section in section_emb:
        current_section = section
        # if empty list then use padding embedding (index = 0)
        if len(section) == 0:
            current_section = [0]
        for el in current_section:
            flatten_section_emb.append(el)
    return flatten_section_emb


def make_embeddingbag_parallel(input, offsets):
    input, offsets = input.unsqueeze(0).expand(torch.cuda.device_count(), input.size(0)), offsets.unsqueeze(0).expand(
        torch.cuda.device_count(), offsets.size(0))

    return input, offsets


def get_config(**args):
    return args


def tranform_section_emb(section_emb):
    section_emb = [[i for i in el if i != 0] for el in section_emb.tolist()]

    section_offsets = [0] + [len(section) if len(section) > 0 else 1 for section in section_emb]
    section_offsets = np.cumsum(np.array(section_offsets[:-1]))
    section_emb_offsets = torch.LongTensor(section_offsets).cuda()
    section_emb = torch.LongTensor(flatten_section_emb_list(section_emb)).cuda()

    section_emb, section_emb_offsets = section_emb, section_emb_offsets
    return section_emb, section_emb_offsets


def transform_fn(batch):
    y = batch['commented'].cuda()
    user_content = batch['user_content'].cuda()

    section_content = batch['section_content'].cuda()

    user_offsets = [0] + [1 for _ in batch['user_emb']]
    user_offsets = np.cumsum(np.array(user_offsets[:-1]))
    user_emb_offsets = torch.LongTensor(user_offsets).cuda()
    user_emb = batch['user_emb'].cuda()

    section_emb, section_emb_offsets = tranform_section_emb(batch['section_emb'])
    user_emb, user_emb_offsets = user_emb, user_emb_offsets

    return (user_content, user_emb, user_emb_offsets, section_content, section_emb,
            section_emb_offsets), y.float()


def transform_fn_pairwise(batch):
    y = batch['y'].cuda()
    user_content = batch['user_content'].cuda()

    section_content_pos = batch['section_content_pos'].cuda()
    section_content_neg = batch['section_content_neg'].cuda()

    user_offsets = [0] + [1 for _ in batch['user_emb']]
    user_offsets = np.cumsum(np.array(user_offsets[:-1]))
    user_emb_offsets = torch.LongTensor(user_offsets).cuda()
    user_emb = batch['user_emb'].cuda()
    user_emb, user_emb_offsets = user_emb, user_emb_offsets

    section_emb_pos, section_emb_offsets_pos = tranform_section_emb(batch['section_emb_pos'])
    section_emb_neg, section_emb_offsets_neg = tranform_section_emb(batch['section_emb_neg'])
    return (user_content, user_emb, user_emb_offsets, (section_content_pos, section_emb_pos, section_emb_offsets_pos),
            (section_content_neg, section_emb_neg, section_emb_offsets_neg)), y.float()


def load_pretrained(pretrained_deepconn_path, model, freeze=False):
    model_dict = model.state_dict()
    checkpoint = torch.load(pretrained_deepconn_path)
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params

    pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict and 'fm' not in k}
    if freeze:
        model.embedding.weight.requires_grad = False
        model.userCNN[0].weight.requires_grad = False
        model.userCNN[0].bias.requires_grad = False
        model.sectionCNN[0].weight.requires_grad = False
        model.sectionCNN[0].bias.requires_grad = False
        model.user_linear[0].weight.requires_grad = False
        model.user_linear[0].bias.requires_grad = False
        model.section_linear[0].weight.requires_grad = False
        model.section_linear[0].bias.requires_grad = False
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def start_train(
        checkpoint_save_path,
        user_embedding_path,
        pretrained_deepconn_path,
        gpu_ids=1,
        reduced_date_size=None,
        # CNN
        learning_rate=0.0001,
        num_epochs=3,
        batch_size=100,
        num_workers=4,
        user_num_kernels=50,  # number of kernels
        user_kernel_size=2,  # number of words in window
        latent_factors_deepconn=100,  # embedding size
        latent_factors=150,  # embedding size
        latent_factors_user=150,
        latent_factors_section=150,
        freeze_embeddings=True,
        dropout=0.1,
        pairwise=False,
        freeze=False,
):
    section_num_kernels = user_num_kernels
    section_kernel_size = user_kernel_size
    section_latent_factors = latent_factors_deepconn

    path, folder_name = create_directory(checkpoint_save_path)

    torch.cuda.empty_cache()

    save_config(os.path.join(path, 'parameter.config'),
                reduced_date_size=reduced_date_size,
                user_embedding_path=user_embedding_path,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                batch_size=batch_size,
                dropout=dropout,
                user_num_kernels=user_num_kernels,
                section_num_kernels=section_num_kernels,
                user_kernel_size=user_kernel_size,
                section_kernel_size=section_kernel_size,
                user_latent_factors=latent_factors_deepconn,
                latent_factors=latent_factors,
                section_latent_factors=section_latent_factors,
                latent_factors_user=latent_factors_user,
                latent_factors_section=latent_factors_section,
                freeze_embeddings=freeze_embeddings)

    NODE2VEC_EMB_DIM, num_authors, node2vec_emb_layer, author_to_pos_dict = create_node2vec_embedding_layer(
        user_embedding_path, True)

    model = None
    if pairwise:
        model = HomophilyContentCNNPairwise(node2vec_emb_layer,
                                            NODE2VEC_EMB_DIM,
                                            MAX_LENGTH_USER_REPRESENATION,
                                            MAX_LENGTH_COMMENT_SECTION,
                                            dropout=dropout,
                                            user_num_kernels=user_num_kernels,
                                            # number of kernels
                                            section_num_kernels=section_num_kernels,
                                            user_kernel_size=user_kernel_size,  # number of words in window
                                            section_kernel_size=section_kernel_size,
                                            latent_factors_deepconn=latent_factors_deepconn,  # embedding size
                                            latent_factors_user=latent_factors_user,
                                            latent_factors_section=latent_factors_section,
                                            freeze_embeddings=freeze_embeddings,
                                            )
    else:
        model = HomophilyContentCNNFM(node2vec_emb_layer,
                                      NODE2VEC_EMB_DIM,
                                      MAX_LENGTH_USER_REPRESENATION,
                                      MAX_LENGTH_COMMENT_SECTION,
                                      dropout=dropout,
                                      user_num_kernels=user_num_kernels,
                                      # number of kernels
                                      section_num_kernels=section_num_kernels,
                                      user_kernel_size=user_kernel_size,  # number of words in window
                                      section_kernel_size=section_kernel_size,
                                      latent_factors_deepconn=latent_factors_deepconn,  # embedding size
                                      latent_factors_user=latent_factors_user,
                                      latent_factors_section=latent_factors_section,
                                      freeze_embeddings=freeze_embeddings)

        if pretrained_deepconn_path is not None:
            load_pretrained(pretrained_deepconn_path, model, freeze)
            print('Successfully loaded pretrained DeepCoNN')

    config = get_config(reduced_date_size=reduced_date_size,
                        user_embedding_path=user_embedding_path,
                        learning_rate=learning_rate,
                        num_epochs=num_epochs,
                        batch_size=batch_size,
                        dropout=dropout,
                        user_num_kernels=user_num_kernels,
                        section_num_kernels=section_num_kernels,
                        user_kernel_size=user_kernel_size,
                        section_kernel_size=section_kernel_size,
                        user_latent_factors=latent_factors_deepconn,
                        latent_factors_deepconn=latent_factors_deepconn,
                        section_latent_factors=section_latent_factors,
                        latent_factors_user=latent_factors_user,
                        latent_factors_section=latent_factors_section,
                        freeze_embeddings=freeze_embeddings)

    if type(gpu_ids) == int:
        print(f'Selected cuda:{gpu_ids}')
        torch.cuda.set_device(gpu_ids)
        model.cuda()
    elif gpu_ids is not None and len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        model.cuda()

        if len(gpu_ids) > 1:
            model = nn.DataParallel(model, device_ids=gpu_ids)

    train_loader, val_loader = None, None
    if pairwise:
        train_loader, val_loader = get_data_loader_pairwise(author_to_pos_dict, batch_size, True,
                                                            num_workers,
                                                            reduced_date_size)
    else:
        train_loader, val_loader = get_data_loader(author_to_pos_dict, batch_size, True, num_workers,
                                                   reduced_date_size)

    loss = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = RAdam(model.parameters())

    trainer = TorchTrainer(model)

    plotter = VisdomLinePlotter(env_name=folder_name)

    validate_every = len(train_loader) // 4
    log_visdom_iteration_every = 200

    early_stopping = EarlyStoppingIteration(patience=15, min_delta=0.001, monitor='val_running_loss')
    callbacks = [
        ProgressBar(),
        CSVLogger(file=os.path.join(path, 'epoch_log.csv')),
        CSVLoggerIteration(file=os.path.join(path, 'iteration_log.csv')),
        VisdomIteration(plotter, monitor='running_loss', on_iteration_every=log_visdom_iteration_every),
        VisdomIteration(plotter, monitor='binary_acc', on_iteration_every=log_visdom_iteration_every),
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
