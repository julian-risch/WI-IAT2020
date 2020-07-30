import os

import fire
import numpy as np
import torch
from torch import nn
from torchtrainer import TorchTrainer
from torchtrainer.average_meter import AverageMeter
from torchtrainer.callbacks import ProgressBar, CSVLogger
from torchtrainer.metrics import BinaryAccuracy
from torchtrainer.metrics.metric_container import MetricContainer
from tqdm import tqdm

from src.constants import MAX_LENGTH_USER_REPRESENATION, MAX_LENGTH_COMMENT_SECTION, \
    CHECKPOINT_SAVE_PATH
from src.dataloader.pairwise.utils import get_data_loader_node2vec_pairwise
from src.dataloader.pointwise.utils import get_data_loader_node2vec
from src.model.pairwise.graph_only import CommunityGraphModelPairwise
from src.model.pointwise.graph_only import CommunityGraphModel
from src.utils.utils import save_config, create_node2vec_embedding_layer
from train_model import flatten_section_emb_list
from utils import create_directory, train_node2vec


def print_divider():
    print('=' * 100)


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
    user_offsets = [0] + [1 for _ in batch['user_emb']]
    user_offsets = np.cumsum(np.array(user_offsets[:-1]))
    user_emb_offsets = torch.LongTensor(user_offsets).cuda()
    user_emb = batch['user_emb'].cuda().long()

    user_emb, user_emb_offsets = user_emb, user_emb_offsets
    section_emb, section_emb_offsets = tranform_section_emb(batch['section_emb'])

    return (user_emb, user_emb_offsets, section_emb,
            section_emb_offsets), y.float()


def transform_fn_pairwise(batch):
    y = batch['y'].cuda()
    user_offsets = [0] + [1 for _ in batch['user_emb']]
    user_offsets = np.cumsum(np.array(user_offsets[:-1]))
    user_emb_offsets = torch.LongTensor(user_offsets).cuda()
    user_emb = batch['user_emb'].cuda().long()

    user_emb, user_emb_offsets = user_emb, user_emb_offsets

    section_emb_pos, section_emb_offsets_pos = tranform_section_emb(batch['section_emb_pos'])
    section_emb_neg, section_emb_offsets_neg = tranform_section_emb(batch['section_emb_neg'])

    return (user_emb, user_emb_offsets, (section_emb_pos, section_emb_offsets_pos),
            (section_emb_neg, section_emb_offsets_neg)), y.float()


def start_train(
        gpu_ids=2,
        reduced_date_size=None,
        user_embedding_path=None,
        num_workers=2,
        learning_rate=0.005,
        num_epochs=20,
        batch_size=150,
        node2vec_workers=6,
        num_dimensions=25,
        walk_length=10,
        walks_per_source=20,
        context_size=10,
        epochs=5,
        layer_dim=16,
        p=1,
        q=1,
        pairwise=True,
        checkpoint_save_path=CHECKPOINT_SAVE_PATH
):
    path, folder_name = create_directory(checkpoint_save_path)

    if user_embedding_path is None:
        print_divider()
        user_embedding_path = train_node2vec(node2vec_workers, num_dimensions, walk_length, walks_per_source,
                                             context_size, epochs, p, q)
        print_divider()
    else:
        print('Using already trained user embeddings')
        print_divider()

    torch.cuda.empty_cache()

    save_config(os.path.join(path, 'parameter.config'),
                reduced_date_size=reduced_date_size,
                user_embedding_path=user_embedding_path,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                batch_size=batch_size,
                node2vec_workers=node2vec_workers,
                num_dimensions=num_dimensions,
                walk_length=walk_length,
                walks_per_source=walks_per_source,
                context_size=context_size,
                epochs=epochs,
                p=p,
                q=q,
                layer_dim=layer_dim)

    NODE2VEC_EMB_DIM, num_authors, node2vec_emb_layer, author_to_pos_dict = create_node2vec_embedding_layer(
        user_embedding_path, True)

    model = None

    if pairwise:
        model = CommunityGraphModelPairwise(node2vec_emb_layer, NODE2VEC_EMB_DIM)
    else:
        model = CommunityGraphModel(node2vec_emb_layer, NODE2VEC_EMB_DIM)

    config = {
        'max_length_user_rep': MAX_LENGTH_USER_REPRESENATION,
        'max_length_comment_section': MAX_LENGTH_COMMENT_SECTION,
        'layer_dim': layer_dim,
    }

    print(f'Selected cuda:{gpu_ids}')
    torch.cuda.set_device(gpu_ids)
    model.cuda()

    train_loader, val_loader = None, None
    if pairwise:
        _, val_loader = get_data_loader_node2vec_pairwise(author_to_pos_dict, batch_size, True, num_workers,
                                                                     reduced_date_size)
    else:
        _, val_loader = get_data_loader_node2vec(author_to_pos_dict, batch_size, True, num_workers,
                                                            reduced_date_size)

    loss = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = RAdam(model.parameters())

    trainer = TorchTrainer(model)

    validate_every = len(val_loader) // 2
    progress_bar = ProgressBar()
    progress_bar.on_train_begin()
    callbacks = [
        progress_bar,
        CSVLogger(file=os.path.join(path, 'epoch_log.csv')),
    ]

    metrics = [BinaryAccuracy()]

    trainer.prepare(optimizer, loss, train_loader, val_loader,
                    transform_fn=transform_fn if not pairwise else transform_fn_pairwise, callbacks=callbacks,
                    metrics=metrics, validate_every=validate_every)

    progress_bar.train_logs = {}
    progress_bar.train_logs['val_num_batches'] = len(val_loader)

    losses = AverageMeter('loss')
    validation_logs = {}

    metrics = MetricContainer(metrics)
    metrics.restart()

    for batch_idx, batch in enumerate(tqdm(val_loader)):
        batch_logs = {}
        y_pred, y_true, loss = trainer.val_loop(batch)

        losses.update(loss.item(), batch_size)
        batch_logs['loss'] = loss.item()
        batch_logs['running_loss'] = losses.avg

        batch_logs.update(metrics(y_pred, y_true))
        validation_logs.update(batch_logs)

    out_val_logs = {}
    for key, item in validation_logs.items():
        out_val_logs['val_' + key] = item
    result = out_val_logs
    print('Finished training with the following results:')
    print(result)

    result['out_acc'] = (result['val_binary_acc'], 0)
    return result


if __name__ == '__main__':
    fire.Fire(start_train)
