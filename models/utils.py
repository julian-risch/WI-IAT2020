import datetime
import os
import subprocess

from src.constants import NODE2VEC_SAVE_DIR, NODE2VEC_EXECUTABLE_PATH, NODE2VEC_GRAPH_INPUT,


def train_node2vec(num_workers=4, num_dimensions=64, walk_length=80, walks_per_source=10, context_size=10, epochs=10,
                   p=1, q=1):
    filename = f'graph_dim_{num_dimensions}_walk_length_{walk_length}_wps_{walks_per_source}_context_{context_size}_epochs_{epochs}_p_{p}_q_{q}.emb'
    out_path = os.path.join(NODE2VEC_SAVE_DIR, filename)
    workers = ','.join(str(x) for x in list(range(num_workers)))
    print('Threads selected: ', workers)
    if not os.path.isfile(out_path):
        subprocess.call(
            ['taskset', '-c', workers,
             NODE2VEC_EXECUTABLE_PATH,
             f'-i:{NODE2VEC_GRAPH_INPUT}', f'-o:{out_path}', f'-d:{num_dimensions}', f'-l:{walk_length}',
             f'-r:{walks_per_source}', f'-k:{context_size}', f'-e:{epochs}', f'-p:{p}', f'-q:{q}', '-v'], )
    else:
        print('Already trained')
    return out_path


def create_directory(checkpoint_save_path):
    folder_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    path = os.path.join(checkpoint_save_path, folder_name)
    os.makedirs(path)
    return path, folder_name
