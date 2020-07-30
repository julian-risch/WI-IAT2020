import pandas as pd
import gensim
import fire
import os
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf
from gensim.models import Word2Vec

from colorama import init, Fore
init()


import sys
sys.path.append("../..")
from embeddings.utils import make_filename


# sg=1 -> skip-gram - sg=0 = cbow
def train(path_out, guardian=True, min_count=1, size=50, workers=6, window=3, sg=1, iter=10):
    train_df = None
    if guardian:
        train_df = pd.read_csv('/mnt/data/datasets/newspapers/guardian/train_test/train_all.csv')
    else:
        train_df = pd.read_csv('/mnt/data/datasets/newspapers/daily-mail/train_test/train_all.csv')

    # sort by timestamp
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    train_df = train_df.sort_values(by='timestamp')
    print(Fore.GREEN + 'Finished sorting train data ')

    train_data = train_df.groupby('article_id')['author_id'].apply(list)
    train_data = list(train_data.reset_index()['author_id'])
    s_train_data = [[str(n) for n in row] for row in train_data]

    data = "guardian" if guardian else "daily-mail"

    filename = make_filename(filename_base=data, min_count=min_count, size=size, window=window, sg=sg, iter=iter)

    model = gensim.models.Word2Vec(s_train_data, min_count=min_count, size=size, workers=workers, window=window, sg=sg, iter=iter)
    model.wv.save_word2vec_format(os.path.join(path_out, filename + '.txt'), binary=False)
    model.save(os.path.join(path_out, filename + '.model'))
    print(Fore.GREEN + 'Model saved to: ' + Fore.WHITE + os.path.join(path_out, filename + '.model'))


def train_all(path_out):
    for sg in [0, 1]:
        for size in [20, 50, 100, 200]:
            for window in [1, 3, 6]:
                for iter in [5, 10, 50]:
                    print(15 * '-')
                    print(Fore.BLUE + 'Start training model with:')
                    print('Type: ' + ('skip-gram' if sg == 1 else 'cbow'))
                    print('Size: ', size)
                    print('Window: ', window)
                    print('Iterations: ', iter)

                    train(path_out, size=size, window=window, iter=iter, sg=sg)

                    print(Fore.GREEN, 'FINISHED')
    print(Fore.YELLOW + 'Finished!')


def visualize_projector(model_path, output_path):
    meta_name = os.path.basename(model_path)
    meta_file = meta_name + '.tsv'
    data = meta_file.split('_')[0]
    embedding_size = meta_file.split('size')[1].split('_')[0].replace('-', '')

    model = Word2Vec.load(model_path)

    placeholder = np.zeros((len(model.wv.index2word), int(embedding_size)))

    user_categories_df = pd.read_csv('~/jp-data-analysis/data/user_categories/user_most_commented_category.csv')
    user_categories_df.index = user_categories_df['author_id']
    user_categories_df = user_categories_df[['category']]

    if data == 'guardian':
        user_categories_df['category'] = user_categories_df['category'].replace(
             ['weather', 'community', 'inequality', 'media_network', 'careers', 'woman-in-leadership', 'law', 'membership', 'news'], 'different')

    print(f'Number of categories: {str(user_categories_df["category"].nunique())}')

    user_categories = user_categories_df.to_dict()['category']

    with open(os.path.join(output_path, meta_file), 'wb') as file_metadata:
        file_metadata.write(f'word\tcategory'.encode('utf-8') + b'\n')
        for i, word in enumerate(model.wv.index2word):
            placeholder[i] = model[word]
            if word == '':
                print("Empty Line, should repleced by any thing else, or will cause a bug of tensorboard")
                file_metadata.write("{0}".format('<Empty Line>').encode('utf-8') + b'\n')
            else:
                file_metadata.write(f'{word}\t{user_categories[int(word)]}'.encode('utf-8') + b'\n')

    # define the model without training
    sess = tf.InteractiveSession()

    embedding = tf.Variable(placeholder, trainable=False, name=meta_name)
    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(output_path, sess.graph)

    # adding into projector
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = meta_name
    embed.metadata_path = meta_file

    # Specify the width and height of a single thumbnail.
    projector.visualize_embeddings(writer, config)
    saver.save(sess, os.path.join(output_path, meta_name + '.ckpt'))
    print('Run `tensorboard --logdir={0}` to run visualize result on tensorboard'.format(output_path))


if __name__ == '__main__':
    fire.Fire()
