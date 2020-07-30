import pandas as pd
import gensim
import fire
import os

train_df = None

def make_filename(guardian, min_count, size, window, sg):
    data = "guardian" if guardian else "dailymail"
    return '{data}_{min_count}_{size}_{window}_{sg}.txt'

def train(path_out,guardian=True, min_count=1, size=50, workers=6, window=3, sg=1):
    train_df = None
    if guardian:
        train_df = pd.read_csv('/mnt/data/datasets/newspapers/guardian/train_test/train_all.csv')
    else:
        train_df = pd.read_csv('/mnt/data/datasets/newspapers/daily-mail/train_test/train_all.csv')
    
    train_data = train_df.groupby('article_id')['author_id'].apply(list)
    train_data = list(train_data.reset_index()['author_id'])
    s_train_data = [[str(n) for n in row] for row in train_data]

    model = gensim.models.Word2Vec(s_train_data, min_count,size, workers, window, sg)
    model.save_word2vec_format(os.path.join(path_out, make_filename(guardian, min_count,size, window, sg)), binary=False)

if __name__ == '__main__':
  fire.Fire(train)
