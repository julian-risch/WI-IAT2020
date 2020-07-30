import fire
import pandas as pd
import os
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline
import plotly_express as px


def save_to_pdf(filename, ax):
    fig = ax.get_figure()
    fig.savefig(filename, bbox_inches='tight')


def get_X(model_path):
    model = KeyedVectors.load_word2vec_format(model_path)
    vocab = list(model.vocab)
    X = model[vocab]
    return vocab, X


def get_embedded_X(X, perplexity, n_iter, verbose):
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, verbose=verbose)
    return tsne.fit_transform(X)


def visualize_tsne(model_path, out_dir, data='guardian', data_slice=None, perplexity=30, n_iter=1000, verbose=0):
    model_name = model_path.split('/')[-1]
    model_path = os.path.expanduser(model_path)

    vocab, X = get_X(model_path)
    X = X[:data_slice]
    vocab = vocab[:data_slice]
    X_embedded = get_embedded_X(X, perplexity, n_iter, verbose)

    df = pd.DataFrame(X_embedded, index=vocab[:data_slice], columns=['x', 'y'])
    df_user_categories = pd.read_csv('~/jp-data-analysis/data/user_categories/user_most_commented_category.csv')
    df['author_id'] = df.index
    df.author_id = df.author_id.astype('int64')
    df_visualize = df.merge(df_user_categories, on='author_id', how='left')

    filename = model_name.replace('.txt', '').replace('.', '-') + '_visualization_'

    # make seaborn visualization
    sns.set_palette("pastel")
    ax = sns.scatterplot(x="x", y="y", hue="category", data=df_visualize, alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    path_to_file = os.path.join(out_dir, filename + 'seaborn.pdf')
    save_to_pdf(os.path.expanduser(path_to_file), ax)
    print('PDF Visualization can be found here: ', path_to_file)

    # make plotly visualization
    figure = px.scatter(df_visualize, x="x", y="y", color="category")
    path_to_file = os.path.join(out_dir, filename + 'plotly.html')
    plotly.offline.plot(figure, filename=os.path.expanduser(path_to_file))

    print('Interactive Visualization can be found here: ', path_to_file)


if __name__ == '__main__':
    fire.Fire(visualize_tsne)
