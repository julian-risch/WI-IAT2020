#%%
import pandas as pd
import os
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline
import plotly_express as px

#%%
model_path = '~/baselines/embeddings/word2vec/'
model_name = 'guardian_min_count-1_size-20_window-5_sg-0_iter-50.txt'
out_dir = './baselines/embeddings/word2vec'
data_slice = None

#%%
def save_to_pdf(filename, ax):
    fig = ax.get_figure()
    fig.savefig(filename, bbox_inches='tight')

#%%
model = KeyedVectors.load_word2vec_format(os.path.join(model_path, model_name))

#%%
vocab = list(model.vocab)
X = model[vocab][:data_slice]

#%%
tsne = TSNE(n_components=2)
X_embedded = tsne.fit_transform(X)

#%%
df = pd.DataFrame(X_embedded, index=vocab[:data_slice], columns=['x', 'y'])

#%%
df_user_categories = pd.read_csv('~/jp-data-analysis/data/user_categories/user_most_commented_category.csv')

#%%
df['author_id'] = df.index

#%%
df.author_id = df.author_id.astype('int64') 

#%%
df_visualize = df.merge(df_user_categories, on='author_id', how='left')

# %%
# make out directory
filename = model_name.replace('.txt', '').replace('.', '-') + '_visualization_'

#%%
sns.set_palette("pastel")
ax = sns.scatterplot(x="x", y="y", hue="category", data=df_visualize, alpha=0.6)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
path_to_file = os.path.join(out_dir, filename + 'seaborn.pdf')
save_to_pdf(path_to_file, ax)

#%%
figure = px.scatter(df_visualize, x="x", y="y", color="category")
figure

#%%
path_to_file = os.path.join(out_dir, filename + 'plotly.html')
plotly.offline.plot(figure, filename=path_to_file)
