# %%
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#%%
G = nx.read_graphml('jp-data-analysis/data/graph_guardian_min_two_occ_max_10.graphml')

#%%
print(nx.info(G))

#%%
degree = sorted(G.degree, key=lambda x: x[1], reverse=True)

#%%
degree[0]

#%%
degree[-8000]

#%%
