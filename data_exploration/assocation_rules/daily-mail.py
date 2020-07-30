#%%
import mlxtend
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd

#%%
df = pd.read_csv('/mnt/data/datasets/newspapers/daily-mail/train_test/user_categories.csv')

#%%
df[df > 0] = 1

#%%
df

#%%
min_support = 0.05
min_threshold = 6
metric = 'lift'
data = 'daily-mail'

#%%
frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

#%%
rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)

#%%
rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x)).astype("unicode")
rules["consequents"] = rules["consequents"].apply(lambda x: list(x)).astype("unicode")

#%%
rules.sort_values(by='confidence', ascending=False).to_csv('~/jp-data-analysis/data/assocation_rule/' + data + '_' + metric + '_' + str(min_support).replace(".", "-") + '_' + str(min_threshold) + '.csv', index=False)

#%%
rules.describe()

#%%
