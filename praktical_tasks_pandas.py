# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# %%
my_data = pd.DataFrame(
    {
        'type': ['A', 'A', 'B', 'B'],
        'value': [10, 14, 12, 23]
    }
)

# %%
my_stat = pd.read_csv('data/my_stat.csv')

# %%
subset_1 = my_stat.iloc[:10, [0, 2]]

# %%
subset_2 = my_stat.iloc[
    np.logical_not(my_stat.index.isin([0, 4])), [1, 3]
]

# %%
subset_1 = my_stat[(my_stat.V1 > 0) & (my_stat.V3 == 'A')]

# %%
subset_2 = my_stat[(my_stat.V2 != 10) | (my_stat.V4 >= 1)]

# %%
my_stat['V5'] = my_stat.V1 + my_stat.V4

# %%
my_stat['V6'] = np.log(my_stat.V2)

# %%
my_stat = my_stat.rename(columns={
    'V1': 'session_value',
    'V2': 'group',
    'V3': 'time',
    'V4': 'n_users'
})

# %%
my_stat = pd.read_csv('data/my_stat_1.csv')

# %%
my_stat = my_stat.fillna(0)

# %%
median = my_stat.n_users[my_stat.n_users > 0].median()

# %%
my_stat.n_users = my_stat.n_users.apply(
    lambda x: median if x < 0 else x
)

# %%
mean_session_value_data = my_stat.groupby(
    'group', as_index=False
).agg(
    {'session_value': 'mean'}
).rename(columns={'session_value': 'mean_session_value'})
