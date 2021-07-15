# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# %%
students_performance = pd.read_csv(
    'data/StudentsPerformance.csv'
)

# %%
students_performance['math score'].hist()
plt.show()

# %%
students_performance.plot.scatter(
    x='math score',
    y='reading score'
)
plt.show()

# %%
sns.lmplot(
    x='math score',
    y='reading score',
    hue='gender',
    data=students_performance
)
plt.show()

# %%
df = pd.read_csv('data/income.csv')

# %%
df.plot(kind='line')
plt.show()

# %%
df.income.plot()
plt.show()

# %%
plt.plot(df.index, df.income)
plt.show()

# %%
df.plot()
plt.show()

# %%
tst = pd.read_csv('data/dataset_209770_6.txt', sep=' ')

# %%
sns.scatterplot(
    x='x',
    y='y',
    data=tst
)
plt.show()

# %%
genome = pd.read_csv('data/genome_matrix.csv')

# %%
genome = genome.set_index('Unnamed: 0')

# %%
g = sns.heatmap(
    data=genome,
    cmap='viridis',
)
g.xaxis.set_ticks_position('top')

plt.show()

# %%
dota = pd.read_csv('data/dota_hero_stats.csv')

# %%
dota['number_of_roles'] = dota.roles.apply(
    lambda x: len(x.split(', '))
)

# %%
sns.countplot(
    x=dota.number_of_roles
)
plt.show()

# %%
iris_df = pd.read_csv('data/iris.csv')

# %%
for column in iris_df[[
    'sepal length',
    'sepal width',
    'petal length',
    'petal width'
]]:
    sns.kdeplot(x=column, data=iris_df)
    plt.show()

# %%
sns.kdeplot(x=iris_df['sepal length'])
plt.show()

# %%
sns.violinplot(
    x=iris_df['petal length']
)
plt.show()

# %%
sns.pairplot(data=iris_df)
plt.show()

# %%

