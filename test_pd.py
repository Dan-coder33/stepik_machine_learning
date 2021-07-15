# %%
import pandas as pd
import numpy as np

# %%
students_performance = pd.read_csv(
    'data/StudentsPerformance.csv'
)

# %%
students_performance.describe()

# %%
types = students_performance.dtypes

# %%
slc_1 = students_performance.iloc[:3, :3]
slc_2 = students_performance.iloc[
    [0, 3, 10], [0, 5, -1]
]

# %%
students_performance_with_names = students_performance.iloc[:5]
students_performance_with_names.index = [
    'Cersei', 'Tuwin', 'Gregor', 'Joffrey', 'Payne'
]

# %%
slc_3 = students_performance_with_names.loc[['Cersei', 'Tuwin']]

# %%
titanic = pd.read_csv('data/titanic.csv')

# %%
titanic.dtypes

# %%
female_students = students_performance[
    students_performance.gender == 'female'
    ]

# %%
students_performance.lunch[
    students_performance.lunch == 'free/reduced'
    ].count() / students_performance.lunch.count()

# %%
students_performance_reduced = students_performance[
    students_performance.lunch == 'free/reduced'
    ]

students_performance_standard = students_performance[
    students_performance.lunch == 'standard'
    ]

# %%
students_performance_reduced_std = students_performance_reduced.describe().std()
students_performance_standard_std = students_performance_standard.describe().std()

# %%
std_per_lunch = pd.DataFrame(
    [students_performance_reduced_std,
     students_performance_standard_std],
    index=['free/reduced', 'standard']
)

# %%
students_performance_reduced = students_performance_reduced.describe()
students_performance_standard = students_performance_standard.describe()

# %%
students_performance_reduced = students_performance[
    students_performance.lunch == 'free/reduced'
    ]

# %%
students_performance_reduced.describe()

# %%
students_performance_standard = students_performance[
    students_performance.lunch == 'standard'
    ]

# %%
students_performance_standard.describe()

# %%
students_performance.filter(like='score')

# %%
dota = pd.read_csv('data/dota_hero_stats.csv')

# %%
dota[dota.legs == 8].count()

# %%
loopa_and_poopa = pd.read_csv('data/accountancy.csv')

# %%
res = loopa_and_poopa.groupby(['Executor', 'Type']).agg(
    {'Salary': 'mean'}
)

# %%
dota.groupby(['attack_type', 'primary_attr']).agg(
    {'id': 'count'}
)

# %%
algae = pd.read_csv('data/algae.csv')

# %%
mean_concentrations = algae.groupby('genus').mean()

# %%
d = algae[algae.genus == 'Fucus'].groupby('genus')['alanin'].describe().min()

# %%
print(' '.join([
    str(d.min().round(2)),
    str(d.mean().round(2)),
    str(d.max().round(2)),
]))

# %%

