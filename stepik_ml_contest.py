# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# %%
events_data = pd.read_csv('data/event_data_train.csv')

# %%
submissions_data = pd.read_csv('data/submissions_data_train.csv')

# %%
events_data.action.unique()

# %%
events_data['date'] = pd.to_datetime(
    events_data.timestamp,
    unit='s'
)

# %%
events_data.dtypes

# %%
events_data['day'] = events_data.date.dt.date

# %%
submissions_data['date'] = pd.to_datetime(
    submissions_data.timestamp,
    unit='s'
)
submissions_data['day'] = submissions_data.date.dt.date

# %%
# Task
submissions_data[
    submissions_data.submission_status == 'wrong'
].groupby('step_id').agg(
    {'submission_status': 'count'}
).sort_values('submission_status')

# %%
# Continue
events_data.groupby('day').user_id.nunique().plot()
plt.show()

# %%
events_data[events_data.action == 'passed'].groupby(
    'user_id', as_index=False
).agg({'step_id': 'count'}).rename(
    columns={'step_id': 'passed_steps'}
).passed_steps.hist()
plt.show()

# %%
events_data.pivot_table(
    index='user_id',
    columns='action',
    values='step_id',
    aggfunc='count',
    fill_value=0
).discovered.hist()
plt.show()

# %%
users_events_data = events_data.pivot_table(
    index='user_id',
    columns='action',
    values='step_id',
    aggfunc='count',
    fill_value=0
).reset_index()

# %%
users_scores = submissions_data.pivot_table(
    index='user_id',
    columns='submission_status',
    values='step_id',
    aggfunc='count',
    fill_value=0
)
plt.show()

# %%
randomizers = users_scores[
    (users_scores.wrong > 5 * users_scores.correct) &
    (users_scores.wrong > 700)
    ]

# %%
users_scores['sum'] = users_scores.wrong + users_scores.correct

# %%
users_scores.sort_values('sum')

# %%
gap_data = events_data[
    ['user_id', 'day', 'timestamp']
].drop_duplicates(
    subset=['user_id', 'day']
).groupby('user_id')['timestamp'].apply(
    list
).apply(np.diff).values

# %%
gap_data = pd.Series(np.concatenate(gap_data, axis=0))

# %%
gap_data = gap_data / (24 * 60 * 60)

# %%
gap_data[gap_data < 125].hist()
plt.show()

# %%
gap_data.quantile(0.90)

# %%
users_data = events_data.groupby(
    'user_id', as_index=False
).agg(
    {'timestamp': 'max'}
).rename(columns={
    'timestamp': 'last_timestamp'
})

# %%
now = 1526772811
drop_out_threshold = 30*24*60*60

# %%
users_data['is_gone_user'] = (now - users_data.last_timestamp) > drop_out_threshold

# %%
users_data = users_data.merge(
    users_scores, on='user_id', how='outer'
)

# %%
users_data = users_data.fillna(0)

# %%
users_data = users_data.merge(
    users_events_data,
    how='outer'
)

# %%
users_days = events_data.groupby(
    'user_id'
).day.nunique().to_frame().reset_index()

# %%
users_data = users_data.merge(
    users_days,
    how='outer'
)

# %%
users_data.user_id.nunique()

# %%
events_data.user_id.nunique()

# %%
users_data['passed_course'] = users_data.passed > 170

# %%
users_data.groupby('passed_course').count()

# %%


# %%

