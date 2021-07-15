# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import HTML
from IPython.display import SVG
from IPython.display import display
from graphviz import Source
from sklearn import tree
from sklearn.metrics import precision_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

style = "<style>svg{width: 70% !important; height: 60% !important;} </style>"
HTML(style)

# %%
titanic_data = pd.read_csv('data/train.csv')

# %%
titanic_data.isnull().sum()

# %%
X = titanic_data.drop([
    'PassengerId', 'Survived',
    'Name', 'Ticket', 'Cabin'
], axis=1)

# %%
y = titanic_data.Survived

# %%
X = pd.get_dummies(X)

# %%
X = X.fillna({'Age': X.Age.median()})

# %%
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(X, y)

# %%
# что-то пошло не так (overfitting)
graph = Source(tree.export_graphviz(
    clf, out_file=None,
    feature_names=list(X),
    class_names=['Died', 'Survived'],
    filled=True
))
display(SVG(graph.pipe(format='svg')))

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# %%
clf.score(X, y)

# %%
clf.fit(X_train, y_train)

# %%
clf.score(X_train, y_train)

# %%
clf.score(X_test, y_test)

# %%
clf = tree.DecisionTreeClassifier(
    criterion='entropy',
    max_depth=3
)

# %%
clf.fit(X_train, y_train)

# %%
clf.score(X_train, y_train)

# %%
clf.score(X_test, y_test)

# %%
max_depth_values = range(1, 100)
scores_data = pd.DataFrame()

for max_depth in max_depth_values:
    clf = tree.DecisionTreeClassifier(
        criterion='entropy',
        max_depth=max_depth
    )
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)

    mean_cross_val_score = cross_val_score(
        clf, X_train, y_train, cv=5
    ).mean()

    temp_score_data = pd.DataFrame({
        'max_depth': [max_depth],
        'train_score': [train_score],
        'test_score': [test_score],
        'cross_val_score': [mean_cross_val_score]
    })

    scores_data = scores_data.append(temp_score_data)

# %%
scores_data_long = pd.melt(
    scores_data, id_vars=['max_depth'],
    value_vars=['train_score', 'test_score', 'cross_val_score'],
    var_name='set_type', value_name='score'
)

# %%
scores_data_long.query('set_type == "cross_val_score"').head(20)

# %%
sns.lineplot(
    x='max_depth',
    y='score',
    hue='set_type',
    data=scores_data_long
)
plt.show()

# %%
clf = tree.DecisionTreeClassifier(
    criterion='entropy', max_depth=10
)

# %%
cross_val_score(clf, X_train, y_train, cv=5).mean()

# %%
cross_val_score(clf, X_test, y_test, cv=5).mean()

# %%
# Iris
iris_train_data = pd.read_csv('data/train_iris.csv')
iris_test_data = pd.read_csv('data/test_iris.csv')

# %%
X_train = iris_train_data.drop([
    'species', 'Unnamed: 0'
], axis=1)

# %%
y_train = iris_train_data.species

# %%
X_test = iris_test_data.drop([
    'species', 'Unnamed: 0'
], axis=1)
y_test = iris_test_data.species


# %%
def depth_optimization(X_train, y_train):
    max_depth_values = range(1, 100)
    scores_data = pd.DataFrame()

    for max_depth in max_depth_values:
        clf = tree.DecisionTreeClassifier(
            criterion='entropy',
            max_depth=max_depth
        )
        clf.fit(X_train, y_train)
        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)

        mean_cross_val_score = cross_val_score(
            clf, X_train, y_train, cv=5
        ).mean()

        temp_score_data = pd.DataFrame({
            'max_depth': [max_depth],
            'train_score': [train_score],
            'test_score': [test_score],
            'cross_val_score': [mean_cross_val_score]
        })

        scores_data = scores_data.append(temp_score_data)

    return scores_data


# %%
def depth_optimization_plot(scores_data):
    scores_data_long = pd.melt(
        scores_data, id_vars=['max_depth'],
        value_vars=['train_score', 'test_score', 'cross_val_score'],
        var_name='set_type', value_name='score'
    )

    sns.lineplot(
        x='max_depth',
        y='score',
        hue='set_type',
        data=scores_data_long
    )
    plt.show()

    return scores_data_long.query(
        'set_type == "cross_val_score"'
    )


# %%
# Котики и собачки
dogs_n_cats = pd.read_csv('data/dogs_n_cats.csv')

# %%
dogs_n_cats = dogs_n_cats.rename(columns={
    'Длина': 'length',
    'Высота': 'height',
    'Шерстист': 'woolly',
    'Гавкает': 'barks',
    'Лазает по деревьям': 'climbing trees',
    'Вид': 'kind'
})

# %%
X_train = dogs_n_cats.drop([
    'kind'
], axis=1)
y_train = dogs_n_cats.kind

# %%
max_depth_values = range(1, 100)
scores_data = pd.DataFrame()

for max_depth in max_depth_values:
    clf = tree.DecisionTreeClassifier(
        criterion='entropy',
        max_depth=max_depth
    )
    clf.fit(X_train, y_train)

    mean_cross_val_score = cross_val_score(
        clf, X_train, y_train, cv=5
    ).mean()

    temp_score_data = pd.DataFrame({
        'max_depth': [max_depth],
        'cross_val_score': [mean_cross_val_score]
    })

    scores_data = scores_data.append(temp_score_data)

# %%
scores_data.head(20)

# %%
sns.lineplot(
    x='max_depth',
    y='cross_val_score',
    data=scores_data
)
plt.show()

# %%
clf = tree.DecisionTreeClassifier(
    criterion='entropy',
    max_depth=1
)
clf.fit(X_train, y_train)

# %%
test = pd.read_json('data/dataset_209691_15.txt')

# %%
np.unique(clf.predict(test), return_counts=True)

# %%
# Task
songs = pd.read_csv('data/songs.csv')

# %%
X = songs.drop([
    'artist', 'lyrics',
    'genre', 'song'
], axis=1)

y = songs.artist

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# %%
scores_data = depth_optimization(X_train, y_train)

# %%
scores_data_long = depth_optimization_plot(scores_data)

# %%
scores_data_long.query(
    'score == score.max()'
)

# %%
clf = tree.DecisionTreeClassifier(
    criterion='entropy',
    max_depth=4
)
clf.fit(X_train, y_train)

# %%
predictions = clf.predict(X_test)

# %%
precision = precision_score(
    y_test, predictions, average='micro'
)

# %%
clf = tree.DecisionTreeClassifier()
params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': range(1, 30)
}

# %%
grid_search_cv_clf = GridSearchCV(clf, params, cv=5)

# %%
grid_search_cv_clf.fit(X_train, y_train)

# %%
grid_search_cv_clf.best_params_

# %%
best_clf = grid_search_cv_clf.best_estimator_

# %%
best_clf.score(X_test, y_test)

# %%
from sklearn.metrics import precision_score, recall_score

# %%
y_pred = best_clf.predict(X_test)

# %%
precision_score(y_test, y_pred, average='macro')

# %%
recall_score(y_test, y_pred, average='macro')

# %%
y_predicted_prob = best_clf.predict_proba(X_test)

# %%
pd.Series(y_predicted_prob[:, 0]).hist()
plt.show()

# %%

