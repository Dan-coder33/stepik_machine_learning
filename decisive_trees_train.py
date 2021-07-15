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
from sklearn.metrics import precision_score, recall_score
from sklearn.datasets import load_iris

# %%
data = pd.read_csv('data/train_data_tree.csv')

# %%
clf = tree.DecisionTreeClassifier()

# %%
X_train = data.drop(['num'], axis=1)
y_train = data.num

# %%
params = {
    'criterion': ['entropy'],
    'max_depth': range(1, 30)
}

# %%
grid_search_cv_clf = GridSearchCV(clf, params, cv=5)

# %%
grid_search_cv_clf.fit(X_train, y_train)

# %%
best_clf = grid_search_cv_clf.best_estimator_

# %%
data.num.value_counts()

# %%
tree.plot_tree(best_clf, filled=True)
plt.show()

# %%
# Ирисы
iris = load_iris()
X = iris.data
y = iris.target

# %%
# Первое задание
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# %%
dt = tree.DecisionTreeClassifier()

# %%
dt.fit(X_train, y_train)

# %%
predicted = dt.predict(X_test)

# %%
# Второе задание
clf = tree.DecisionTreeClassifier()
params = {
    'max_depth': range(1, 11),
    'min_samples_split': range(2, 11),
    'min_samples_leaf': range(1, 11)
}

# %%
search = GridSearchCV(clf, params, cv=5)

# %%
search.fit(X, y)

# %%
best_clf = search.best_estimator_
