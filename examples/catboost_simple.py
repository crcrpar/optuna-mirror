"""
Optuna example that optimizes a classifier configuration for cancer dataset using CatBoost.

In this example, we optimize the validation accuracy of cancer detection using CatBoost.
We optimize both the choice of booster model and their hyperparameters.

We have following two ways to execute this example:

(1) Execute this code directly.
    $ python catboost_simple.py


(2) Execute through CLI.
    $ STUDY_NAME=`optuna create-study --storage sqlite:///example.db`
    $ optuna study optimize catboost_simple.py objective --n-trials=100 --study $STUDY_NAME \
      --storage sqlite:///example.db

"""

from catboost import CatBoostClassifier, Pool
import numpy as np
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split

import optuna


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):
    data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.25)

    param = {
        'iterations': 10,
        'objective': 'binary',
        # 'custom_metric': 'Logloss',
        'verbose': False,
        'boosting_type': trial.suggest_categorical('boosting', ['Ordered', 'Plain']),
        'max_leaves': trial.suggest_int('max_leaves', 10, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-8, 1.0)
    }

    model = CatBoostClassifier(**param)
    model.fit(train_x, train_y)
    pred_labels = model.predict(test_x)
    accuracy = sklearn.metrics.accuracy_score(test_y, pred_labels)
    return 1.0 - accuracy


if __name__ == '__main__':
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)

    print('Number of finished trials: {}'.format(len(study.trials)))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
