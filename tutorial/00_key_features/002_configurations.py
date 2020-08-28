"""
.. _configurations:

Pythonic Search Space
=====================

Defining Parameter Spaces
-------------------------

Optuna supports three kinds of parameters:
* Categorical parameters
* Integer parameters
* Floating point parameters
. Integer and floating point parameters can be either discretized or in log domain.
"""


import optuna


def objective(trial):
    # Categorical parameter
    optimizer = trial.suggest_categorical("optimizer", ["MomentumSGD", "Adam"])

    # Integer parameter
    num_layers = trial.suggest_int("num_layers", 1, 3)

    # Integer parameter (log)
    num_channels = trial.suggest_int("num_channels", 32, 512, log=True)

    # Integer parameter (discretized)
    num_units = trial.suggest_int("num_units", 10, 100, step=5)

    # Floating point parameter
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 1.0)

    # Floating point parameter (log)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    # Floating point parameter (discretized)
    drop_path_rate = trial.suggest_float("drop_path_rate", 0.0, 1.0, step=0.1)


###################################################################################################
# Branches and Loops
# ------------------
#
# You can use branches or loops depending on the parameter values.
#
# For more various use, see `examples <https://github.com/optuna/optuna/tree/master/examples>`_

###################################################################################################
# Branches:
import sklearn.ensemble
import sklearn.svm


def objective(trial):
    classifier_name = trial.suggest_categorical("classifier", ["SVC", "RandomForest"])
    if classifier_name == "SVC":
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
        classifier_obj = sklearn.svm.SVC(C=svc_c)
    else:
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        classifier_obj = sklearn.ensemble.RandomForestClassifier(max_depth=rf_max_depth)


###################################################################################################
# Loops:
import torch
import torch.nn as nn


def create_model(trial, in_size):
    n_layers = trial.suggest_int("n_layers", 1, 3)

    layers = []
    for i in range(n_layers):
        n_units = trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True)
        layers.append(nn.Linear(in_size, n_units))
        layers.append(nn.ReLU())
        in_size = n_units
    layers.append(nn.Linear(in_size, 10))

    return nn.Sequential(*layers)


###################################################################################################
# Note on the Number of Parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The difficulty of optimization increases roughly exponentially with regard to the number of parameters. That is, the number of necessary trials increases exponentially when you increase the number of parameters, so it is recommended to not add unimportant parameters.


###################################################################################################
# Arguments for `Study.optimize`
# --------------------------------
#
# The method :func:`~optuna.study.Study.optimize` (and ``optuna study optimize`` CLI command as well)
# has several useful options such as ``timeout``.
# For details, please refer to the API reference for :func:`~optuna.study.Study.optimize`.
#
# **FYI**: If you give neither ``n_trials`` nor ``timeout`` options, the optimization continues until it receives a termination signal such as Ctrl+C or SIGTERM.
# This is useful for use cases such as when it is hard to estimate the computational costs required to optimize your objective function.
